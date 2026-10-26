from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

import numpy as np
import torch

GROUP_RULES = [
    (re.compile(r"^arc\.net_[uv]\."), "kernel"),
    (re.compile(r"^arc\."), "arc_emb"),
    (re.compile(r"^message\.(pool|ln_pool)"), "attn_pool"),
    (re.compile(r"^message\.(netpool|ctx_proj)"), "net_pool"),
    (re.compile(r"^message\.sab"), "sab"),
    (re.compile(r"^message\.log_tau"), "soft_tau"),
    (re.compile(r"^message\."), "message"),
    (re.compile(r"^encoder\.(layers|norms)\.(\d+)\."), None),
    (re.compile(r"^encoder\.(pin_head)"), "pin_head"),
    (re.compile(r"^encoder\.(arc_hidden|arc_out|emb_pair)"), "arc_head"),
    (re.compile(r"^encoder\."), "node_enc"),
]
GAP_EDGES = [0.0, 1e-9, 0.01, 0.1, 1.0, 10.0, math.inf]
GAP_LABELS = ["exact tie", "(0,0.01)", "[0.01,0.1)", "[0.1,1)", "[1,10)", ">=10 ps"]


def group_of(name: str) -> str:
    for rx, g in GROUP_RULES:
        m = rx.match(name)
        if m:
            return g if g is not None else f"layer{m.group(2)}"
    return "other"


def gap_band(gap: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(GAP_EDGES, gap, side="right") - 1, 0, len(GAP_LABELS) - 1)


class GradMonitor:

    def __init__(self, model, every: int = 20, explode_factor: float = 10.0, window: int = 50):
        self.params = [(n, p, group_of(n)) for n, p in model.named_parameters() if p.requires_grad]
        self.groups = sorted({g for _, _, g in self.params}, key=lambda g: (not g.startswith("layer"), g))
        self.every, self.explode_factor = every, explode_factor
        self.recent = collections.deque(maxlen=window)
        self.reset()

    def reset(self):
        self.norms, self.max_batch, self.n_nonfinite, self.n_explode = [], ("", 0.0), 0, 0
        self.group_sum, self.group_n = {g: 0.0 for g in self.groups}, 0

    def group_norms(self) -> dict:
        out = {}
        for g in self.groups:
            sq = 0.0
            for _, p, gg in self.params:
                if gg == g and p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            out[g] = math.sqrt(sq)
        return out

    def record(self, step: int, total_norm: float, names: list, force_groups: bool = False) -> dict:
        finite = math.isfinite(total_norm)
        r = {"step": step, "total": total_norm, "finite": finite, "explode": False, "groups": None}
        if not finite:
            self.n_nonfinite += 1
            return r
        med = float(np.median(self.recent)) if len(self.recent) >= 10 else math.inf
        if total_norm > self.explode_factor * med:
            self.n_explode += 1
            r["explode"] = True
        self.recent.append(total_norm)
        self.norms.append(total_norm)
        if total_norm > self.max_batch[1]:
            self.max_batch = (" ".join(names[:4]), total_norm)
        if force_groups or (self.every and step % self.every == 0):
            r["groups"] = self.group_norms()
            for g, v in r["groups"].items():
                self.group_sum[g] += v
            self.group_n += 1
        return r

    def epoch_row(self) -> dict:
        a = np.asarray(self.norms) if self.norms else np.zeros(1)
        row = {"grad_mean": float(a.mean()), "grad_max": float(a.max()), "grad_p99": float(np.quantile(a, 0.99)),
               "grad_max_batch": self.max_batch[0], "nonfinite": self.n_nonfinite, "explode": self.n_explode}
        for g in self.groups:
            row[f"g_{g}"] = self.group_sum[g] / self.group_n if self.group_n else float("nan")
        self.reset()
        return row


class StatMonitor:

    def __init__(self, model):
        self.sources = [m for m in (getattr(model, "message", None), model.encoder) if m is not None and hasattr(m, "pop_stats")]
        self.sum, self.n = collections.defaultdict(float), collections.defaultdict(int)

    def arm(self, on: bool):
        for m in self.sources:
            m.monitor = on

    def collect(self):
        for m in self.sources:
            for k, v in m.pop_stats().items():
                if math.isfinite(v):
                    self.sum[k] += v; self.n[k] += 1

    def epoch_row(self) -> dict:
        row = {k: self.sum[k] / self.n[k] for k in sorted(self.sum)}
        self.sum.clear(); self.n.clear()
        return row


class Diag:

    def __init__(self, run_dir: str, model, every: int = 20, explode_factor: float = 10.0, append: bool = False):
        self.run_dir, self.every = run_dir, every
        self.grads = GradMonitor(model, every, explode_factor)
        self.stats = StatMonitor(model)
        tsv = os.path.join(run_dir, "diag.tsv")
        self.cols = None
        if append and os.path.exists(tsv):
            with open(tsv) as f:
                header = f.readline().rstrip("\n")
            self.cols = header.split("\t") if header else None
        self.tsv = open(tsv, "a" if append else "w")
        self.events = open(os.path.join(run_dir, "diag_events.jsonl"), "a" if append else "w")
        self._armed = False

    def before_batch(self, step: int, first: bool = False):
        self._armed = bool(self.every) and (step % self.every == 0 or first)
        self.stats.arm(self._armed)

    def after_backward(self, step: int, epoch: int, total_norm: float, loss: float, names: list) -> bool:
        r = self.grads.record(step, total_norm, names, force_groups=self._armed)
        if self._armed:
            self.stats.collect()
        self.stats.arm(False)
        ok = r["finite"] and math.isfinite(loss)
        if not ok:
            self.event({"kind": "nonfinite", "step": step, "epoch": epoch, "loss": loss, "grad_norm": total_norm, "designs": names[:8]})
        elif r["explode"]:
            self.event({"kind": "explode", "step": step, "epoch": epoch, "loss": loss, "grad_norm": total_norm, "designs": names[:8],
                        "median_recent": float(np.median(self.grads.recent)) if len(self.grads.recent) else None,
                        "groups": r["groups"] or self.grads.group_norms()})
        return ok

    def event(self, e: dict):
        self.events.write(json.dumps(e) + "\n"); self.events.flush()

    def epoch(self, epoch: int, extra: dict | None = None):
        row = {"epoch": epoch}
        row.update(self.grads.epoch_row()); row.update(self.stats.epoch_row()); row.update(extra or {})
        if self.cols is None:
            self.cols = list(row)
            self.tsv.write("\t".join(self.cols) + "\n")
        for c in row:
            if c not in self.cols:
                self.cols.append(c)
        self.tsv.write("\t".join(f"{row.get(c, ''):.4g}" if isinstance(row.get(c, ""), float) else str(row.get(c, "")) for c in self.cols) + "\n")
        self.tsv.flush()

    def close(self):
        self.tsv.close(); self.events.close()


def summarise_grads(run_dir: str) -> str:
    import csv
    p = os.path.join(run_dir, "diag.tsv")
    if not os.path.exists(p):
        return f"{p}: not found"
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    if not rows:
        return f"{p}: empty"
    f = lambda r, k: float(r[k]) if r.get(k, "") not in ("", "nan") else float("nan")
    out = [f"{len(rows)} epochs in {p}"]
    worst = max(rows, key=lambda r: f(r, "grad_max"))
    out.append(f"largest gradient norm {f(worst, 'grad_max'):.3g} at epoch {worst['epoch']} (batch {worst['grad_max_batch']}); "
               f"events: nonfinite {sum(int(r['nonfinite']) for r in rows)}, explode {sum(int(r['explode']) for r in rows)}")
    groups = [c for c in rows[0] if c.startswith("g_")]
    last = rows[-1]
    out.append("group norms, first -> last epoch: " + ", ".join(f"{g[2:]} {f(rows[0], g):.2g} -> {f(last, g):.2g}" for g in groups))
    stats = [c for c in rows[0] if not c.startswith("g_") and c not in ("epoch", "grad_mean", "grad_max", "grad_p99", "grad_max_batch", "nonfinite", "explode")]
    if stats:
        out.append("arm statistics, first -> last epoch: " + ", ".join(f"{s} {f(rows[0], s):.3g} -> {f(last, s):.3g}" for s in stats))
    ev = os.path.join(run_dir, "diag_events.jsonl")
    if os.path.exists(ev):
        events = [json.loads(l) for l in open(ev) if l.strip()]
        kinds = collections.Counter(e["kind"] for e in events)
        out.append(f"events: {dict(kinds)}" + (f"; first: {events[0]}" if events else ""))
    return "\n".join(out)


def attention_report(checkpoint: str, design: str, out_path: str | None = None, device: str = "cpu") -> str:
    from cell_model.data import load_bank
    from graph.vocab import load_vocab
    from model.data import collate, graph_root, load_design
    from model.encoder import ExactMessage, LevelEncoder, build_schedule, instance_level_offset
    from model.train_encoder import load_checkpoint
    root = graph_root("BC")
    vocab = load_vocab(root, "BC")
    model, ck = load_checkpoint(checkpoint, vocab, device)
    if getattr(model, "arch", "mlp") != "attn":
        return f"{checkpoint}: arch {getattr(model, 'arch', 'mlp')}, the attention report needs the attn arm"
    d = load_design(design, root)
    b = collate([d]).to(device)
    sch = build_schedule(b)
    exact = LevelEncoder(ExactMessage(load_bank([], graph_root=root, device=device)))
    with torch.no_grad():
        ex = exact(b, sch, keep=True)
        model.message.record = []
        model.message.monitor = True
        model(b, sch=sch, keep=False)
        stats = model.message.pop_stats()
        rec = model.message.record
        model.message.record = None
    k0 = instance_level_offset(sch)
    ex_cand = ex["msg_arrival"].cpu().numpy()
    rows = []
    for tgt, loc, w, cnd, level in rec:
        i0, i1 = sch.inst_ptr[level] - k0, sch.inst_ptr[level + 1] - k0
        tgt, loc, w, cnd = tgt.numpy(), loc.numpy(), w.numpy(), cnd.numpy()
        ce = ex_cand[i0:i1]
        wm = w.mean(1)
        k = np.bincount(loc, minlength=len(tgt))
        for g in np.flatnonzero(k >= 2):
            sel = np.flatnonzero(loc == g)
            wg, cg, co = wm[sel], ce[sel], cnd[sel]
            if np.isfinite(cg).sum() < 2:
                continue
            order = np.argsort(-cg)
            gap = float(cg[order[0]] - cg[order[1]])
            if not np.isfinite(gap):
                continue
            ent = float(-(wg * np.log(np.clip(wg, 1e-30, None))).sum() / math.log(len(sel)))
            win, att = int(order[0]), int(np.argmax(wg))
            own = int(np.argmax(co))
            own_unique = np.isfinite(co[own]) and int((co == co[own]).sum()) == 1
            rows.append({"pin": int(tgt[g]) // 2, "rf": int(tgt[g]) % 2, "level": level, "k": int(len(sel)), "gap_ps": gap,
                         "entropy": ent, "max_w": float(wg.max()), "agree": int(gap > 0 and att == win),
                         "agree_own": int(own_unique and att == own), "own_is_exact": int(gap > 0 and own_unique and own == win),
                         "w_on_winner": float(wg[cg == cg[win]].sum())})
    if out_path:
        with open(out_path, "w") as f:
            f.write("\t".join(rows[0]) + "\n" if rows else "")
            for r in rows:
                f.write("\t".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    lines = [f"{design}: {len(rows)} pin transitions with >= 2 finite candidates (exact message); the run's own stats on this design: "
             f"{json.dumps({k: round(v, 3) for k, v in stats.items()})}",
             "| gap band (exact) | pin transitions | attention argmax = exact winner | attention argmax = the model's own winner | the model's own winner = exact winner | mean weight on the exact winner(s) | mean entropy |",
             "|---|---|---|---|---|---|---|"]
    if rows:
        band = gap_band(np.array([r["gap_ps"] for r in rows]))
        for i, lab in enumerate(GAP_LABELS):
            sel = [r for r, bb in zip(rows, band) if bb == i]
            if sel:
                pct = lambda key: f"{100 * np.mean([r[key] for r in sel]):.1f} %"
                agree, own_ex = (pct("agree"), pct("own_is_exact")) if i > 0 else ("(ties)", "(ties)")
                lines.append(f"| {lab} | {len(sel)} | {agree} | {pct('agree_own')} | {own_ex} | {np.mean([r['w_on_winner'] for r in sel]):.3f} | {np.mean([r['entropy'] for r in sel]):.3f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grads"); g.add_argument("run_dir")
    a = sub.add_parser("attention"); a.add_argument("--checkpoint", required=True); a.add_argument("--design", required=True)
    a.add_argument("--out", default=None); a.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.cmd == "grads":
        print(summarise_grads(args.run_dir))
    else:
        print(attention_report(args.checkpoint, args.design, args.out, args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
