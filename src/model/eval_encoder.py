from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import torch

from baselines.score_step1 import labelled_paths, teacher_forced_steps, tie_breaker
from cell_model.data import load_bank
from cell_model.model import make_model
from encoder.forward_pass import StoredDelayScorer, exact_cell_fn, forward_pass, model_cell_fn
from graph.reference_sta import run_reference_sta
from graph.vocab import load_vocab
from walk.candidates import WalkGraph, allowed_next
from walk.oracle import decide, oracle_walk

from paths import RUNS

from .data import DATA, SIDECAR_VERSION, collate, default_split_file, design_from_graph, graph_root, load_graph, select_designs, sidecar_path
from .diagnostics import GAP_LABELS, gap_band
from .encoder import ExactMessage, LevelEncoder, arc_delay, build_schedule, instance_level_offset

FIELDS = ["group", "design", "nodes", "levels", "pins_rf", "arr_mean", "arr_p99", "arr_max", "slew_mean", "slew_p99", "slew_max",
          "arr_err_at_deepest_level", "n_ep", "slack_mean", "slack_max", "n_tied_worst", "endpoint", "worst_path",
          "n_ep_paths", "ep_path", "n_steps", "step", "seconds", "rss_mb", "error"]
EDGES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 100000]
STEP2 = f"{DATA}/cell_model/model.pt"
_S = {}


def _init(root, corner, source, checkpoint, device, seed, stage="synth", labels_ref=False, gpu_route=False):
    torch.set_num_threads(1)
    vocab = load_vocab(root, corner)
    _S.update(root=root, vocab=vocab, source=source, device=device, seed=seed, stage=stage, labels_ref=labels_ref or stage != "synth", gpu_route=False)
    bank = load_bank([], graph_root=root, corner=corner)
    _S["exact"] = LevelEncoder(ExactMessage(bank))
    if source == "checkpoint":
        from .train_encoder import load_checkpoint
        _S["model"], _ = load_checkpoint(checkpoint, vocab, device)
    elif source == "decoder":
        from .train_decoder import load_decoder_checkpoint
        _S["model"], _S["decoder"], _ = load_decoder_checkpoint(checkpoint, vocab, device, bank=bank)
        _S["gpu_route"] = needs_gpu_route(_S["decoder"]) or bool(gpu_route)
    else:
        if source == "exact":
            _S["fn"] = exact_cell_fn(bank)
        else:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = make_model(ck["arm"], ck["width"], ck["depth"])
            model.load_state_dict(ck["state_dict"]); model.eval()
            _S["fn"] = model_cell_fn(bank, model)


@torch.no_grad()
def run_pass(data, gd, ref) -> dict:
    if _S["source"] not in ("checkpoint", "decoder"):
        return forward_pass(data, _S["fn"], "cpu", clk_arr=ref["clk_arr"])
    side = None
    p = sidecar_path(_S["root"], gd)
    if os.path.exists(p):
        side = torch.load(p, weights_only=True)
        if side["sidecar_version"] != SIDECAR_VERSION:
            side = None
    d = design_from_graph(data, side, gd)
    b = collate([d]).to(_S["device"])
    out = _S["model"](b)
    res = {k: out[k].double().cpu().numpy() for k in ("arr", "slew", "req", "slack", "ep_slack")}
    res["ep_rf"] = out["ep_rf"].cpu().numpy()
    res["arc_delay"] = arc_delay(out, int(data.edge_index_arc.shape[1])).double().cpu().numpy()
    res["clk_arr"] = ref["clk_arr"]
    if _S["source"] == "decoder":
        from .decoder import walk_logit_table
        dec = _S["decoder"](out, b)
        res["walk_logit"] = walk_logit_table(out, dec["logit"], int(data.edge_index_arc.shape[1])).double().cpu().numpy()
        res["ep_score"] = dec["ep_score"].double().cpu().numpy()
        if _S["gpu_route"]:
            res["_gpu"] = (out, b, dec)
    return res


def needs_gpu_route(dec) -> bool:
    return dec.kind != "pointer" or int(dec.config.get("ep_rerank_k", 0)) > 0


@torch.no_grad()
def gpu_walks(res: dict, g, starts: list) -> dict:
    from walk.candidates import LOAD_ROLES, ROLE_CELL_OUT
    out, b, d = res["_gpu"]
    dev = b.level.device
    sp = torch.tensor([e for e, _ in starts], dtype=torch.long, device=dev)
    sr = torch.tensor([rf for _, rf in starts], dtype=torch.long, device=dev)
    r = _S["decoder"].decode(out, b, d["logit"], sp, sr, keep_tokens=True)
    toks, rfs, n_tok, complete = (r[k].cpu().numpy() for k in ("tokens", "token_rf", "n_tok", "complete"))
    walks = {}
    for i, key in enumerate(starts):
        t = toks[i, :n_tok[i]].tolist(); trf = rfs[i, :n_tok[i]].tolist()
        pins, pin_rf = [], []
        for tok, rf in zip(t, trf):
            pins.append(tok); pin_rf.append(rf)
            if g.role[tok] in LOAD_ROLES:
                drv = int(g.driver_of_net[tok])
                if drv >= 0 and g.role[drv] == ROLE_CELL_OUT:
                    pins.append(drv); pin_rf.append(rf)
        walks[key] = {"tokens": t, "token_rf": trf, "pins": pins, "pin_rf": pin_rf, "complete": bool(complete[i])}
    return walks


@torch.no_grad()
def pick_endpoint(res: dict, ends: list, slack: np.ndarray, choose) -> int | None:
    if np.all(np.isnan(slack)):
        return None
    e = int(ends[choose(-np.where(np.isnan(slack), np.inf, slack))])
    dec = _S.get("decoder")
    if dec is not None and dec.rerank_k > 0 and "_gpu" in res:
        out, b, d = res["_gpu"]
        cand = torch.tensor(ends, dtype=torch.long, device=b.level.device)
        rr = dec.rerank(out, b, d, cand, torch.zeros_like(cand), 1)
        p = int(rr["pick"][0])
        if p >= 0:
            e = int(ends[p])
    return e


@torch.no_grad()
def gpu_steps(res: dict, lab) -> tuple:
    from .decoder import decoder_losses, label_positives
    from .decoder_data import collate_labels, path_labels_from_graph
    from .seq_decoder import seq_losses
    out, b, d = res["_gpu"]
    dec = _S["decoder"]
    labels = collate_labels([path_labels_from_graph(lab)], b).to(b.level.device)
    pos = label_positives(out, labels, b.num_nodes)
    if dec.kind == "pointer":
        _, _, st = decoder_losses(d, out, b, labels, pos)
    else:
        _, st, _ = seq_losses(dec, out, b, labels, pos)
    return int(round(st["dec_right"])), int(round(st["dec_n"]))


@torch.no_grad()
def race_gaps(data, design: str) -> np.ndarray:
    d = design_from_graph(data, None, design)
    b = collate([d])
    sch = build_schedule(b)
    out = _S["exact"](b, sch, keep=True)
    k = instance_level_offset(sch)
    cand = out["msg_arrival"].numpy(); ft = sch.inst_ft[k:k + len(cand)].numpy()
    fin = np.isfinite(cand); cand, ft = cand[fin], ft[fin]
    gap = np.full(2 * int(data.num_nodes), np.nan)
    if len(cand):
        order = np.lexsort((-cand, ft)); cand, ft = cand[order], ft[order]
        first = np.r_[True, ft[1:] != ft[:-1]]; idx = np.flatnonzero(first); cnt = np.diff(np.r_[idx, len(ft)])
        m = cnt >= 2
        gap[ft[idx[m]]] = cand[idx[m]] - cand[idx[m] + 1]
    return gap.reshape(-1, 2)


def eval_design(design: str):
    t0 = time.time()
    row = dict.fromkeys(FIELDS, 0)
    row.update(group=design.split("/")[0], design=design.split("/")[1], error="")
    gaps = np.zeros((len(GAP_LABELS), 2))
    try:
        data = torch.load(os.path.join(_S["root"], design, "graph.pt"), weights_only=False)
        vocab = _S["vocab"]
        ref = run_reference_sta(data, vocab)
        res = run_pass(data, design, ref)
        level = data.level.numpy()
        if _S["labels_ref"]:
            side = torch.load(sidecar_path(_S["root"], design, _S["stage"]), weights_only=True)
            lab = load_graph(design, _S["root"], stage=_S["stage"])
            has = side["y_mask_arrival"].numpy() & np.isfinite(res["arr"])
            has_s = side["y_mask_slew"].numpy() & has
            ref_arr, ref_slew = side["y_arrival"].double().numpy(), side["y_slew"].double().numpy()
            ep = lab.y_ep_index.numpy()
            ref_ep = lab.y_ep_slack.double().numpy()
        else:
            lab = data
            has = np.isfinite(ref["arr"])
            ref_arr, ref_slew = ref["arr"], ref["slew"]
            g = race_gaps(data, design)
            sel = has & np.isfinite(g)
            band = gap_band(g[sel])
            err = np.abs(res["arr"][sel] - ref["arr"][sel])
            gaps[:, 0] = np.bincount(band, minlength=len(GAP_LABELS)); gaps[:, 1] = np.bincount(band, err, len(GAP_LABELS))
            if not np.array_equal(has, np.isfinite(res["arr"])):
                row["error"] = "the pass and the reference do not give an arrival to the same pins; "
                has &= np.isfinite(res["arr"])
            has_s = has
            ep = data.y_ep_index.numpy()
            ref_ep = ref["ep_slack"][ep]
        ea = np.abs(res["arr"][has] - ref_arr[has])
        es = np.abs(res["slew"] - ref_slew)[has_s]
        lv = np.broadcast_to(level[:, None], has.shape)[has]
        lvs = np.broadcast_to(level[:, None], has_s.shape)[has_s]
        L = int(level.max()) + 1
        per_level = np.stack([np.bincount(lv, minlength=L), np.bincount(lv, ea, L), np.bincount(lvs, es, L)], 1)
        mx = np.zeros((L, 2)); np.maximum.at(mx[:, 0], lv, ea); np.maximum.at(mx[:, 1], lvs, es)
        row.update(nodes=int(data.num_nodes), levels=L, pins_rf=int(has.sum()),
                   arr_mean=ea.mean(), arr_p99=np.quantile(ea, 0.99), arr_max=ea.max(),
                   slew_mean=es.mean(), slew_p99=np.quantile(es, 0.99), slew_max=es.max(),
                   arr_err_at_deepest_level=ea[lv == lv.max()].mean())
        se = np.abs(res["ep_slack"][ep] - ref_ep)
        row.update(n_ep=len(ep), slack_mean=np.nanmean(se), slack_max=np.nanmax(se))

        g = WalkGraph(data)
        gpu = "_gpu" in res
        if "walk_logit" in res:
            from .decoder import LogitScorer
            scorer = LogitScorer(g, res, vocab)
        else:
            scorer = StoredDelayScorer(g, res, vocab)
        choose = tie_breaker(np.random.default_rng([zlib.crc32(design.encode()), _S["seed"]]))
        cache = {}
        paths = labelled_paths(lab)
        path_of = {e: p for e, p, _ in paths}
        tied = set(lab.y_worst_endpoints.tolist())
        ends = allowed_next(g, None)
        slack = res["ep_slack"][ends]
        if "ep_score" in res:
            slack = -res["ep_score"][ends]
        row.update(n_tied_worst=len(tied), n_ep_paths=len(paths))
        e = pick_endpoint(res, ends, slack, choose)
        walks = None
        if gpu:
            starts = ([(e, int(res["ep_rf"][e]))] if e is not None else []) + [(ep, prf[-1]) for ep, _, prf in paths]
            walks = gpu_walks(res, g, starts)
        walk = (lambda ep, rf: walks[(ep, rf)]) if gpu else (lambda ep, rf: oracle_walk(g, res, ep, rf, scorer=scorer, cache=cache, choose=choose))
        if e is not None and e in tied:
            row["endpoint"] = 1
            w = walk(e, int(res["ep_rf"][e]))
            row["worst_path"] = int(w["complete"] and w["pins"] == path_of[e][::-1])
        for ep, p, prf in paths:
            w = walk(ep, prf[-1])
            row["ep_path"] += int(w["complete"] and w["pins"] == p[::-1])
            if not gpu:
                for tok, rf, truth, _ in teacher_forced_steps(g, p, prf)[0]:
                    row["n_steps"] += 1
                    row["step"] += int(decide(g, scorer, tok, rf, cache, choose)["choice"] == truth)
        if gpu:
            row["step"], row["n_steps"] = gpu_steps(res, lab)
    except Exception as err:
        row["error"] += f"{type(err).__name__}: {err}"[:300]
        per_level, mx = np.zeros((0, 3)), np.zeros((0, 2))
    row["seconds"] = round(time.time() - t0, 2)
    row["rss_mb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return row, per_level, mx, gaps


def gaps_md(gaps: np.ndarray) -> list:
    out = ["| Race gap at the pin (exact winner - runner-up) | Pin transitions | Arrival error, mean (ps) |", "|---|---|---|"]
    for lab, (n, s) in zip(GAP_LABELS, gaps):
        if n:
            out.append(f"| {lab} | {int(n):,} | {s / n:.3f} |")
    return out + [""]


def scores(rows) -> dict:
    f = lambda r, k: float(r[k] or 0)
    ok = [r for r in rows if f(r, "pins_rf")]
    n, pins, n_ep = len(ok), sum(f(r, "pins_rf") for r in ok), sum(f(r, "n_ep") for r in ok)
    paths, steps = sum(f(r, "n_ep_paths") for r in ok), sum(f(r, "n_steps") for r in ok)
    tot = {k: sum(f(r, k) for r in ok) for k in ("endpoint", "worst_path", "ep_path", "step")}
    return {"designs": len(rows), "designs_without_result": len(rows) - n,
            "arrival_mean_ps": sum(f(r, "arr_mean") * f(r, "pins_rf") for r in ok) / max(pins, 1),
            "arrival_max_ps": max((f(r, "arr_max") for r in ok), default=0.0),
            "slew_mean_ps": sum(f(r, "slew_mean") * f(r, "pins_rf") for r in ok) / max(pins, 1),
            "slack_mean_ps": sum(f(r, "slack_mean") * f(r, "n_ep") for r in ok) / max(n_ep, 1),
            "right_endpoint_pct": 100 * tot["endpoint"] / max(n, 1), "right_worst_path_pct": 100 * tot["worst_path"] / max(n, 1),
            "right_worst_path": int(tot["worst_path"]), "right_path_endpoint_given_pct": 100 * tot["ep_path"] / max(paths, 1),
            "right_choice_per_step_pct": 100 * tot["step"] / max(steps, 1)}


GPU_ROUTE_NOTE = ("Walks by the decoder's GPU read-out (model.decoder.PathDecoder.decode), not the Python walk: exact logit ties go to the "
                  "smallest pin, then the first candidate (the Python walk draws at random); the per-step score is teacher forced on the GPU along "
                  "the labelled paths over the steps whose decision offers >= 2 candidate pins with an arrival (train_decoder's dec_acc), so n_steps "
                  "can differ a little from the table route's count of steps with more than one allowed pin. With the endpoint re-ranker "
                  "(ep_rerank_k > 0) the endpoint is the re-ranked pick among the k endpoints of lowest endpoint-head score (ties: the first).")


def summary_md(rows, levels, lmax, title: str, gaps: np.ndarray | None = None, notes: list | None = None) -> str:
    ok = [r for r in rows if r["pins_rf"]]
    n = len(ok)
    pins = sum(r["pins_rf"] for r in ok)
    n_ep = sum(r["n_ep"] for r in ok)
    out = [f"## {title}", "", f"{len(rows)} designs ({len(rows) - n} without a result), {pins:,} pin transitions, {n_ep:,} endpoints.", "",
           "| | Mean | Largest |", "|---|---|---|",
           f"| Arrival (ps) | {sum(r['arr_mean'] * r['pins_rf'] for r in ok) / pins:.3f} | {max(r['arr_max'] for r in ok):.2f} |",
           f"| Slew (ps) | {sum(r['slew_mean'] * r['pins_rf'] for r in ok) / pins:.3f} | {max(r['slew_max'] for r in ok):.2f} |",
           f"| Endpoint slack (ps) | {sum(r['slack_mean'] * r['n_ep'] for r in ok) / n_ep:.3f} | {max(r['slack_max'] for r in ok):.2f} |", "",
           "| Levels | Pin transitions | Arrival error, mean (ps) | Arrival error, largest (ps) | Slew error, mean (ps) |",
           "|---|---|---|---|---|"]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = levels[lo:hi]
        if len(s) and s[:, 0].sum():
            top = min(hi, len(levels)) - 1
            lab = f"{lo:,}" if top == lo else f"{lo:,} to {top:,}"
            out.append(f"| {lab} | {int(s[:, 0].sum()):,} | {s[:, 1].sum() / s[:, 0].sum():.3f} | {lmax[lo:hi, 0].max():.2f} | "
                       f"{s[:, 2].sum() / s[:, 0].sum():.3f} |")
    if gaps is not None and gaps[:, 0].sum():
        out += [""] + gaps_md(gaps)
    paths = sum(r["n_ep_paths"] for r in ok); steps = sum(r["n_steps"] for r in ok)
    en, wp, epp, st = (sum(r[k] for r in ok) for k in ("endpoint", "worst_path", "ep_path", "step"))
    out += ["", "| Score | This pass |", "|---|---|",
            f"| Right endpoint | {100 * en / n:.2f} % ({en} of {n}) |",
            f"| Right worst path | {100 * wp / n:.2f} % ({wp} of {n}) |",
            f"| Right path, endpoint given | {100 * epp / paths:.2f} % ({epp:,} of {paths:,}) |",
            f"| Right choice per step | {100 * st / steps:.2f} % ({st:,} of {steps:,}) |", ""]
    for note in notes or []:
        out += [note, ""]
    lost = [r for r in ok if not r["worst_path"]]
    if lost:
        out += ["Designs whose worst path is lost:", "", "| Design | Levels | Endpoints | Tied worst endpoints | Largest arrival error (ps) | Endpoint | Worst path |",
                "|---|---|---|---|---|---|---|"]
        for r in sorted(lost, key=lambda r: r["design"]):
            out.append(f"| `{r['group']}/{r['design']}` | {r['levels']} | {r['n_ep']} | {r['n_tied_worst']} | {r['arr_max']:.2f} | "
                       f"{'right' if r['endpoint'] else 'wrong'} | wrong |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("checkpoint", "decoder", "plugin", "exact"), required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--corner", default="BC")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--val-only", nargs="*", default=[])
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=f"{RUNS}/eval")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-stage", default="synth", choices=("synth", "place", "groute"))
    ap.add_argument("--labels-as-reference", action="store_true")
    ap.add_argument("--gpu-route", action="store_true")
    a = ap.parse_args(argv)
    if a.device != "cpu" and a.workers != 1:
        ap.error("--device cuda needs --workers 1")
    if a.source in ("checkpoint", "decoder") and not a.checkpoint:
        ap.error(f"--source {a.source} needs --checkpoint")
    ckpt = a.checkpoint or STEP2
    root = graph_root(a.corner)
    if a.designs:
        designs = list(a.designs)
    else:
        designs = select_designs(default_split_file(a.corner), [a.split], a.val_only, allow_test=a.allow_test)[a.split]
    designs = designs[:a.limit] if a.limit else designs
    tag = a.tag or f"{a.source}_{a.split}" + (f"_{a.label_stage}" if a.label_stage != "synth" else "")
    out = os.path.join(a.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    rows = []
    levels = np.zeros((0, 3)); lmax = np.zeros((0, 2)); gaps = np.zeros((len(GAP_LABELS), 2))
    t0 = time.time()
    init = (root, a.corner, a.source, ckpt, a.device, a.seed, a.label_stage, a.labels_as_reference, a.gpu_route)
    with open(os.path.join(out, "designs.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"); w.writeheader()
        if a.workers == 1:
            _init(*init)
            it, pool = map(eval_design, designs), None
        else:
            pool = Pool(a.workers, initializer=_init, initargs=init)
            it = pool.imap_unordered(eval_design, designs, chunksize=1)
        for i, (row, pl, mx, gp) in enumerate(it):
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}); f.flush()
            rows.append(row); gaps += gp
            if row["error"]:
                print(f"ERROR {row['group']}/{row['design']}: {row['error']}")
            if len(pl) > len(levels):
                levels = np.pad(levels, ((0, len(pl) - len(levels)), (0, 0))); lmax = np.pad(lmax, ((0, len(mx) - len(lmax)), (0, 0)))
            levels[:len(pl)] += pl; lmax[:len(mx)] = np.maximum(lmax[:len(mx)], mx)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(designs)} designs, {time.time() - t0:.0f}s", flush=True)
        if pool:
            pool.close(); pool.join()
    with open(os.path.join(out, "levels.tsv"), "w") as f:
        f.write("level\tpins_rf\tarr_mean\tarr_max\tslew_mean\tslew_max\n")
        for l in range(len(levels)):
            if levels[l, 0]:
                f.write(f"{l}\t{int(levels[l, 0])}\t{levels[l, 1] / levels[l, 0]:.6g}\t{lmax[l, 0]:.6g}\t{levels[l, 2] / levels[l, 0]:.6g}\t{lmax[l, 1]:.6g}\n")
    with open(os.path.join(out, "gaps.tsv"), "w") as f:
        f.write("gap_band\tpins_rf\tarr_mean\n")
        for lab, (n, s) in zip(GAP_LABELS, gaps):
            f.write(f"{lab}\t{int(n)}\t{s / n if n else float('nan'):.6g}\n")
    title = f"{a.source}{' ' + os.path.abspath(a.checkpoint) if a.source == 'checkpoint' else ''}, split {a.split if not a.designs else 'designs'}"
    if a.label_stage != "synth" or a.labels_as_reference:
        title += f", against the {a.label_stage} labels"
    gpu = a.gpu_route
    if a.source == "decoder" and not gpu:
        from .train_decoder import load_decoder_checkpoint
        gpu = needs_gpu_route(load_decoder_checkpoint(ckpt, load_vocab(root, a.corner))[1])
    md = summary_md(rows, levels, lmax, title, gaps, [GPU_ROUTE_NOTE] if gpu else None)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"{time.time() - t0:.0f} s wall; max worker RSS {max(r['rss_mb'] for r in rows)} MB; {out}")
    n = sum(1 for r in rows if r["pins_rf"])
    return 0 if n == len(rows) and not any(r["error"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
