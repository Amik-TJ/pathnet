from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from paths import RUNS

from .data import DATA, DepthBatcher, collate, default_split_file, graph_root, load_designs, select_designs, shuffle_labels
from .encoder import ARCHES, ARM_DEFAULTS, DEFAULT_CONFIG, TimingEncoder
from .diagnostics import Diag
from .heads import arc_delay_targets, arc_losses, dense_losses, dense_terms

STEP2 = f"{DATA}/cell_model/model.pt"


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out-root", default=RUNS)
    ap.add_argument("--corner", default="BC")
    ap.add_argument("--label-stage", default="synth", choices=("synth", "place", "groute"))
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--val-only", nargs="*", default=[])
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--arc-mode", default=DEFAULT_CONFIG["arc_mode"], choices=("tables", "key"))
    ap.add_argument("--kernel-width", type=int, default=DEFAULT_CONFIG["kernel_width"])
    ap.add_argument("--kernel-depth", type=int, default=DEFAULT_CONFIG["kernel_depth"])
    ap.add_argument("--d-arc", type=int, default=DEFAULT_CONFIG["d_arc"])
    ap.add_argument("--dz", type=int, default=DEFAULT_CONFIG["dz"])
    ap.add_argument("--mlp-width", type=int, default=DEFAULT_CONFIG["mlp_width"])
    ap.add_argument("--mlp-depth", type=int, default=DEFAULT_CONFIG["mlp_depth"])
    ap.add_argument("--correction", type=int, default=int(DEFAULT_CONFIG["correction"]))
    ap.add_argument("--arch", default=ARM_DEFAULTS["arch"], choices=ARCHES)
    ap.add_argument("--heads", type=int, default=ARM_DEFAULTS["heads"])
    ap.add_argument("--net-pool", type=int, default=ARM_DEFAULTS["net_pool"])
    ap.add_argument("--fanin-self-attn", type=int, default=ARM_DEFAULTS["fanin_self_attn"])
    ap.add_argument("--soft-arrival", type=int, default=ARM_DEFAULTS["soft_arrival"])
    ap.add_argument("--soft-tau", type=float, default=ARM_DEFAULTS["soft_tau"])
    ap.add_argument("--layers", type=int, default=ARM_DEFAULTS["layers"])
    ap.add_argument("--d-model", type=int, default=ARM_DEFAULTS["d_model"])
    ap.add_argument("--d-edge", type=int, default=ARM_DEFAULTS["d_edge"])
    ap.add_argument("--level-pe", type=int, default=ARM_DEFAULTS["level_pe"])
    ap.add_argument("--global-attn", default=ARM_DEFAULTS["global_attn"], choices=("none", "linear", "multihead"))
    ap.add_argument("--checkpoint-layers", type=int, default=ARM_DEFAULTS["checkpoint_layers"])
    ap.add_argument("--out-scale", type=float, default=ARM_DEFAULTS["out_scale"])
    ap.add_argument("--arc-init", default="scratch", choices=("scratch", "step2"))
    ap.add_argument("--step2-checkpoint", default=STEP2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--final-lr-frac", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--max-inst", type=int, default=300_000)
    ap.add_argument("--max-designs", type=int, default=64)
    ap.add_argument("--deep-levels", type=int, default=256)
    ap.add_argument("--eval-max-inst", type=int, default=1_500_000)
    ap.add_argument("--w-arr", type=float, default=1.0)
    ap.add_argument("--w-slew", type=float, default=1.0)
    ap.add_argument("--w-slack", type=float, default=1.0)
    ap.add_argument("--w-arc", type=float, default=1.0)
    ap.add_argument("--arrival-from-delays", type=int, default=ARM_DEFAULTS["arrival_from_delays"])
    ap.add_argument("--delay-norm", default=ARM_DEFAULTS["delay_norm"], choices=("scale", "stats"))
    ap.add_argument("--arc-stats-designs", type=int, default=256)
    ap.add_argument("--per-design-loss", type=int, default=0)
    ap.add_argument("--loss-weighting", default="fixed", choices=("fixed", "uncertainty"))
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--aux-batch", type=int, default=4096)
    ap.add_argument("--tf-prob", type=float, default=0.0)
    ap.add_argument("--shuffle-labels", type=int, default=0)
    ap.add_argument("--knockout-tables", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=46.0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--gl-alpha", type=float, default=10.0)
    ap.add_argument("--gl-strips", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--diag-every", type=int, default=20)
    ap.add_argument("--explode-factor", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-after", type=int, default=0)
    ap.add_argument("--eval-workers", type=int, default=6)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--resume", default=None)
    pre, _ = ap.parse_known_args(argv)
    known = {act.dest for act in ap._actions}
    if pre.config:
        with open(pre.config) as f:
            cfg = json.load(f)
        unknown = sorted(k for k in cfg if k not in known or k == "config")
        if unknown:
            raise SystemExit(f"{pre.config}: unknown keys {unknown}")
        ap.set_defaults(**cfg)
    pre, _ = ap.parse_known_args(argv)
    run_dir = find_run(pre.out_root, pre.resume) if pre.resume else None
    saved = None
    if run_dir is not None:
        saved = (_read_json(os.path.join(run_dir, "config.json")) or {}).get("args")
        if saved is None:
            raise SystemExit(f"{run_dir}: no config.json with the run's settings, nothing to resume")
        ap.set_defaults(**{k: v for k, v in saved.items() if k in known})
    a = ap.parse_args(argv)
    if a.smoke:
        a.train_limit, a.val_limit, a.epochs = a.train_limit or 4, a.val_limit or 4, min(a.epochs, 2)
        a.kernel_width, a.kernel_depth, a.dz, a.mlp_width, a.aux_batch, a.warmup_steps = 32, 2, 8, 16, 256, 2
        a.layers, a.d_model, a.d_edge, a.heads = min(a.layers, 2), 16, 8, 2
    a.resume = run_dir
    if saved is not None:
        changed = {k: (saved[k], getattr(a, k)) for k in saved if k in known and k not in NOT_SETTINGS and saved[k] != getattr(a, k)}
        if changed:
            raise SystemExit(f"--resume {run_dir}: these flags would change the run's settings (saved, given): {changed}")
        a.out_root, a.run_tag, a.config = os.path.dirname(run_dir), os.path.basename(run_dir), saved.get("config")
    a.run_tag = a.run_tag or time.strftime("enc_%Y%m%d_%H%M%S")
    return a


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_run(out_root: str, key: str) -> str:
    for d in (key, os.path.join(out_root, key)):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            return os.path.abspath(d)
    raise SystemExit(f"--resume {key}: no run folder of that name in {out_root}")


def model_config(a) -> dict:
    c = {"arc_mode": a.arc_mode, "kernel_width": a.kernel_width, "kernel_depth": a.kernel_depth, "d_arc": a.d_arc,
         "dz": a.dz, "mlp_width": a.mlp_width, "mlp_depth": a.mlp_depth, "correction": bool(a.correction)}
    c.update({k: getattr(a, k) for k in ARM_DEFAULTS})
    return c


SNAPSHOT_CODE_DIRS = ("model", "cell_model", "graph", "encoder", "walk")
NOT_SETTINGS = ("run_tag", "out_root", "load_workers", "config", "device", "smoke", "resume", "eval_after", "eval_workers", "eval_limit")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def non_default_args(a) -> dict:
    d = vars(build_args(["--run-tag", "x"]))
    return {k: v for k, v in sorted(vars(a).items()) if k not in NOT_SETTINGS and d.get(k, object()) != v}


def snapshot_run(run_dir: str, a) -> dict:
    import shutil
    snap = os.path.join(run_dir, "snapshot")
    os.makedirs(os.path.join(snap, "code"), exist_ok=True)
    out = {"config_sha256": None, "experiment": None}
    if a.config:
        shutil.copyfile(a.config, os.path.join(snap, "config.json"))
        out["config_sha256"] = sha256_file(a.config)
        out["experiment"] = os.path.basename(os.path.dirname(os.path.abspath(a.config)))
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in sorted(os.listdir(os.path.join(src, "model"))):
        if f.endswith(".py"):
            shutil.copyfile(os.path.join(src, "model", f), os.path.join(snap, "code", f))
    text, out["code_sha256"] = code_manifest()
    with open(os.path.join(snap, "code_manifest.tsv"), "w") as f:
        f.write("file\tsha256\n" + text)
    return out


def code_manifest() -> tuple:
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for d in SNAPSHOT_CODE_DIRS:
        for root, _, files in sorted(os.walk(os.path.join(src, d))):
            rows += [(os.path.relpath(os.path.join(root, f), src), sha256_file(os.path.join(root, f))) for f in files if f.endswith(".py")]
    text = "".join(f"{r}\t{h}\n" for r, h in sorted(rows))
    return text, hashlib.sha256(text.encode()).hexdigest()


def atomic_save(obj, path: str):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def atomic_json(obj, path: str):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def truncate_tsv(path: str, last_epoch: int):
    with open(path) as f:
        lines = f.read().splitlines(keepends=True)
    keep = lines[:1] + [l for l in lines[1:] if l.strip() and int(l.split("\t", 1)[0]) <= last_epoch]
    with open(path, "w") as f:
        f.writelines(keep)


def load_checkpoint(path: str, vocab, device="cpu") -> tuple:
    ck = torch.load(path, map_location=device, weights_only=False)
    if "encoder_state" in ck and "state_dict" not in ck:
        orig_path = ck.get("encoder_checkpoint")
        if not orig_path or not os.path.exists(orig_path):
            raise FileNotFoundError(f"{path}: a train_decoder checkpoint; its encoder's own settings live in the original encoder "
                                    f"checkpoint {orig_path!r}, which was not found")
        orig = torch.load(orig_path, map_location="cpu", weights_only=False)
        lw = (ck.get("resume") or {}).get("loss_weights")
        ck = {"config": ck["encoder_config"], "state_dict": ck["encoder_state"], "args": orig.get("args"),
              "loss_weights": lw if lw is not None else orig.get("loss_weights"), "val": ck.get("val"), "epoch": ck.get("epoch"),
              "from_decoder_checkpoint": path, "original_encoder_checkpoint": orig_path,
              "loss_weights_from": "decoder checkpoint (resume)" if lw is not None else "original encoder checkpoint"}
    m = TimingEncoder(vocab, ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


class EarlyStopper:

    def __init__(self, patience: int = 20, min_delta: float = 0.005, gl_alpha: float = 10.0, gl_strips: int = 5,
                 min_epochs: int = 10):
        self.patience, self.min_delta, self.gl_alpha, self.gl_strips, self.min_epochs = patience, min_delta, gl_alpha, gl_strips, min_epochs
        self.best, self.best_epoch, self.since, self.gl_run = math.inf, 0, 0, 0

    def gl(self, value: float) -> float:
        return 100.0 * (value / self.best - 1.0) if math.isfinite(self.best) and self.best > 0 else 0.0

    def update(self, epoch: int, value: float) -> str:
        if value < self.best * (1.0 - self.min_delta):
            self.since = 0
        else:
            self.since += 1
        if value < self.best:
            self.best, self.best_epoch = value, epoch
        g = self.gl(value)
        self.gl_run = self.gl_run + 1 if self.gl_alpha > 0 and g > self.gl_alpha else 0
        if epoch < self.min_epochs:
            return ""
        if self.gl_alpha > 0 and self.gl_run >= self.gl_strips:
            return (f"overfitting: val arrival MAE {g:.1f} % above its best (epoch {self.best_epoch}) "
                    f"for {self.gl_run} evaluations (GL > {self.gl_alpha})")
        if self.patience > 0 and self.since >= self.patience:
            return (f"plateau: no improvement over {100 * self.min_delta:.2g} % for {self.since} evaluations "
                    f"(best at epoch {self.best_epoch})")
        return ""


class LossWeights(nn.Module):
    TERMS = ("arr", "slew", "slack", "arc")

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(len(self.TERMS)))

    def forward(self, terms: dict, weights: dict) -> torch.Tensor:
        total = 0.0
        for i, t in enumerate(self.TERMS):
            if t in terms and weights.get(t, 0):
                total = total + weights[t] * (terms[t] * torch.exp(-self.s[i]) + self.s[i])
        return total

    def scales(self) -> dict:
        return {f"lw_{t}": float(torch.exp(self.s[i])) for i, t in enumerate(self.TERMS)}


@torch.no_grad()
def arc_target_stats(model, designs: list, bank, device, n: int, seed: int) -> tuple:
    from .encoder import build_schedule
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(designs), min(n, len(designs)), replace=False)
    ys = []
    for i in idx:
        b = collate([designs[int(i)]]).to(device)
        y, m = arc_delay_targets({"schedule": build_schedule(b)}, b, bank)
        ys.append(y[m].double())
    y = torch.cat(ys)
    return y.mean(0).float(), y.std(0).float(), int(len(y))


def lr_factor(progress: float, step: int, warmup: int, final_frac: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = min(max(progress, 0.0), 1.0)
    return final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, designs, batches, device, knockout=False, bank=None) -> dict:
    model.eval()
    layered = getattr(model, "layered", False) and bank is not None
    tot = {"arr_sum": 0.0, "arr_n": 0, "slew_sum": 0.0, "slew_n": 0, "slack_sum": 0.0, "slack_n": 0,
           "arr_unreached": 0, "slack_unreached": 0, "arc_delay_sum": 0.0, "arc_n": 0}
    for ids in batches:
        b = collate([designs[i] for i in ids], knockout_tables=knockout).to(device)
        out = model(b, keep=layered)
        _, st = dense_losses(out, b)
        if layered:
            y, m = arc_delay_targets(out, b, bank)
            _, sa = arc_losses(out, y, m)
            st.update(arc_delay_sum=sa["arc_delay_sum"], arc_n=sa["arc_n"])
        for k in tot:
            tot[k] += st.get(k, 0)
    model.train()
    r = {"arr_mae": tot["arr_sum"] / max(tot["arr_n"], 1), "slew_mae": tot["slew_sum"] / max(tot["slew_n"], 1),
         "slack_mae": tot["slack_sum"] / max(tot["slack_n"], 1), "arr_unreached": tot["arr_unreached"],
         "arc_mae": tot["arc_delay_sum"] / max(tot["arc_n"], 1)}
    r["loss"] = r["arr_mae"] + r["slew_mae"] + r["slack_mae"]
    return r


def main(argv=None):
    a = build_args(argv)
    from cell_model.data import Sampler, load_bank
    from graph.vocab import load_vocab
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    root = graph_root(a.corner)
    run_dir = os.path.join(a.out_root, a.run_tag)
    ck = None
    if a.resume:
        if os.path.exists(os.path.join(run_dir, "result.json")):
            print(f"{run_dir} already finished (result.json): nothing to resume", flush=True)
            return evaluate_run(run_dir, a) if a.eval_after else 0
        if os.path.exists(os.path.join(run_dir, "last.pt")):
            ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)
            if "resume" not in ck:
                raise SystemExit(f"{run_dir}/last.pt was written before resume existed (no optimizer state): start a new run")
            print(f"resuming {run_dir} after epoch {ck['epoch']} (step {ck['step']})", flush=True)
        else:
            print(f"{run_dir} has no checkpoint yet: starting it over in the same folder", flush=True)
    elif os.path.exists(os.path.join(run_dir, "train_log.tsv")) and not a.smoke:
        raise SystemExit(f"{run_dir} already has a run: choose another --run-tag, or continue it with --resume {a.run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    trace = snapshot_run(run_dir, a) if ck is None else None
    sf = a.split_file or default_split_file(a.corner)
    sel = select_designs(sf, ("train", "val"), a.val_only)
    rng = np.random.default_rng(a.seed)
    tr_names, va_names = sel["train"], sel["val"]
    if a.train_limit:
        tr_names = sorted(rng.choice(tr_names, a.train_limit, replace=False).tolist())
    if a.val_limit:
        va_names = sorted(rng.choice(va_names, a.val_limit, replace=False).tolist())
    t0 = time.time()
    print(f"run {a.run_tag}: loading {len(tr_names)} train + {len(va_names)} val designs", flush=True)
    stage = getattr(a, "label_stage", "synth")
    tr = load_designs(tr_names, root, a.corner, workers=a.load_workers, stage=stage)
    va = load_designs(va_names, root, a.corner, workers=a.load_workers, stage=stage)
    if a.shuffle_labels:
        tr = [shuffle_labels(d, a.seed) for d in tr]
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    vocab = load_vocab(root, a.corner)
    cfg = model_config(a)
    model = TimingEncoder(vocab, cfg).to(dev)
    if a.arc_init == "step2":
        model.arc.load_step2(a.step2_checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    arc_stats = None
    if model.layered and cfg.get("delay_norm") == "stats":
        from cell_model.data import load_bank as _lb
        mu, sd, n_arc = arc_target_stats(model, tr, _lb([], graph_root=root, corner=a.corner, n_val=0, n_test=0, device=dev),
                                         dev, a.arc_stats_designs, a.seed)
        model.encoder.set_arc_stats(mu.to(dev), sd.to(dev))
        arc_stats = {"mean": mu.tolist(), "std": sd.tolist(), "instances": n_arc, "designs": min(a.arc_stats_designs, len(tr))}
        print(f"  delay_norm stats from {arc_stats['designs']} train designs, {n_arc:,} arc instances: "
              f"delay {mu[0]:.2f} +- {sd[0]:.2f} ps, slew {mu[1]:.2f} +- {sd[1]:.2f} ps", flush=True)
    tr_b = DepthBatcher([d.levels for d in tr], [d.n_inst for d in tr], a.max_inst, a.max_designs, a.deep_levels, seed=a.seed)
    va_b = DepthBatcher([d.levels for d in va], [d.n_inst for d in va], a.eval_max_inst, 256, a.deep_levels, seed=a.seed)
    va_batches = va_b.epoch()
    first = tr_b.epoch()
    planned = len(first) * a.epochs
    with open(sf, "rb") as f:
        split_md5 = hashlib.md5(f.read()).hexdigest()
    biggest = max(tr, key=lambda d: d.n_inst)
    if ck is None:
        info = {"args": vars(a), "config_file": a.config and os.path.abspath(a.config), **trace,
                "non_default_args": non_default_args(a), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "model": cfg, "parameters": n_params,
                "split_file": sf, "split_md5": split_md5,
                "train_designs": len(tr), "val_designs": len(va), "val_only": a.val_only,
                "train_pin_transitions_labelled": int(sum(int(d.m_arr.sum()) for d in tr)),
                "train_batches_per_epoch": len(first), "train_level_steps_per_epoch": tr_b.cost(first),
                "val_batches": len(va_batches), "planned_steps": planned,
                "largest_train_design": [biggest.name, biggest.n_inst, biggest.levels],
                "deepest_train_design": max(((d.levels, d.name) for d in tr)), "lib_sha256": vocab.lib_sha256, "arc_stats": arc_stats,
                "host": os.uname().nodename}
    else:
        info = _read_json(os.path.join(run_dir, "config.json"))
        code_sha = code_manifest()[1]
        if (info.get("split_md5"), info.get("planned_steps")) != (split_md5, planned):
            raise SystemExit(f"{run_dir}: the split file or the batch plan changed since the run started, it cannot be continued")
        if info.get("code_sha256") and code_sha != info["code_sha256"]:
            print(f"warning: the code changed since the run started (code_sha256 {info['code_sha256'][:8]} -> {code_sha[:8]})", flush=True)
        info.setdefault("segments", []).append({
            "host": os.uname().nodename, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "from_epoch": ck["epoch"],
            "from_step": ck["step"], "code_sha256": code_sha})
    atomic_json(info, os.path.join(run_dir, "config.json"))
    print(json.dumps({k: v for k, v in info.items() if k != "args"}, indent=None), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lw = None
    if a.loss_weighting == "uncertainty":
        lw = LossWeights().to(dev)
        opt.add_param_group({"params": lw.parameters(), "weight_decay": 0.0})
    sampler = bank = None
    if a.aux_weight > 0 or model.layered:
        bank = load_bank([], graph_root=root, corner=a.corner, n_val=0, n_test=0, device=dev)
    if a.aux_weight > 0:
        sampler = Sampler(bank, "train", None, seed=a.seed)
    arc_bank = bank if model.layered else None
    resumed = ck is not None
    if resumed:
        for name in ("train_log.tsv", "diag.tsv"):
            if os.path.exists(os.path.join(run_dir, name)):
                truncate_tsv(os.path.join(run_dir, name), ck["epoch"])
    diag = Diag(run_dir, model, a.diag_every, a.explode_factor, append=resumed) if a.diag_every > 0 else None
    huber = nn.HuberLoss(delta=1.0)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    log = open(os.path.join(run_dir, "train_log.tsv"), "a" if resumed else "w")
    cols = ["epoch", "step", "lr", "train_loss", "train_arr_mae", "train_slew_mae", "train_slack_mae", "train_aux", "train_arc_mae",
            "val_arr_mae", "val_slew_mae", "val_slack_mae", "val_arc_mae", "val_loss", "val_arr_unreached", "oom", "skipped", "seconds", "gpu_mem_gb", "best", "gl", "no_improve"]
    if not resumed:
        log.write("\t".join(cols) + "\n")
    best = (math.inf, math.inf)
    best_ck_epoch = 0
    stopper = EarlyStopper(a.patience, a.min_delta, a.gl_alpha, a.gl_strips, a.min_epochs)
    stop_reason = ""
    step, start_epoch, elapsed = 0, 0, 0.0

    def rng_state() -> dict:
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                "numpy": np.random.get_state(), "python": random.getstate(), "batcher": tr_b.rng.bit_generator.state,
                "sampler": sampler.gen.get_state() if sampler is not None else None, "gen": gen.get_state(), "device": dev.type}

    if resumed:
        model.load_state_dict(ck["state_dict"])
        if lw is not None:
            lw.load_state_dict(ck["loss_weights"])
        R = ck["resume"]
        opt.load_state_dict(R["optimizer"])
        step, start_epoch, elapsed, stop_reason = ck["step"], ck["epoch"], R["elapsed"], R["stop_reason"]
        best, best_ck_epoch = tuple(R["best"]), R["best_ck_epoch"]
        vars(stopper).update(R["stopper"])
        r = R["rng"]
        torch.set_rng_state(r["torch"]); np.random.set_state(r["numpy"]); random.setstate(r["python"])
        tr_b.rng.bit_generator.state = r["batcher"]
        if r["device"] == dev.type:
            if r["cuda"] is not None:
                torch.cuda.set_rng_state_all(r["cuda"])
            if sampler is not None:
                sampler.gen.set_state(r["sampler"])
            gen.set_state(r["gen"])
        if diag is not None:
            diag.grads.recent.extend(R["diag_recent"])
            diag.event({"kind": "resume", "epoch": start_epoch, "step": step})
        bp = os.path.join(run_dir, "best.pt")
        held = torch.load(bp, map_location="cpu", weights_only=False)["epoch"] if os.path.exists(bp) else None
        if best_ck_epoch == start_epoch and held != best_ck_epoch:
            atomic_save({k: v for k, v in ck.items() if k != "resume"}, bp)
        ck = None
    t_start = time.time() - elapsed
    budget = a.max_hours * 3600.0

    def save(name, epoch, val):
        c = {"state_dict": model.state_dict(), "config": cfg, "args": vars(a), "epoch": epoch, "step": step,
             "val": val, "lib_sha256": vocab.lib_sha256}
        if lw is not None:
            c["loss_weights"] = lw.state_dict()
        if name == "last.pt":
            c["resume"] = {"optimizer": opt.state_dict(), "elapsed": time.time() - t_start, "stop_reason": stop_reason,
                           "best": list(best), "best_ck_epoch": best_ck_epoch, "stopper": dict(vars(stopper)), "rng": rng_state(),
                           "diag_recent": list(diag.grads.recent) if diag is not None else []}
        atomic_save(c, os.path.join(run_dir, name))

    def log_row(epoch, tr_stats, val, oom, is_best, skipped=0):
        n = lambda k: max(tr_stats.get(k + "_n", 0), 1)
        row = {"epoch": epoch, "step": step, "lr": f"{opt.param_groups[0]['lr']:.3g}",
               "train_loss": f"{tr_stats.get('loss', 0) / max(tr_stats.get('batches', 0), 1):.5g}",
               "train_arr_mae": f"{tr_stats.get('arr_sum', 0) / n('arr'):.5g}", "train_slew_mae": f"{tr_stats.get('slew_sum', 0) / n('slew'):.5g}",
               "train_slack_mae": f"{tr_stats.get('slack_sum', 0) / n('slack'):.5g}",
               "train_aux": f"{tr_stats.get('aux', 0) / max(tr_stats.get('batches', 0), 1):.5g}",
               "train_arc_mae": f"{tr_stats.get('arc_delay_sum', 0) / n('arc'):.5g}",
               "val_arr_mae": f"{val['arr_mae']:.5g}", "val_slew_mae": f"{val['slew_mae']:.5g}", "val_slack_mae": f"{val['slack_mae']:.5g}",
               "val_arc_mae": f"{val.get('arc_mae', 0):.5g}",
               "val_loss": f"{val['loss']:.5g}", "val_arr_unreached": val["arr_unreached"], "oom": oom, "skipped": skipped,
               "seconds": round(time.time() - t_start, 1),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0, "best": int(is_best),
               "gl": f"{stopper.gl(val['arr_mae']):.2f}", "no_improve": stopper.since}
        log.write("\t".join(str(row[c]) for c in cols) + "\n"); log.flush()
        print("  ".join(f"{c} {row[c]}" for c in cols), flush=True)

    if not resumed:
        val = evaluate(model, va, va_batches, dev, bool(a.knockout_tables), arc_bank)
        log_row(0, {}, val, 0, False)
    stop = False
    batches = tr_b.epoch() if resumed else first
    epoch = start_epoch
    last_epoch = start_epoch if stop_reason else a.epochs
    for epoch in range(start_epoch + 1, last_epoch + 1):
        st = {"loss": 0.0, "aux": 0.0, "batches": 0}
        oom, skipped, t_ep = [], 0, time.time()
        for ids in batches:
            progress = max(step / planned, (time.time() - t_start) / budget)
            for g in opt.param_groups:
                g["lr"] = a.lr * lr_factor(progress, step, a.warmup_steps, a.final_lr_frac)
            try:
                if diag is not None:
                    diag.before_batch(step, first=st["batches"] == 0)
                b = collate([tr[i] for i in ids], knockout_tables=bool(a.knockout_tables)).to(dev)
                out = model(b, keep=model.layered, tf_prob=a.tf_prob, generator=gen if a.tf_prob > 0 else None)
                loss, s = dense_losses(out, b, a.w_arr, a.w_slew, a.w_slack)
                terms = None
                if a.per_design_loss or lw is not None:
                    terms = dense_terms(out, b, per_design=bool(a.per_design_loss))
                if model.layered and a.w_arc > 0:
                    y_arc, m_arc = arc_delay_targets(out, b, arc_bank)
                    if a.per_design_loss:
                        k0 = len(out["schedule"].inst_o) - len(out["msg_delay"])
                        arc_loss, sa = arc_losses(out, y_arc, m_arc, b.node_graph[out["schedule"].inst_o[k0:]], len(b.names))
                    else:
                        arc_loss, sa = arc_losses(out, y_arc, m_arc)
                    s.update(sa)
                    if terms is not None:
                        terms["arc"] = arc_loss
                    else:
                        loss = loss + a.w_arc * arc_loss
                if terms is not None:
                    wts = {"arr": a.w_arr, "slew": a.w_slew, "slack": a.w_slack, "arc": a.w_arc}
                    loss = lw(terms, wts) if lw is not None else sum(wts[t] * v for t, v in terms.items() if wts[t])
                if sampler is not None:
                    key, trf, sl, c = sampler.synthetic(a.aux_batch)
                    y = sampler.bank.targets(key, trf, sl, c).float()
                    pred = model.arc.lookup(key, trf, sl.float(), c.float())
                    aux = huber((pred - y) / model.arc.scale[key, trf], torch.zeros_like(y))
                    loss = loss + a.aux_weight * aux
                    st["aux"] += float(aux)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip))
                if diag is not None and not diag.after_backward(step, epoch, total_norm, float(loss), b.names):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"non-finite loss or gradient, step skipped: loss {float(loss)}, grad norm {total_norm}, {b.names[:4]}", flush=True)
                    continue
                opt.step()
            except torch.cuda.OutOfMemoryError:
                oom.append([tr[i].name for i in ids])
                print(f"OOM, batch skipped: {len(ids)} designs, {sum(tr[i].n_inst for i in ids)} arc instances: {oom[-1][:4]}", flush=True)
                if diag is not None:
                    diag.event({"kind": "oom", "step": step, "epoch": epoch, "designs": oom[-1][:8]})
                b = out = loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            step += 1
            st["loss"] += float(loss); st["batches"] += 1
            for k, v in s.items():
                st[k] = st.get(k, 0) + v
            if time.time() - t_start > budget:
                stop = True
                break
        if epoch == 1:
            print(f"  epoch 1 took {time.time() - t_ep:.0f}s for {len(batches)} batches", flush=True)
        is_last = stop or epoch == a.epochs
        if diag is not None:
            diag.epoch(epoch, dict({"oom": len(oom), "skipped": skipped, "train_loss": st["loss"] / max(st["batches"], 1)},
                                   **(lw.scales() if lw is not None else {})))
        if epoch % a.eval_every == 0 or is_last:
            val = evaluate(model, va, va_batches, dev, bool(a.knockout_tables), arc_bank)
            key = (val["arr_mae"], val["loss"])
            is_best = key < best
            if is_best:
                best, best_ck_epoch = key, epoch
            stop_reason = stopper.update(epoch, val["arr_mae"])
            if stop:
                stop_reason = f"time limit ({a.max_hours} h)"
            log_row(epoch, st, val, len(oom), is_best, skipped)
            save("last.pt", epoch, val)
            if is_best:
                save("best.pt", epoch, val)
        if stop_reason:
            print(f"stopped after epoch {epoch}: {stop_reason}", flush=True)
            break
        batches = tr_b.epoch()
    log.close()
    if diag is not None:
        diag.close()
    stop_reason = stop_reason or f"all {a.epochs} epochs"
    atomic_json({"best_checkpoint": os.path.join(run_dir, "best.pt"), "best_epoch": stopper.best_epoch,
                 "best_val_arr_mae": best[0], "best_val_loss": best[1], "last_epoch": epoch, "stop_reason": stop_reason,
                 "steps": step, "hours": round((time.time() - t_start) / 3600, 3), "segments": 1 + len(info.get("segments", []))},
                os.path.join(run_dir, "result.json"))
    print(f"done ({stop_reason}): best val arrival MAE {best[0]:.5f} ps at epoch {stopper.best_epoch} -> "
          f"{os.path.join(run_dir, 'best.pt')}", flush=True)
    if not a.eval_after:
        return 0
    tr = va = model = opt = sampler = bank = arc_bank = None
    import gc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return evaluate_run(run_dir, a)


def evaluate_run(run_dir: str, a) -> int:
    import csv
    from . import eval_encoder
    res_path = os.path.join(run_dir, "result.json")
    res = _read_json(res_path)
    if res is None:
        raise SystemExit(f"{run_dir}: no result.json, the run has not finished")
    if "eval_val" in res:
        print(f"{run_dir}: already evaluated (result.json eval_val): right worst path {res['eval_val']['right_worst_path_pct']:.2f} %", flush=True)
        return 0
    out = os.path.join(run_dir, "eval_val")
    t0 = time.time()
    print(f"evaluating {run_dir}/best.pt on the val designs -> {out}", flush=True)
    argv = ["--source", "checkpoint", "--checkpoint", os.path.join(run_dir, "best.pt"), "--corner", a.corner, "--split", "val",
            "--out-dir", run_dir, "--tag", "eval_val", "--workers", str(max(a.eval_workers, 1))]
    if getattr(a, "label_stage", "synth") != "synth":
        argv += ["--label-stage", a.label_stage]
    if a.split_file:
        print("warning: eval_encoder reads the default split file, not --split-file", flush=True)
    if a.val_only:
        argv += ["--val-only", *a.val_only]
    if a.eval_limit:
        argv += ["--limit", str(a.eval_limit)]
    if a.val_limit:
        print(f"note: the run used --val-limit {a.val_limit}; the evaluation covers every val design", flush=True)
    code = eval_encoder.main(argv)
    with open(os.path.join(out, "designs.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    res["eval_val"] = dict(eval_encoder.scores(rows), designs_with_error=sum(1 for r in rows if r["error"]),
                           checkpoint_epoch=torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)["epoch"],
                           dir=out, minutes=round((time.time() - t0) / 60, 1))
    atomic_json(res, res_path)
    e = res["eval_val"]
    print(f"evaluation done in {e['minutes']} min: right worst path {e['right_worst_path_pct']:.2f} % ({e['right_worst_path']} of "
          f"{e['designs']}), right endpoint {e['right_endpoint_pct']:.2f} %, right path endpoint given "
          f"{e['right_path_endpoint_given_pct']:.2f} %, mean arrival error {e['arrival_mean_ps']:.3f} ps", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
