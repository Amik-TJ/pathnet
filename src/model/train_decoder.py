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

from paths import RUNS, resolve

from .data import DATA, DepthBatcher, collate, default_split_file, graph_root, select_designs
from .decoder import DECODER_DEFAULTS, DECODER_KINDS, SCORERS, PathDecoder, WalkIndex, decode_paths, decoder_losses, endpoint_losses, endpoint_pick, label_positives
from .decoder_data import collate_labels, load_designs_with_paths
from .encoder import TimingEncoder
from .heads import arc_delay_targets, arc_losses, dense_losses, dense_terms
from .margin import CLOSE_PS, MARGIN_DEFAULTS, margin_loss
from .seq_decoder import rerank_losses, seq_losses
from .train_encoder import (LossWeights, _read_json, atomic_json, atomic_save, code_manifest, find_run, load_checkpoint, lr_factor,
                            sha256_file, snapshot_run, truncate_tsv)

NOT_SETTINGS = ("run_tag", "out_root", "load_workers", "config", "device", "smoke", "resume", "eval_after", "eval_workers", "eval_limit")
COLS = ["epoch", "step", "lr", "train_loss", "train_walk", "train_ep", "train_dec_acc", "val_loss", "val_walk", "val_ep",
        "val_dec_acc", "val_ep_path", "val_endpoint", "val_worst_path", "val_worst_path_n", "temp", "ep_temp", "oom", "skipped",
        "seconds", "epoch_seconds", "gpu_mem_gb", "train_enc", "train_arr_mae", "val_arr_mae", "val_slew_mae", "val_slack_mae", "enc_lr", "best"]
ENC_LOSS_DEFAULTS = {"w_arr": 1.0, "w_slew": 1.0, "w_slack": 1.0, "w_arc": 1.0, "aux_weight": 1.0, "aux_batch": 4096,
                     "per_design_loss": 0, "loss_weighting": "fixed", "tf_prob": 0.0, "weight_decay": 0.0}
MARGIN_COLS = ["train_margin", "val_margin", "val_close_err", "val_rule_wrong", "val_dec_wrong"]
RERANK_COLS = ["train_rr", "val_rr", "val_rr_right", "val_rr_in_k"]


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
    ap.add_argument("--encoder-checkpoint", default=None)
    ap.add_argument("--train-encoder", type=int, default=0)
    ap.add_argument("--encoder-lr", type=float, default=None)
    ap.add_argument("--scorer", default=DECODER_DEFAULTS["scorer"], choices=tuple(SCORERS))
    ap.add_argument("--scorer-width", type=int, default=DECODER_DEFAULTS["scorer_width"])
    ap.add_argument("--scorer-depth", type=int, default=DECODER_DEFAULTS["scorer_depth"])
    ap.add_argument("--scorer-heads", type=int, default=DECODER_DEFAULTS["scorer_heads"])
    ap.add_argument("--temp", type=float, default=DECODER_DEFAULTS["temp"])
    ap.add_argument("--ep-temp", type=float, default=DECODER_DEFAULTS["ep_temp"])
    ap.add_argument("--score-chunk", type=int, default=DECODER_DEFAULTS["score_chunk"])
    ap.add_argument("--decoder-kind", default=DECODER_DEFAULTS["decoder_kind"], choices=DECODER_KINDS)
    ap.add_argument("--lookahead-hops", type=int, default=DECODER_DEFAULTS["lookahead_hops"], choices=(0, 1, 2))
    ap.add_argument("--seq-layers", type=int, default=DECODER_DEFAULTS["seq_layers"])
    ap.add_argument("--seq-width", type=int, default=DECODER_DEFAULTS["seq_width"])
    ap.add_argument("--seq-heads", type=int, default=DECODER_DEFAULTS["seq_heads"])
    ap.add_argument("--seq-window", type=int, default=DECODER_DEFAULTS["seq_window"])
    ap.add_argument("--seq-dropout", type=float, default=DECODER_DEFAULTS["seq_dropout"])
    ap.add_argument("--seq-head", default=DECODER_DEFAULTS["seq_head"], choices=("mlp", "bilinear"))
    ap.add_argument("--seq-tf-mix", type=float, default=DECODER_DEFAULTS["seq_tf_mix"])
    ap.add_argument("--ep-rerank-k", type=int, default=DECODER_DEFAULTS["ep_rerank_k"])
    ap.add_argument("--ep-rerank-layers", type=int, default=DECODER_DEFAULTS["ep_rerank_layers"])
    ap.add_argument("--ep-rerank-width", type=int, default=DECODER_DEFAULTS["ep_rerank_width"])
    ap.add_argument("--ep-referee", default=DECODER_DEFAULTS["ep_referee"], choices=("none", "path"))
    ap.add_argument("--ep-referee-rank", type=int, default=DECODER_DEFAULTS["ep_referee_rank"], choices=(0, 1))
    ap.add_argument("--beam", type=int, default=DECODER_DEFAULTS["beam"])
    ap.add_argument("--calibrate-batches", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--final-lr-frac", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--max-inst", type=int, default=300_000)
    ap.add_argument("--max-designs", type=int, default=64)
    ap.add_argument("--deep-levels", type=int, default=256)
    ap.add_argument("--eval-max-inst", type=int, default=1_500_000)
    ap.add_argument("--w-walk", type=float, default=1.0)
    ap.add_argument("--w-ep", type=float, default=1.0)
    ap.add_argument("--w-margin", type=float, default=MARGIN_DEFAULTS["w_margin"])
    ap.add_argument("--margin-eps", type=float, default=MARGIN_DEFAULTS["margin_eps"])
    ap.add_argument("--margin-transition", type=int, default=MARGIN_DEFAULTS["margin_transition"])
    ap.add_argument("--margin-max-gap", type=float, default=MARGIN_DEFAULTS["margin_max_gap"])
    ap.add_argument("--margin-log", type=int, default=MARGIN_DEFAULTS["margin_log"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=20.0)
    ap.add_argument("--load-workers", type=int, default=8)
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
    a.encoder_checkpoint = resolve(a.encoder_checkpoint)
    if a.smoke:
        a.train_limit, a.val_limit, a.epochs = a.train_limit or 4, a.val_limit or 4, min(a.epochs, 2)
        a.scorer_width, a.scorer_heads, a.warmup_steps = 16, 2, 2
        a.seq_width, a.seq_heads, a.ep_rerank_width = 16, 2, 16
    a.resume = run_dir
    if saved is not None:
        changed = {k: (saved[k], getattr(a, k)) for k in saved if k in known and k not in NOT_SETTINGS and saved[k] != getattr(a, k)}
        changed.update({k: (v, getattr(a, k)) for k, v in MARGIN_DEFAULTS.items() if k not in saved and getattr(a, k) != v})
        changed.update({k: (v, getattr(a, k)) for k, v in DECODER_DEFAULTS.items() if k not in saved and getattr(a, k) != v})
        if changed:
            raise SystemExit(f"--resume {run_dir}: these flags would change the run's settings (saved, given): {changed}")
        a.out_root, a.run_tag, a.config = os.path.dirname(run_dir), os.path.basename(run_dir), saved.get("config")
    if not a.encoder_checkpoint:
        ap.error("--encoder-checkpoint (or the config key encoder_checkpoint) is required")
    if a.w_margin < 0 or a.margin_eps <= 0 or a.margin_max_gap < 0:
        ap.error("--w-margin and --margin-max-gap must be >= 0 and --margin-eps > 0")
    if (a.w_margin > 0 or a.margin_log) and not a.train_encoder:
        ap.error("--w-margin > 0 / --margin-log 1 need --train-encoder 1: the margin loss trains the encoder, a frozen one has nothing to learn from it")
    if (a.ep_referee != "none" or a.ep_referee_rank) and a.ep_rerank_k <= 0:
        ap.error("--ep-referee / --ep-referee-rank need --ep-rerank-k > 0")
    a.run_tag = a.run_tag or time.strftime("dec_%Y%m%d_%H%M%S")
    return a


def head_parts(dec) -> tuple:
    sd = dec.state_dict()
    w = {k: v.detach().cpu().clone() for k, v in sd.items() if k.startswith("walk.") or k == "log_t"}
    e = {k: v.detach().cpu().clone() for k, v in sd.items() if k.startswith(("ep.", "ln_e.", "ep_rr.")) or k == "log_t_ep"}
    if len(w) + len(e) != len(sd):
        raise RuntimeError(f"decoder parameters outside the two parts: {sorted(set(sd) - set(w) - set(e))}")
    return w, e


def encoder_loss_settings(eck: dict) -> dict:
    ea = eck.get("args") or {}
    return {k: ea.get(k, v) for k, v in ENC_LOSS_DEFAULTS.items()}


def encoder_loss(model, out: dict, b, es: dict, lw, arc_bank, sampler, huber) -> tuple:
    loss, s = dense_losses(out, b, es["w_arr"], es["w_slew"], es["w_slack"])
    terms = None
    if es["per_design_loss"] or lw is not None:
        terms = dense_terms(out, b, per_design=bool(es["per_design_loss"]))
    if model.layered and es["w_arc"] > 0:
        y_arc, m_arc = arc_delay_targets(out, b, arc_bank)
        if es["per_design_loss"]:
            k0 = len(out["schedule"].inst_o) - len(out["msg_delay"])
            arc_loss, sa = arc_losses(out, y_arc, m_arc, b.node_graph[out["schedule"].inst_o[k0:]], len(b.names))
        else:
            arc_loss, sa = arc_losses(out, y_arc, m_arc)
        s.update(sa)
        if terms is not None:
            terms["arc"] = arc_loss
        else:
            loss = loss + es["w_arc"] * arc_loss
    if terms is not None:
        wts = {"arr": es["w_arr"], "slew": es["w_slew"], "slack": es["w_slack"], "arc": es["w_arc"]}
        loss = lw(terms, wts) if lw is not None else sum(wts[t] * v for t, v in terms.items() if wts[t])
    if sampler is not None:
        key, trf, sl, c = sampler.synthetic(es["aux_batch"])
        y = sampler.bank.targets(key, trf, sl, c).float()
        pred = model.arc.lookup(key, trf, sl.float(), c.float())
        aux = huber((pred - y) / model.arc.scale[key, trf], torch.zeros_like(y))
        loss = loss + es["aux_weight"] * aux
        s["aux"] = float(aux)
    return loss, s


def decoder_config(a) -> dict:
    return {k: getattr(a, k) for k in DECODER_DEFAULTS}


def load_decoder_checkpoint(path: str, vocab, device="cpu", bank=None) -> tuple:
    ck = torch.load(path, map_location=device, weights_only=False)
    enc = TimingEncoder(vocab, ck["encoder_config"]).to(device)
    enc.load_state_dict(ck["encoder_state"])
    dec = PathDecoder(ck["dz"], ck["decoder_config"]).to(device)
    dec.load_state_dict(ck["decoder_state"])
    if dec.config.get("ep_referee", "none") == "path":
        if bank is None:
            from cell_model.data import load_bank
            from .data import graph_root
            bank = load_bank([], graph_root=graph_root(), n_val=0, n_test=0, device=device)
        dec.bank = bank
    enc.eval(); dec.eval()
    return enc, dec, ck


CAL_GRID = tuple(float(x) for x in np.geomspace(0.005, 5.0, 31))


@torch.no_grad()
def calibrate_temps(enc, dec, designs, labels, batches, device) -> dict:
    tot = np.zeros((len(CAL_GRID), 4))
    start = (float(dec.log_t), float(dec.log_t_ep))
    for ids in batches:
        b = collate([designs[i] for i in ids]).to(device)
        lab = collate_labels([labels[i] for i in ids], b).to(device)
        out = enc(b, keep=True)
        pos = label_positives(out, lab, b.num_nodes)
        none = torch.zeros_like(pos["lab_seg"])
        for i, t in enumerate(CAL_GRID):
            dec.log_t.fill_(math.log(t)); dec.log_t_ep.fill_(math.log(t))
            _, _, st = decoder_losses(dec(out, b, only_seg=none), out, b, lab, pos)
            tot[i] += (st["walk_sum"], st["walk_n"], st["ep_sum"], st["ep_n"])
    walk, ep = tot[:, 0] / np.maximum(tot[:, 1], 1), tot[:, 2] / np.maximum(tot[:, 3], 1)
    iw, ie = int(np.argmin(walk)), int(np.argmin(ep))
    dec.log_t.fill_(math.log(CAL_GRID[iw]) if tot[0, 1] else start[0])
    dec.log_t_ep.fill_(math.log(CAL_GRID[ie]) if tot[0, 3] else start[1])
    return {"temp": float(torch.exp(dec.log_t)), "ep_temp": float(torch.exp(dec.log_t_ep)), "batches": len(batches),
            "decisions": int(tot[0, 1]), "designs": int(tot[0, 3]), "grid_ps": list(CAL_GRID),
            "walk_ce": [round(float(x), 6) for x in walk], "ep_ce": [round(float(x), 6) for x in ep],
            "at_grid_edge": bool(iw in (0, len(CAL_GRID) - 1) or ie in (0, len(CAL_GRID) - 1))}


@torch.no_grad()
def evaluate(enc, dec, designs, labels, batches, device, margin: dict | None = None) -> dict:
    was_training = enc.training
    enc.eval(); dec.eval()
    t = {k: 0.0 for k in ("walk_sum", "walk_n", "ep_sum", "ep_n", "dec_right", "dec_n", "ep_right", "ep_designs",
                          "path_right", "paths", "worst_right", "designs", "unmatched", "fallback", "decisions",
                          "arr_sum", "arr_n", "slew_sum", "slew_n", "slack_sum", "slack_n",
                          "rr_sum", "rr_n", "rr_right", "rr_designs", "rr_skipped", "rr_in_k")}
    mt, gaps = {}, []
    for bi, ids in enumerate(batches):
        b = collate([designs[i] for i in ids]).to(device)
        lab = collate_labels([labels[i] for i in ids], b).to(device)
        out = enc(b, keep=True)
        d = dec(out, b)
        n_ep = len(b.ep_index)
        prio = torch.randperm(n_ep, generator=torch.Generator().manual_seed(7919 * bi + n_ep)).to(device)
        seq = dec.kind != "pointer"
        if not seq:
            _, _, st = decoder_losses(d, out, b, lab, ep_priority=prio)
        else:
            pos = label_positives(out, lab, b.num_nodes)
            _, st, det_s = seq_losses(dec, out, b, lab, pos)
            _, es = endpoint_losses(d["ep_score"], b, lab, prio)
            st.update(es)
        for k, v in st.items():
            t[k] += v
        _, ds = dense_losses(out, b)
        for k in ("arr_sum", "arr_n", "slew_sum", "slew_n", "slack_sum", "slack_n"):
            t[k] += ds[k]
        wi = WalkIndex(out, b)
        best = wi.best(d["logit"]) if not seq else None
        if margin is not None:
            pos = label_positives(out, lab, b.num_nodes)
            _, ms, det = margin_loss(out, b, pos, *arc_delay_targets(out, b, margin["bank"]), margin["eps"], bool(margin["transition"]), margin["max_gap"])
            for k, v in ms.items():
                mt[k] = mt.get(k, 0) + v
            gaps.append(torch.stack([det["true_gap"], det["pred_gap"].double()], 1).cpu())
            if not seq:
                chosen = best[det["seg"]]
                mt["dec_wrong"] = mt.get("dec_wrong", 0) + int(((chosen < 0) | ~pos["pin_pos"][chosen.clamp_min(0)]).sum())
            else:
                mt["dec_wrong"] = mt.get("dec_wrong", 0) + int(det_s["seg_wrong"][det["seg"]].sum())
        pick = endpoint_pick(d["ep_score"][b.ep_index], b.ep_graph, len(b.names), prio)
        if dec.rerank_k > 0:
            rr = dec.rerank(out, b, d, b.ep_index, b.ep_graph, len(b.names), prio)
            _, rs = rerank_losses(rr, lab.ep_worst[rr["sel"]], len(b.names))
            for k, v in rs.items():
                t[k] += v
            pick = rr["pick"]
            t["ep_right"] += rs["rr_right"] - st["ep_right"]
        pk = pick[pick >= 0]
        length = lab.tok_ptr[1:] - lab.tok_ptr[:-1]
        starts = (torch.cat([b.ep_index, b.ep_index[pk]]), torch.cat([lab.ep_rf, out["ep_rf"][b.ep_index[pk]]]),
                  lab.tok, torch.cat([lab.tok_ptr[:-1], lab.tok_ptr[pk]]), torch.cat([length, length[pk]]))
        r = (decode_paths(wi, best, *starts) if not seq else dec.decode(out, b, d["logit"], *starts))["right"]
        t["path_right"] += int(r[:n_ep].sum()); t["paths"] += n_ep
        t["worst_right"] += int((r[n_ep:] & lab.ep_worst[pk]).sum()); t["designs"] += len(b.names)
    dec.train(); enc.train(was_training)
    div = lambda x, n: t[x] / max(t[n], 1)
    res = {"arr_mae": div("arr_sum", "arr_n"), "slew_mae": div("slew_sum", "slew_n"), "slack_mae": div("slack_sum", "slack_n"),
           "walk": div("walk_sum", "walk_n"), "ep": div("ep_sum", "ep_n"), "dec_acc": div("dec_right", "dec_n"),
           "ep_path": div("path_right", "paths"), "endpoint": div("ep_right", "designs"), "worst_path": div("worst_right", "designs"),
           "worst_path_n": int(t["worst_right"]), "designs": int(t["designs"]), "paths": int(t["paths"]),
           "decisions": int(t["decisions"]), "scored_decisions": int(t["dec_n"]), "fallback": int(t["fallback"]), "unmatched": int(t["unmatched"])}
    if dec.rerank_k > 0:
        res.update(rr=div("rr_sum", "rr_n"), rr_right=int(t["rr_right"]), rr_in_k=int(t["rr_in_k"]), rr_skipped=int(t["rr_skipped"]))
    if margin is not None:
        g = torch.cat(gaps).numpy() if gaps else np.zeros((0, 2))
        close = g[:, 0] < CLOSE_PS
        res.update(margin=mt.get("margin_sum", 0.0) / max(mt.get("margin_n", 0.0), 1e-30), close_n=int(close.sum()),
                   close_err=float(np.median(np.abs(g[close, 1] - g[close, 0]))) if close.any() else float("nan"),
                   rule_wrong=int((g[:, 1] <= 0).sum()), dec_wrong=int(mt.get("dec_wrong", 0)), margin_pin_decisions=len(g),
                   **{k: int(mt.get(k, 0)) for k in ("margin_decisions", "margin_rivals", "margin_not_max", "margin_no_true", "margin_skipped")})
    return res


def main(argv=None):
    a = build_args(argv)
    from graph.vocab import load_vocab
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
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
    print(f"run {a.run_tag}: loading {len(tr_names)} train + {len(va_names)} val designs ({a.label_stage} labels)", flush=True)
    tr, trl = load_designs_with_paths(tr_names, root, a.corner, a.load_workers, stage=a.label_stage)
    va, val_l = load_designs_with_paths(va_names, root, a.corner, a.load_workers, stage=a.label_stage)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    vocab = load_vocab(root, a.corner)
    joint = bool(a.train_encoder)
    enc, eck = load_checkpoint(a.encoder_checkpoint, vocab, dev)
    es = encoder_loss_settings(eck)
    for p in enc.parameters():
        p.requires_grad_(joint)
    dcfg = decoder_config(a)
    dec = PathDecoder(enc.dz, dcfg).to(dev)
    n_params = sum(p.numel() for p in dec.parameters())
    lw = sampler = arc_bank = None
    margin = None
    cols = COLS + MARGIN_COLS if a.w_margin > 0 or a.margin_log else COLS
    rerank = a.ep_rerank_k > 0
    if rerank:
        cols = cols + RERANK_COLS
    huber = nn.HuberLoss(delta=1.0)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if joint:
        from cell_model.data import Sampler, load_bank
        if es["loss_weighting"] == "uncertainty":
            lw = LossWeights().to(dev)
            if eck.get("loss_weights") is not None:
                lw.load_state_dict(eck["loss_weights"])
        if es["aux_weight"] > 0 or enc.layered or a.w_margin > 0 or a.margin_log:
            bank = load_bank([], graph_root=root, corner=a.corner, n_val=0, n_test=0, device=dev)
            sampler = Sampler(bank, "train", None, seed=a.seed) if es["aux_weight"] > 0 else None
            arc_bank = bank if enc.layered else None
            if a.w_margin > 0 or a.margin_log:
                margin = {"bank": bank, "eps": a.margin_eps, "transition": a.margin_transition, "max_gap": a.margin_max_gap}
    if a.ep_referee == "path":
        from cell_model.data import load_bank
        dec.bank = margin["bank"] if margin is not None else load_bank([], graph_root=root, corner=a.corner, n_val=0, n_test=0, device=dev)
    tr_b = DepthBatcher([d.levels for d in tr], [d.n_inst for d in tr], a.max_inst, a.max_designs, a.deep_levels, seed=a.seed)
    va_b = DepthBatcher([d.levels for d in va], [d.n_inst for d in va], a.eval_max_inst, 256, a.deep_levels, seed=a.seed)
    va_batches = va_b.epoch()
    first = tr_b.epoch()
    planned = len(first) * a.epochs
    with open(sf, "rb") as f:
        split_md5 = hashlib.md5(f.read()).hexdigest()
    if ck is None:
        cal = None
        if a.calibrate_batches > 0:
            cal = calibrate_temps(enc, dec, tr, trl, first[:a.calibrate_batches], dev)
            print(f"  temperatures fitted on {cal['batches']} training batches ({cal['decisions']:,} decisions, {cal['designs']} designs): "
                  f"pointer {cal['temp']:.4g} ps, endpoint {cal['ep_temp']:.4g} ps" + ("  WARNING: at the edge of the grid" if cal["at_grid_edge"] else ""), flush=True)
        info = {"args": vars(a), "config_file": a.config and os.path.abspath(a.config), **trace, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_encoder": int(joint), "encoder_loss": es if joint else None, "encoder_parameters": sum(p.numel() for p in enc.parameters()),
                "decoder": dcfg, "calibration": cal, "decoder_parameters": n_params, "encoder_checkpoint": os.path.abspath(a.encoder_checkpoint),
                "encoder_checkpoint_sha256": sha256_file(a.encoder_checkpoint), "encoder_epoch": eck.get("epoch"),
                "encoder_arch": eck["config"].get("arch", "mlp"), "encoder": eck["config"], "split_file": sf, "split_md5": split_md5,
                "train_designs": len(tr), "val_designs": len(va), "val_only": a.val_only,
                "train_endpoint_paths": int(sum(len(l.ep_worst) for l in trl)), "train_labelled_decisions": int(sum(len(l.dec_ft) for l in trl)),
                "train_batches_per_epoch": len(first), "val_batches": len(va_batches), "planned_steps": planned,
                "lib_sha256": vocab.lib_sha256, "host": os.uname().nodename}
    else:
        info = _read_json(os.path.join(run_dir, "config.json"))
        code_sha = code_manifest()[1]
        if (info.get("split_md5"), info.get("planned_steps")) != (split_md5, planned):
            raise SystemExit(f"{run_dir}: the split file or the batch plan changed since the run started, it cannot be continued")
        if info.get("encoder_checkpoint_sha256") != sha256_file(a.encoder_checkpoint):
            raise SystemExit(f"{run_dir}: the encoder checkpoint changed since the run started, it cannot be continued")
        if info.get("code_sha256") and code_sha != info["code_sha256"]:
            print(f"warning: the code changed since the run started (code_sha256 {info['code_sha256'][:8]} -> {code_sha[:8]})", flush=True)
        info.setdefault("segments", []).append({
            "host": os.uname().nodename, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "from_epoch": ck["epoch"],
            "from_step": ck["step"], "code_sha256": code_sha})
    atomic_json(info, os.path.join(run_dir, "config.json"))
    print(json.dumps({k: v for k, v in info.items() if k not in ("args", "encoder")}, indent=None), flush=True)

    enc_lr = a.lr if a.encoder_lr is None else a.encoder_lr
    groups, base_lrs = [{"params": list(dec.parameters()), "weight_decay": a.weight_decay}], [a.lr]
    if joint:
        groups.append({"params": list(enc.parameters()), "weight_decay": es["weight_decay"]}); base_lrs.append(enc_lr)
        if lw is not None:
            groups.append({"params": list(lw.parameters()), "weight_decay": 0.0}); base_lrs.append(enc_lr)
    opt = torch.optim.AdamW(groups, lr=a.lr)
    clip_params = list(dec.parameters()) + (list(enc.parameters()) if joint else [])
    resumed = ck is not None
    if resumed and os.path.exists(os.path.join(run_dir, "train_log.tsv")):
        truncate_tsv(os.path.join(run_dir, "train_log.tsv"), ck["epoch"])
    log = open(os.path.join(run_dir, "train_log.tsv"), "a" if resumed else "w")
    if not resumed:
        log.write("\t".join(cols) + "\n")
    best_w = best_e = best_j = None
    epoch_w = epoch_e = 0
    part_w = part_e = part_enc = None
    step, start_epoch, elapsed, stop_reason = 0, 0, 0.0, ""
    if resumed:
        dec.load_state_dict(ck["decoder_state"])
        if joint:
            enc.load_state_dict(ck["encoder_state"])
        R = ck["resume"]
        if lw is not None and R.get("loss_weights") is not None:
            lw.load_state_dict(R["loss_weights"])
        opt.load_state_dict(R["optimizer"])
        step, start_epoch, elapsed, stop_reason = ck["step"], ck["epoch"], R["elapsed"], R["stop_reason"]
        best_w, best_e, epoch_w, epoch_e = tuple(R["best_w"]), tuple(R["best_e"]), R["epoch_w"], R["epoch_e"]
        part_w, part_e, part_enc = R["part_w"], R["part_e"], R.get("part_enc")
        best_j = tuple(R["best_j"]) if R.get("best_j") is not None else None
        r = R["rng"]
        torch.set_rng_state(r["torch"]); np.random.set_state(r["numpy"]); random.setstate(r["python"])
        tr_b.rng.bit_generator.state = r["batcher"]
        if r.get("device") == dev.type:
            if r.get("cuda") is not None:
                torch.cuda.set_rng_state_all(r["cuda"])
            if sampler is not None and r.get("sampler") is not None:
                sampler.gen.set_state(r["sampler"])
            if r.get("gen") is not None:
                gen.set_state(r["gen"])
    t_start = time.time() - elapsed
    budget = a.max_hours * 3600.0

    def save(name, epoch, val):
        last = name == "last.pt"
        c = {"decoder_state": dec.state_dict() if last else {**dec.state_dict(), **part_w, **part_e}, "decoder_config": dcfg,
             "dz": enc.dz, "encoder_state": enc.state_dict() if (last or not joint) else part_enc, "encoder_config": eck["config"],
             "train_encoder": int(joint),
             "encoder_checkpoint": os.path.abspath(a.encoder_checkpoint), "encoder_epoch": eck.get("epoch"), "args": vars(a),
             "epoch": epoch, "pointer_epoch": epoch if last else epoch_w, "endpoint_epoch": epoch if last else epoch_e,
             "step": step, "val": val, "lib_sha256": vocab.lib_sha256}
        if last:
            c["resume"] = {"optimizer": opt.state_dict(), "elapsed": time.time() - t_start, "stop_reason": stop_reason,
                           "best_w": list(best_w), "best_e": list(best_e), "epoch_w": epoch_w, "epoch_e": epoch_e,
                           "part_w": part_w, "part_e": part_e, "part_enc": part_enc,
                           "best_j": list(best_j) if best_j is not None else None,
                           "loss_weights": lw.state_dict() if lw is not None else None,
                           "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                                   "numpy": np.random.get_state(), "python": random.getstate(), "batcher": tr_b.rng.bit_generator.state,
                                   "sampler": sampler.gen.get_state() if sampler is not None else None, "gen": gen.get_state(),
                                   "device": dev.type}}
        atomic_save(c, os.path.join(run_dir, name))

    if resumed:
        save("best.pt", start_epoch, ck.get("val"))
        ck = None

    def val_loss(v):
        return a.w_walk * v["walk"] + a.w_ep * v["ep"]

    def key_walk(v):
        return (v["ep_path"], v["dec_acc"], -v["walk"])

    def key_ep(v):
        return (v["endpoint"], -v["ep"])

    def key_joint(v):
        return (v["worst_path_n"], v["ep_path"], -val_loss(v))

    def enc_copy():
        return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}

    def log_row(epoch, st, val, oom, skipped, is_best, ep_seconds):
        nb = max(st.get("batches", 0), 1)
        t, t_ep = dec.temps()
        row = {"epoch": epoch, "step": step, "lr": f"{opt.param_groups[0]['lr']:.3g}", "train_loss": f"{st.get('loss', 0) / nb:.5g}",
               "train_walk": f"{st.get('walk_sum', 0) / max(st.get('walk_n', 0), 1):.5g}", "train_ep": f"{st.get('ep_sum', 0) / max(st.get('ep_n', 0), 1):.5g}",
               "train_dec_acc": f"{st.get('dec_right', 0) / max(st.get('dec_n', 0), 1):.6f}",
               "val_loss": f"{val_loss(val):.5g}", "val_walk": f"{val['walk']:.5g}", "val_ep": f"{val['ep']:.5g}",
               "val_dec_acc": f"{val['dec_acc']:.6f}", "val_ep_path": f"{val['ep_path']:.6f}", "val_endpoint": f"{val['endpoint']:.6f}",
               "val_worst_path": f"{val['worst_path']:.6f}", "val_worst_path_n": val["worst_path_n"],
               "temp": f"{float(t):.4g}", "ep_temp": f"{float(t_ep):.4g}", "oom": oom, "skipped": skipped,
               "seconds": round(time.time() - t_start, 1), "epoch_seconds": round(ep_seconds, 1),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0,
               "train_enc": f"{st.get('enc', 0) / nb:.5g}", "train_arr_mae": f"{st.get('arr_sum', 0) / max(st.get('arr_n', 0), 1):.5g}",
               "val_arr_mae": f"{val['arr_mae']:.5g}", "val_slew_mae": f"{val['slew_mae']:.5g}", "val_slack_mae": f"{val['slack_mae']:.5g}",
               "enc_lr": f"{opt.param_groups[1]['lr']:.3g}" if joint else "0", "best": is_best or "-"}
        if margin is not None:
            row.update(train_margin=f"{st.get('margin_sum', 0) / max(st.get('margin_n', 0), 1e-30):.5g}", val_margin=f"{val['margin']:.5g}",
                       val_close_err=f"{val['close_err']:.5g}", val_rule_wrong=val["rule_wrong"], val_dec_wrong=val["dec_wrong"])
        if rerank:
            row.update(train_rr=f"{st.get('rr_sum', 0) / max(st.get('rr_n', 0), 1e-30):.5g}", val_rr=f"{val['rr']:.5g}",
                       val_rr_right=val["rr_right"], val_rr_in_k=val["rr_in_k"])
        log.write("\t".join(str(row[c]) for c in cols) + "\n"); log.flush()
        print("  ".join(f"{c} {row[c]}" for c in cols), flush=True)

    if not resumed:
        t_ev = time.time()
        val = evaluate(enc, dec, va, val_l, va_batches, dev, margin)
        print(f"  val: {val['designs']} designs, {val['paths']:,} endpoint paths, {val['decisions']:,} distinct labelled decisions "
              f"({val['scored_decisions']:,} path steps with >= 2 candidate pins; {val['fallback']} matched by pin only, {val['unmatched']} unmatched); "
              f"evaluation {time.time() - t_ev:.0f}s", flush=True)
        if joint:
            stored = (eck.get("val") or {}).get("arr_mae")
            print(f"  encoder at the start: val arrival error {val['arr_mae']:.5f} ps (its checkpoint recorded "
                  f"{stored if stored is None else format(stored, '.5f')}), slew {val['slew_mae']:.5f}, endpoint slack {val['slack_mae']:.5f}; "
                  f"its own loss settings: {es}", flush=True)
            best_j, part_enc = key_joint(val), enc_copy()
        if margin is not None:
            print(f"  race-margin loss, weight {a.w_margin} (eps {a.margin_eps} ps, transition {a.margin_transition}, max gap {a.margin_max_gap}): on val "
                  f"L_margin {val['margin']:.5f} next to the arrival error {val['arr_mae']:.5f} ps; {val['margin_decisions']:,} decisions and "
                  f"{val['margin_rivals']:,} rivals in the loss; {val['margin_pin_decisions']:,} decisions with a rival pin, {val['close_n']:,} of them races "
                  f"below {CLOSE_PS:g} ps, close-race margin error {val['close_err']:.5f} ps, the rule wrong at {val['rule_wrong']}; label not the true "
                  f"maximum at {val['margin_not_max']}; left out for want of a true score: {val['margin_no_true']} candidates, "
                  f"{val['margin_skipped']} decisions", flush=True)
        if rerank:
            print(f"  endpoint re-ranker, k = {a.ep_rerank_k}, referee {a.ep_referee}{' (rank only)' if a.ep_referee_rank else ''}: on val the true worst "
                  f"endpoint is among the k in {val['rr_in_k']} of {val['designs']} designs; untrained it picks {val['rr_right']} right (the head's pick)", flush=True)
        best_w, best_e = key_walk(val), key_ep(val)
        part_w, part_e = head_parts(dec)
        log_row(0, {}, val, 0, 0, "j" if joint else "pe", 0.0)
        save("last.pt", 0, val)
        save("best.pt", 0, val)
    stop = False
    batches = tr_b.epoch() if resumed else first
    epoch = start_epoch
    last_epoch = start_epoch if stop_reason else a.epochs
    for epoch in range(start_epoch + 1, last_epoch + 1):
        st = {"loss": 0.0, "enc": 0.0, "batches": 0}
        oom, skipped, t_ep = 0, 0, time.time()
        enc.train(joint)
        for ids in batches:
            progress = max(step / max(planned, 1), (time.time() - t_start) / budget)
            factor = lr_factor(progress, step, a.warmup_steps, a.final_lr_frac)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * factor
            try:
                b = collate([tr[i] for i in ids]).to(dev)
                lab = collate_labels([trl[i] for i in ids], b).to(dev)
                if joint:
                    out = enc(b, keep=True, tf_prob=es["tf_prob"], generator=gen if es["tf_prob"] > 0 else None)
                    enc_loss, s_enc = encoder_loss(enc, out, b, es, lw, arc_bank, sampler, huber)
                else:
                    with torch.no_grad():
                        out = enc(b, keep=True)
                    enc_loss, s_enc = 0.0, {}
                pos = label_positives(out, lab, b.num_nodes)
                if dec.kind == "pointer":
                    d = dec(out, b, only_seg=pos["lab_seg"])
                    walk_loss, ep_loss, s = decoder_losses(d, out, b, lab, pos)
                else:
                    d = dec(out, b)
                    walk_loss, s, _ = seq_losses(dec, out, b, lab, pos, sample=gen)
                    ep_loss, s_ep = endpoint_losses(d["ep_score"], b, lab)
                    s.update(s_ep)
                loss = enc_loss + a.w_walk * walk_loss + a.w_ep * ep_loss
                if rerank:
                    prio = torch.randperm(len(b.ep_index), device=dev, generator=gen)
                    rr = dec.rerank(out, b, d, b.ep_index, b.ep_graph, len(b.names), prio)
                    rr_loss, s_rr = rerank_losses(rr, lab.ep_worst[rr["sel"]], len(b.names), learn=not dec.ep_rr.rank_only)
                    loss = loss + a.w_ep * rr_loss
                    s.update(s_rr)
                if margin is not None:
                    with torch.set_grad_enabled(a.w_margin > 0):
                        m_loss, s_m, _ = margin_loss(out, b, pos, *arc_delay_targets(out, b, margin["bank"]), a.margin_eps,
                                                     bool(a.margin_transition), a.margin_max_gap)
                    if a.w_margin > 0:
                        loss = loss + a.w_margin * m_loss
                    s.update(s_m)
                s.update(s_enc)
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    skipped += 1
                    print(f"batch skipped (loss {float(loss)}, nothing to learn from or not finite): {b.names[:4]}", flush=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_norm = float(nn.utils.clip_grad_norm_(clip_params, a.grad_clip))
                if not math.isfinite(total_norm):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"non-finite gradient, step skipped: {b.names[:4]}", flush=True)
                    continue
                opt.step()
            except torch.cuda.OutOfMemoryError:
                oom += 1
                print(f"OOM, batch skipped: {len(ids)} designs, {sum(tr[i].n_inst for i in ids)} arc instances: {[tr[i].name for i in ids][:4]}", flush=True)
                b = lab = out = d = loss = enc_loss = walk_loss = ep_loss = m_loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            step += 1
            st["loss"] += float(loss); st["enc"] += float(enc_loss); st["batches"] += 1
            for k, v in s.items():
                st[k] = st.get(k, 0) + v
            if time.time() - t_start > budget:
                stop = True
                break
        ep_seconds = time.time() - t_ep
        if epoch == start_epoch + 1:
            print(f"  epoch {epoch} took {ep_seconds:.0f}s for {len(batches)} batches", flush=True)
        val = evaluate(enc, dec, va, val_l, va_batches, dev, margin)
        pw, pe = head_parts(dec)
        if joint:
            new_w = new_e = key_joint(val) > best_j
            if new_w:
                best_j, epoch_w, epoch_e, part_w, part_e, part_enc = key_joint(val), epoch, epoch, pw, pe, enc_copy()
            flag = "j" if new_w else ""
        else:
            new_w, new_e = key_walk(val) > best_w, key_ep(val) > best_e
            if new_w:
                best_w, epoch_w, part_w = key_walk(val), epoch, pw
            if new_e:
                best_e, epoch_e, part_e = key_ep(val), epoch, pe
            flag = ("p" if new_w else "") + ("e" if new_e else "")
        if stop:
            stop_reason = f"time limit ({a.max_hours} h)"
        log_row(epoch, st, val, oom, skipped, flag, ep_seconds)
        save("last.pt", epoch, val)
        if new_w or new_e:
            save("best.pt", epoch, val)
        if stop_reason:
            print(f"stopped after epoch {epoch}: {stop_reason}", flush=True)
            break
        batches = tr_b.epoch()
    log.close()
    stop_reason = stop_reason or f"all {a.epochs} epochs"
    import csv
    with open(os.path.join(run_dir, "train_log.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    keep = ("val_worst_path_n", "val_endpoint", "val_ep_path", "val_dec_acc", "val_loss", "val_arr_mae")
    if margin is not None:
        keep += ("val_margin", "val_close_err", "val_rule_wrong", "val_dec_wrong")
    if rerank:
        keep += ("val_rr", "val_rr_right", "val_rr_in_k")
    pick = lambda r: {k: float(r[k]) for k in keep}
    by_epoch = {int(r["epoch"]): r for r in rows}
    tail = [float(r["val_worst_path_n"]) for r in rows if int(r["epoch"]) > 0][-10:]
    if joint:
        enc.load_state_dict(part_enc)
    dec.load_state_dict({**dec.state_dict(), **part_w, **part_e})
    vb = evaluate(enc, dec, va, val_l, va_batches, dev, margin)
    val_best = {"val_worst_path_n": float(vb["worst_path_n"]), "val_endpoint": vb["endpoint"], "val_ep_path": vb["ep_path"],
                "val_dec_acc": vb["dec_acc"], "val_loss": val_loss(vb), "val_arr_mae": vb["arr_mae"]}
    if margin is not None:
        val_best.update(val_margin=vb["margin"], val_close_err=vb["close_err"], val_rule_wrong=float(vb["rule_wrong"]), val_dec_wrong=float(vb["dec_wrong"]))
    if rerank:
        val_best.update(val_rr=vb["rr"], val_rr_right=float(vb["rr_right"]), val_rr_in_k=float(vb["rr_in_k"]))
    atomic_json({"best_checkpoint": os.path.join(run_dir, "best.pt"), "train_encoder": int(joint), "pointer_epoch": epoch_w, "endpoint_epoch": epoch_e,
                 "val_designs": len(va_names), "val_untrained_epoch0": pick(by_epoch[0]), "val_best": val_best, "val_last": pick(rows[-1]),
                 "val_worst_path_n_mean_last10": round(sum(tail) / len(tail), 2) if tail else None,
                 "last_epoch": epoch, "stop_reason": stop_reason, "steps": step, "hours": round((time.time() - t_start) / 3600, 3),
                 "segments": 1 + len(info.get("segments", []))}, os.path.join(run_dir, "result.json"))
    print(f"done ({stop_reason}): right worst paths on val (GPU read-out, of {len(va_names)}): untrained {by_epoch[0]['val_worst_path_n']}, "
          f"best.pt {vb['worst_path_n']} (pointer of epoch {epoch_w}, endpoint head of epoch {epoch_e}), last epoch {rows[-1]['val_worst_path_n']} "
          f"-> {os.path.join(run_dir, 'best.pt')}", flush=True)
    if not a.eval_after:
        return 0
    tr = trl = va = val_l = enc = dec = opt = sampler = arc_bank = lw = margin = None
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
    argv = ["--source", "decoder", "--checkpoint", os.path.join(run_dir, "best.pt"), "--corner", a.corner, "--split", "val",
            "--out-dir", run_dir, "--tag", "eval_val", "--workers", str(max(a.eval_workers, 1))]
    if a.label_stage != "synth":
        argv += ["--label-stage", a.label_stage]
    if a.split_file:
        print("warning: eval_encoder reads the default split file, not --split-file", flush=True)
    if a.val_only:
        argv += ["--val-only", *a.val_only]
    if a.eval_limit:
        argv += ["--limit", str(a.eval_limit)]
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
          f"{e['right_path_endpoint_given_pct']:.2f} %, right choice per step {e['right_choice_per_step_pct']:.2f} %", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
