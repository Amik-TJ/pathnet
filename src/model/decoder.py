from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cell_model.model import mlp, squash
from graph.vocab import RF_PAIRS
from walk.oracle import OracleScorer

from .attn_message import SetBlock
from .encoder import instance_level_offset

NEG_INF = float("-inf")
N_FEATS = 4
N_EP_FEATS = 4
N_UP_FEATS = 4
DECODER_DEFAULTS = {"scorer": "mlp",
                    "scorer_width": 128,
                    "scorer_depth": 2,
                    "scorer_heads": 4,
                    "temp": 1.0,
                    "ep_temp": 1.0,
                    "score_chunk": 200_000,
                    "decoder_kind": "pointer",
                    "lookahead_hops": 0,
                    "seq_layers": 2,
                    "seq_width": 128,
                    "seq_heads": 4,
                    "seq_window": 8,
                    "seq_dropout": 0.0,
                    "seq_head": "mlp",
                    "seq_tf_mix": 0.0,
                    "ep_rerank_k": 0,
                    "ep_rerank_layers": 2,
                    "ep_rerank_width": 128,
                    "ep_referee": "none",
                    "ep_referee_rank": 0,
                    "beam": 1}
DECODER_KINDS = ("pointer", "seq")


@dataclass
class Candidates:
    seg: torch.Tensor
    n_seg: int
    q: torch.Tensor
    m: torch.Tensor
    feats: torch.Tensor
    up_feats: torch.Tensor | None = None
    up_z: torch.Tensor | None = None

    def inputs(self, ln_q, ln_m) -> torch.Tensor:
        x = [ln_q(self.q[self.seg]), ln_m(self.m), self.feats]
        if self.up_feats is not None:
            x += [self.up_feats, self.up_z]
        return torch.cat(x, 1)


def _zero_last(layer: nn.Linear) -> nn.Linear:
    with torch.no_grad():
        layer.weight.zero_(); layer.bias.zero_()
    return layer


def lookahead_width(dz: int, hops: int) -> int:
    return hops * (N_UP_FEATS + dz)


class MLPScorer(nn.Module):

    def __init__(self, dz: int, width: int = 128, depth: int = 2, heads: int = 4, extra: int = 0):
        super().__init__()
        self.ln_q, self.ln_m = nn.LayerNorm(dz), nn.LayerNorm(dz)
        self.net = mlp(2 * dz + N_FEATS + extra, 1, width, depth)
        _zero_last(self.net[-1])

    def forward(self, c: Candidates) -> torch.Tensor:
        return self.net(c.inputs(self.ln_q, self.ln_m))[:, 0]


class TransformerScorer(nn.Module):

    def __init__(self, dz: int, width: int = 128, depth: int = 2, heads: int = 4, extra: int = 0):
        super().__init__()
        if width % heads:
            raise ValueError(f"scorer_width {width} is not a multiple of scorer_heads {heads}")
        self.ln_q, self.ln_m = nn.LayerNorm(dz), nn.LayerNorm(dz)
        self.inp = nn.Linear(2 * dz + N_FEATS + extra, width)
        self.blocks = nn.ModuleList([SetBlock(width, heads) for _ in range(depth)])
        self.out = _zero_last(nn.Linear(width, 1))

    def forward(self, c: Candidates) -> torch.Tensor:
        x = self.inp(c.inputs(self.ln_q, self.ln_m))
        for blk in self.blocks:
            x, _, _ = blk(x, c.seg, c.n_seg)
        return self.out(x)[:, 0]


SCORERS = {"mlp": MLPScorer, "transformer": TransformerScorer}


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    m = torch.full((n_seg,), NEG_INF, dtype=x.dtype, device=x.device).scatter_reduce(0, seg, x.detach(), "amax", include_self=True)
    fin = torch.isfinite(m)
    m0 = torch.where(fin, m, torch.zeros_like(m))
    s = torch.zeros(n_seg, dtype=x.dtype, device=x.device).index_add_(0, seg, torch.exp(x - m0[seg]))
    return torch.where(fin, torch.log(s.clamp_min(1e-30)) + m0, torch.full_like(m, NEG_INF))


def segment_best(logit: torch.Tensor, seg: torch.Tensor, n_seg: int, u: torch.Tensor) -> torch.Tensor:
    n = len(logit)
    mx = torch.full((n_seg,), NEG_INF, dtype=logit.dtype, device=logit.device).scatter_reduce(0, seg, logit, "amax", include_self=True)
    is_max = torch.isfinite(logit) & (logit == mx[seg])
    big = torch.iinfo(torch.int64).max
    key = torch.where(is_max, u * n + torch.arange(n, device=logit.device), torch.full_like(u, big))
    kmin = torch.full((n_seg,), big, dtype=torch.long, device=logit.device).scatter_reduce(0, seg, key, "amin", include_self=True)
    return torch.where(kmin < big, kmin % max(n, 1), torch.full_like(kmin, -1))


def _chunks(seg: torch.Tensor, size: int) -> list:
    n = len(seg)
    if n <= size:
        return [(0, n)]
    starts = (torch.nonzero(seg[1:] != seg[:-1], as_tuple=True)[0] + 1).tolist()
    out, lo, j = [], 0, 0
    while n - lo > size:
        while j < len(starts) and starts[j] <= lo + size:
            j += 1
        hi = starts[j - 1] if j > 0 and starts[j - 1] > lo else (starts[j] if j < len(starts) else n)
        out.append((lo, hi))
        lo = hi
    if lo < n:
        out.append((lo, n))
    return out


class PathDecoder(nn.Module):

    def __init__(self, dz: int, config: dict | None = None):
        super().__init__()
        c = dict(DECODER_DEFAULTS, **(config or {}))
        if c["scorer"] not in SCORERS:
            raise ValueError(f"scorer {c['scorer']!r} not in {tuple(SCORERS)}")
        if c["decoder_kind"] not in DECODER_KINDS:
            raise ValueError(f"decoder_kind {c['decoder_kind']!r} not in {DECODER_KINDS}")
        if dz <= 0:
            raise ValueError("the decoder needs an encoder with a latent state (not the exact message)")
        if int(c["lookahead_hops"]) not in (0, 1, 2):
            raise ValueError("lookahead_hops is 0, 1 or 2")
        self.config, self.dz, self.chunk = c, dz, int(c["score_chunk"])
        self.kind, self.hops = c["decoder_kind"], int(c["lookahead_hops"])
        if self.hops:
            raise NotImplementedError("lookahead_hops > 0 is not supported")
        extra = lookahead_width(dz, self.hops)
        if self.kind == "pointer":
            self.walk = SCORERS[c["scorer"]](dz, width=c["scorer_width"], depth=c["scorer_depth"], heads=c["scorer_heads"], extra=extra)
        else:
            from .seq_decoder import SeqPointer
            self.walk = SeqPointer(dz, c, extra)
        self.log_t = nn.Parameter(torch.tensor(math.log(c["temp"])))
        self.ln_e = nn.LayerNorm(dz)
        self.ep = mlp(dz + N_EP_FEATS, 1, c["scorer_width"], 2)
        _zero_last(self.ep[-1])
        self.log_t_ep = nn.Parameter(torch.tensor(math.log(c["ep_temp"])))
        self.rerank_k = int(c["ep_rerank_k"])
        self.bank = None
        if self.rerank_k > 0:
            from .seq_decoder import EndpointReranker
            self.ep_rr = EndpointReranker(dz, c)
        elif c["ep_referee"] != "none" or int(c["ep_referee_rank"]):
            raise ValueError("ep_referee / ep_referee_rank need ep_rerank_k > 0")

    def temps(self) -> tuple:
        return torch.exp(self.log_t).clamp_min(1e-3), torch.exp(self.log_t_ep).clamp_min(1e-3)

    def forward(self, out: dict, b, only_seg: torch.Tensor | None = None) -> dict:
        if out.get("z") is None or out.get("msg_z") is None or "msg_arrival" not in out:
            raise ValueError("the decoder needs the encoder's z and its kept messages: call the encoder with keep=True")
        sch = out["schedule"]
        k = instance_level_offset(sch)
        a = out["msg_arrival"]
        n, G = len(a), len(sch.seg_ft)
        seg = sch.inst_seg[k:k + n]
        valid = torch.isfinite(a)
        best = torch.full((G,), NEG_INF, dtype=a.dtype, device=a.device).scatter_reduce(0, seg, a, "amax", include_self=True)
        a_rel = torch.where(valid, a - best[seg], torch.zeros_like(a)).float()
        t, t_ep = self.temps()
        logit = torch.where(valid, a_rel / t, torch.full_like(a_rel, NEG_INF))
        idx = torch.nonzero(valid if only_seg is None else valid & only_seg[seg], as_tuple=True)[0]
        if len(idx) and self.kind == "pointer":
            q = out["z"].reshape(-1, self.dz)[sch.seg_ft]
            delay, slew = out["msg_delay"].float(), out["msg_slew"].float()
            parts = []
            for lo, hi in _chunks(seg[idx], self.chunk):
                i = idx[lo:hi]
                feats = torch.stack([squash(a_rel[i]), squash(a_rel[i] / 10.0), squash(delay[i] / 10.0), squash(slew[i] / 10.0)], 1)
                parts.append(self.walk(Candidates(seg=seg[i], n_seg=G, q=q, m=out["msg_z"][i], feats=feats)))
            logit = logit.index_add(0, idx, torch.cat(parts))
        return {"logit": logit, "ep_score": self.endpoint_scores(out, b, t_ep)}

    @torch.no_grad()
    def decode(self, out: dict, b, logit: torch.Tensor, start_pin: torch.Tensor, start_rf: torch.Tensor, lab_tok=None, lab_start=None,
               lab_len=None, keep_tokens: bool = False) -> dict:
        if self.kind == "pointer":
            wi = WalkIndex(out, b)
            return decode_paths(wi, wi.best(logit), start_pin, start_rf, lab_tok, lab_start, lab_len, keep_tokens)
        t, _ = self.temps()
        return self.walk.readout(out, b, t, start_pin, start_rf, lab_tok, lab_start, lab_len, keep_tokens)

    def rerank(self, out: dict, b, d: dict, cand_idx: torch.Tensor, cand_graph: torch.Tensor, n_graph: int, priority: torch.Tensor | None = None) -> dict:
        return self.ep_rr(self, out, b, d, cand_idx, cand_graph, n_graph, priority)

    def endpoint_scores(self, out: dict, b, t_ep: torch.Tensor) -> torch.Tensor:
        s = out["ep_slack"]
        score = torch.full((b.num_nodes,), float("nan"), dtype=torch.float32, device=s.device)
        idx = torch.nonzero(torch.isfinite(s), as_tuple=True)[0]
        if len(idx) == 0:
            return score
        sv, g = s[idx].float(), b.node_graph[idx]
        mn = torch.full((len(b.names),), float("inf"), device=s.device).scatter_reduce(0, g, sv, "amin", include_self=True)
        rel = sv - mn[g]
        rf = out["ep_rf"][idx]
        z = out["z"].reshape(-1, self.dz)[2 * idx + rf]
        feats = torch.stack([squash(rel), squash(rel / 10.0), squash(out["slew"][idx, rf].float() / 10.0),
                             (b.x_cat[idx, 0] == 1).float()], 1)
        val = -rel / t_ep + self.ep(torch.cat([self.ln_e(z), feats], 1))[:, 0]
        return score.index_put((idx,), val)


def label_positives(out: dict, lab, num_nodes: int) -> dict:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    n, G = len(out["msg_arrival"]), len(sch.seg_ft)
    seg, fu, ft, u = sch.inst_seg[k:k + n], sch.inst_fu[k:k + n], sch.inst_ft[k:k + n], sch.inst_u[k:k + n]
    M = 2 * num_nodes
    exact = torch.isin(ft * M + fu, lab.dec_ft * M + lab.dec_fu)
    pin_pos = torch.isin(ft * num_nodes + u, lab.dec_ft * num_nodes + lab.dec_fu // 2)
    count = lambda m: torch.zeros(G, dtype=torch.long, device=seg.device).index_add_(0, seg, m.long())
    has_exact, lab_seg = count(exact) > 0, count(pin_pos) > 0
    n_dec = int(len(torch.unique(lab.dec_ft)))
    seg_of_flat = torch.full((M,), -1, dtype=torch.long, device=seg.device)
    seg_of_flat[sch.seg_ft] = torch.arange(G, device=seg.device)
    g = seg_of_flat[lab.dec_ft]
    w = torch.zeros(G, dtype=torch.float32, device=seg.device).index_add_(0, g[g >= 0], lab.dec_w[g >= 0].float())
    return {"pos": torch.where(has_exact[seg], exact, pin_pos), "pin_pos": pin_pos, "lab_seg": lab_seg, "w": w,
            "decisions": n_dec, "fallback": int((lab_seg & ~has_exact).sum()), "unmatched": n_dec - int(lab_seg.sum())}


def decoder_losses(dec: dict, out: dict, b, lab, pos: dict | None = None, ep_priority: torch.Tensor | None = None) -> tuple:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    logit = dec["logit"]
    n, G = len(logit), len(sch.seg_ft)
    seg, u = sch.inst_seg[k:k + n], sch.inst_u[k:k + n]
    pos = pos or label_positives(out, lab, b.num_nodes)
    fin = torch.isfinite(logit)
    lse_all = segment_logsumexp(logit, seg, G)
    lse_pos = segment_logsumexp(logit.masked_fill(~pos["pos"], NEG_INF), seg, G)
    n_fin = torch.zeros(G, dtype=torch.long, device=seg.device).index_add_(0, seg, fin.long())
    sel = torch.isfinite(lse_pos) & (n_fin >= 2)
    ce, cw = (lse_all - lse_pos)[sel], pos["w"][sel]
    walk_loss = (ce * cw).sum() / cw.sum() if len(ce) else logit.new_zeros(())
    with torch.no_grad():
        best = segment_best(logit, seg, G, u)
        pins = torch.unique(seg[fin] * b.num_nodes + u[fin]) // b.num_nodes
        n_pin = torch.zeros(G, dtype=torch.long, device=seg.device).index_add_(0, pins, torch.ones_like(pins))
        scored = pos["lab_seg"] & (n_pin >= 2) & (best >= 0)
        right = pos["pin_pos"][best.clamp_min(0)] & scored
    ep_loss, ep_stats = endpoint_losses(dec["ep_score"], b, lab, ep_priority)
    stats = {"walk_sum": float((ce * cw).sum()), "walk_n": float(cw.sum()), **ep_stats,
             "dec_right": float(pos["w"][right].sum()), "dec_n": float(pos["w"][scored].sum()),
             "decisions": pos["decisions"], "fallback": pos["fallback"], "unmatched": pos["unmatched"]}
    return walk_loss, ep_loss, stats


def endpoint_losses(ep_score: torch.Tensor, b, lab, ep_priority: torch.Tensor | None = None) -> tuple:
    B = len(b.names)
    s = ep_score[b.ep_index]
    ok = torch.isfinite(s)
    s = s.masked_fill(~ok, NEG_INF)
    e_all = segment_logsumexp(s, b.ep_graph, B)
    e_pos = segment_logsumexp(s.masked_fill(~lab.ep_worst, NEG_INF), b.ep_graph, B)
    cnt = lambda m: torch.zeros(B, dtype=torch.long, device=s.device).index_add_(0, b.ep_graph, m.long())
    esel = torch.isfinite(e_pos) & (cnt(ok) > cnt(ok & lab.ep_worst))
    ece = (e_all - e_pos)[esel]
    ep_loss = ece.mean() if len(ece) else s.new_zeros(())
    with torch.no_grad():
        pick = endpoint_pick(s, b.ep_graph, B, ep_priority)
        ep_right = lab.ep_worst[pick.clamp_min(0)] & (pick >= 0)
    return ep_loss, {"ep_sum": float(ece.sum()), "ep_n": int(len(ece)), "ep_right": int(ep_right.sum()), "ep_designs": int((pick >= 0).sum())}


def endpoint_pick(score: torch.Tensor, graph: torch.Tensor, n_graph: int, priority: torch.Tensor | None = None) -> torch.Tensor:
    s = torch.where(torch.isfinite(score), score, torch.full_like(score, NEG_INF))
    return segment_best(s, graph, n_graph, torch.zeros_like(graph) if priority is None else priority)


class WalkIndex:

    def __init__(self, out: dict, b):
        sch = out["schedule"]
        k = instance_level_offset(sch)
        n, N, dev = len(out["msg_arrival"]), b.num_nodes, b.level.device
        self.role = b.x_cat[:, 0]
        self.driver = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.driver[b.net[1]] = b.net[0]
        self.seg, self.u, self.frf = sch.inst_seg[k:k + n], sch.inst_u[k:k + n], sch.inst_frf[k:k + n]
        self.n_seg = len(sch.seg_ft)
        self.seg_of_flat = torch.full((2 * N,), -1, dtype=torch.long, device=dev)
        self.seg_of_flat[sch.seg_ft] = torch.arange(self.n_seg, device=dev)
        self.levels = b.levels

    def best(self, logit: torch.Tensor) -> torch.Tensor:
        return segment_best(logit, self.seg, self.n_seg, self.u)


def walk_step(wi: WalkIndex, tok: torch.Tensor, rf: torch.Tensor, alive: torch.Tensor) -> dict:
    minus = torch.full_like(tok, -1)
    r = wi.role[tok]
    is_load = (r == 2) | (r == 5) | (r == 1)
    d = torch.where(is_load, wi.driver[tok], torch.where(r == 3, tok, minus))
    has = alive & (d >= 0)
    dc = d.clamp_min(0)
    rd = wi.role[dc]
    at_start = has & is_load & ((rd == 6) | (rd == 0))
    go = has & ~at_start
    g = torch.where(go, wi.seg_of_flat[2 * dc + rf], minus)
    return {"dc": dc, "at_start": at_start, "go": go, "g": g}


class WalkTrace:

    def __init__(self, start_pin: torch.Tensor, start_rf: torch.Tensor, lab_tok, lab_start, lab_len, keep_tokens: bool):
        P, dev = len(start_pin), start_pin.device
        self.P, self.dev = P, dev
        self.tok, self.rf = start_pin.clone(), start_rf.clone()
        self.alive = torch.ones(P, dtype=torch.bool, device=dev)
        self.complete = torch.zeros(P, dtype=torch.bool, device=dev)
        self.n_tok = torch.ones(P, dtype=torch.long, device=dev)
        self.lab_tok, self.lab_start, self.lab_len, self.keep = lab_tok, lab_start, lab_len, keep_tokens
        self.check = lab_tok is not None and P > 0
        if self.check:
            self.last = max(len(lab_tok) - 1, 0)
            self.ok = (lab_len >= 1) & (lab_tok[lab_start.clamp(0, self.last)] == self.tok)
        self.cols, self.rf_cols, self.msg_cols = [self.tok.clone()], [self.rf.clone()], []

    def emit(self, at_start: torch.Tensor, dc: torch.Tensor, found: torch.Tensor, nxt_u: torch.Tensor, nxt_rf: torch.Tensor, msg: torch.Tensor):
        minus = torch.full_like(self.tok, -1)
        nxt = torch.where(at_start, dc, torch.where(found, nxt_u, minus))
        emit = at_start | found
        if self.check:
            lt = self.lab_tok[(self.lab_start + self.n_tok).clamp(0, self.last)]
            self.ok = self.ok & (~emit | ((self.n_tok < self.lab_len) & (lt == nxt)))
        if self.keep:
            self.cols.append(nxt)
            self.rf_cols.append(torch.where(found, nxt_rf, self.rf))
            self.msg_cols.append(torch.where(found, msg, minus))
        self.n_tok = self.n_tok + emit.long()
        self.complete = self.complete | at_start
        self.rf = torch.where(found, nxt_rf, self.rf)
        self.tok = torch.where(found, nxt, self.tok)
        self.alive = found

    def result(self) -> dict:
        res = {"complete": self.complete, "n_tok": self.n_tok}
        if self.check:
            res["right"] = self.ok & self.complete & (self.n_tok == self.lab_len)
        elif self.lab_tok is not None:
            res["right"] = torch.zeros(0, dtype=torch.bool, device=self.dev)
        if self.keep:
            res["tokens"] = torch.stack(self.cols, 1)
            res["token_rf"] = torch.stack(self.rf_cols, 1)
            res["msg"] = torch.stack(self.msg_cols, 1) if self.msg_cols else torch.zeros((self.P, 0), dtype=torch.long, device=self.dev)
        return res


@torch.no_grad()
def decode_paths(wi: WalkIndex, best: torch.Tensor, start_pin: torch.Tensor, start_rf: torch.Tensor,
                 lab_tok: torch.Tensor | None = None, lab_start: torch.Tensor | None = None,
                 lab_len: torch.Tensor | None = None, keep_tokens: bool = False) -> dict:
    tr = WalkTrace(start_pin, start_rf, lab_tok, lab_start, lab_len, keep_tokens)
    minus = torch.full((tr.P,), -1, dtype=torch.long, device=tr.dev)
    for _ in range(wi.levels + 2):
        if not bool(tr.alive.any()):
            break
        s = walk_step(wi, tr.tok, tr.rf, tr.alive)
        g = s["g"]
        i = torch.where(g >= 0, best[g.clamp_min(0)], minus)
        found = s["go"] & (i >= 0)
        ic = i.clamp_min(0)
        tr.emit(s["at_start"], s["dc"], found, wi.u[ic], wi.frf[ic], i)
    return tr.result()


@torch.no_grad()
def referee_arrival(out: dict, b, bank, r: dict) -> torch.Tensor:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    msg, dev = r["msg"], r["tokens"].device
    S = msg.shape[0]
    total = torch.zeros(S, dtype=torch.float64, device=dev)
    if msg.numel():
        s_idx, _ = torch.nonzero(msg >= 0, as_tuple=True)
        j = msg[msg >= 0] + k
        e, trf, fu, o, launch = sch.inst_e[j], sch.inst_trf[j], sch.inst_fu[j], sch.inst_o[j], sch.inst_launch[j]
        slew = out["slew"].reshape(-1).double()
        s_in = torch.where(launch, torch.zeros_like(slew[fu]), slew[fu])
        load = b.x_num.double()[:, 2:4][o, trf]
        total.index_add_(0, s_idx, bank.targets(b.arc_key[e], trf, s_in, load)[:, 0].double())
    last = (r["n_tok"] - 1).clamp_min(0)
    start_tok = r["tokens"].gather(1, last[:, None])[:, 0].clamp_min(0)
    start_rf = r["token_rf"].gather(1, last[:, None])[:, 0]
    a0 = out["arr"].reshape(-1).double()[2 * start_tok + start_rf]
    arrival = a0 + total
    return torch.where(r["complete"], arrival, torch.full_like(arrival, float("nan"))).float()


def walk_logit_table(out: dict, logit: torch.Tensor, num_arcs: int) -> torch.Tensor:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    t = torch.full((num_arcs, 4), float("nan"), dtype=logit.dtype, device=logit.device)
    t[sch.inst_e[k:k + len(logit)], sch.inst_pair[k:k + len(logit)]] = logit.detach()
    return t


class LogitScorer(OracleScorer):

    def __init__(self, g, res: dict, vocab):
        super().__init__(g, res, vocab)
        self.logit = res["walk_logit"]

    def score(self, u: int, o: int, rf: int, from_rf: int | None = None):
        best = (-np.inf, -1, -1)
        for e in self.g.arcs_of_pair.get((u, o), []):
            if not self.arc_on[e]:
                continue
            for pair_idx, (frf, trf) in enumerate(RF_PAIRS):
                if trf != rf or not self.arc_mask[e, pair_idx] or (from_rf is not None and frf != from_rf):
                    continue
                v = self.logit[e, pair_idx]
                if np.isfinite(v) and v > best[0]:
                    best = (float(v), frf, e)
        return best
