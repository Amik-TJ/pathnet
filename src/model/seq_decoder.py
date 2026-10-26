from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cell_model.model import mlp, squash

from .decoder import N_EP_FEATS, N_FEATS, NEG_INF, WalkIndex, WalkTrace, _zero_last, referee_arrival, segment_best, segment_logsumexp, walk_step
from .encoder import instance_level_offset

POS_MAX = 64
N_ROLES = 7
MAX_PT2 = 400_000_000


class Block(nn.Module):

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if d % heads:
            raise ValueError(f"seq_width {d} is not a multiple of seq_heads {heads}")
        self.d, self.h, self.dh = d, heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p_drop = dropout

    def split(self, x: torch.Tensor) -> tuple:
        P, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(P, T, 3, self.h, self.dh).unbind(2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_drop if self.training else 0.0)
        return a.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.split(x)
        x = x + self.drop(self.proj(self.attend(q, k, v, mask)))
        return x + self.drop(self.ff(self.ln2(x)))


def causal_window_mask(T: int, window: int, device) -> torch.Tensor:
    i = torch.arange(T, device=device)
    m = i[None, :] <= i[:, None]
    if window > 0:
        m &= (i[:, None] - i[None, :]) < window
    return m


class TokenEmbedding(nn.Module):

    def __init__(self, dz: int, d: int, n_pos: int):
        super().__init__()
        self.dz, self.n_pos = dz, n_pos
        self.ln_tok = nn.LayerNorm(dz)
        self.embed = nn.Linear(dz + N_ROLES + 5, d)
        self.pos = nn.Embedding(n_pos + 1, d)

    def forward(self, out: dict, b, tok: torch.Tensor, rf: torch.Tensor, pos: torch.Tensor, delay_in: torch.Tensor) -> torch.Tensor:
        t = tok.clamp_min(0)
        flat = 2 * t + rf
        z = out["z"].reshape(-1, self.dz)[flat]
        role = b.x_cat[:, 0][t]
        arr, slew = out["arr"].reshape(-1)[flat].float(), out["slew"].reshape(-1)[flat].float()
        arr = torch.where(torch.isfinite(arr), arr, torch.zeros_like(arr))
        x = torch.cat([self.ln_tok(z), F.one_hot(role, N_ROLES).float(), squash(arr / 10.0)[..., None], squash(slew / 10.0)[..., None],
                       squash(delay_in.float() / 10.0)[..., None], (pos == 0).float()[..., None], ((role == 6) | (role == 0)).float()[..., None]], -1)
        return self.embed(x) + self.pos(pos.clamp(0, self.n_pos))


def expand_csr(start: torch.Tensor, count: torch.Tensor) -> tuple:
    M = int(count.sum())
    row = torch.repeat_interleave(torch.arange(len(count), device=count.device), count)
    first = torch.cumsum(count, 0) - count
    off = torch.arange(M, device=count.device) - first[row]
    return row, start[row] + off


class SeqPointer(nn.Module):
    def __init__(self, dz: int, config: dict, extra: int = 0):
        super().__init__()
        c = config
        d, H, L = int(c["seq_width"]), int(c["seq_heads"]), int(c["seq_layers"])
        self.dz, self.d, self.window, self.extra = dz, d, int(c["seq_window"]), extra
        self.tf_mix, self.beam = float(c["seq_tf_mix"]), int(c["beam"])
        if self.window < 0 or L < 1 or self.tf_mix < 0 or self.tf_mix > 1:
            raise ValueError("seq_window >= 0, seq_layers >= 1, 0 <= seq_tf_mix <= 1")
        if self.beam != 1:
            raise NotImplementedError("beam > 1 is a knob kept for later; only the greedy read-out exists")
        self.n_pos = self.window if self.window > 0 else POS_MAX
        self.tok = TokenEmbedding(dz, d, self.n_pos)
        self.blocks = nn.ModuleList([Block(d, H, float(c["seq_dropout"])) for _ in range(L)])
        self.ln_out = nn.LayerNorm(d)
        self.ln_q, self.ln_m = nn.LayerNorm(dz), nn.LayerNorm(dz)
        self.q = nn.Linear(d + dz, d)
        self.head_kind = c["seq_head"]
        if self.head_kind == "mlp":
            self.head = mlp(d + dz + N_FEATS + extra, 1, d, 2)
            _zero_last(self.head[-1])
        elif self.head_kind == "bilinear":
            self.key = nn.Linear(dz + N_FEATS + extra, d)
            self.bil = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.bil.weight.zero_()
        else:
            raise ValueError(f"seq_head {self.head_kind!r} is mlp or bilinear")

    @staticmethod
    def terms(out: dict) -> dict:
        sch = out["schedule"]
        k = instance_level_offset(sch)
        a = out["msg_arrival"]
        n, G = len(a), len(sch.seg_ft)
        seg = sch.inst_seg[k:k + n]
        valid = torch.isfinite(a)
        best = torch.full((G,), NEG_INF, dtype=a.dtype, device=a.device).scatter_reduce(0, seg, a, "amax", include_self=True)
        a_rel = torch.where(valid, a - best[seg], torch.zeros_like(a)).float()
        delay, slew = out["msg_delay"].float(), out["msg_slew"].float()
        feats = torch.stack([squash(a_rel), squash(a_rel / 10.0), squash(delay / 10.0), squash(slew / 10.0)], 1)
        count = torch.bincount(seg, minlength=G)
        start = torch.cumsum(count, 0) - count
        return {"seg": seg, "u": sch.inst_u[k:k + n], "frf": sch.inst_frf[k:k + n], "e": sch.inst_e[k:k + n], "a_rel": a_rel, "valid": valid,
                "feats": feats, "m": out["msg_z"], "delay": delay, "seg_start": start, "seg_count": count, "G": G, "n": n, "k": k, "z": out["z"]}

    def token_embedding(self, out: dict, b, tok: torch.Tensor, rf: torch.Tensor, pos: torch.Tensor, delay_in: torch.Tensor) -> torch.Tensor:
        return self.tok(out, b, tok, rf, pos, delay_in)

    def logits(self, h: torch.Tensor, o: torch.Tensor, rf: torch.Tensor, pr: dict, tm: dict, t: torch.Tensor, up: tuple | None) -> torch.Tensor:
        q = self.q(torch.cat([h, self.ln_q(tm["z"].reshape(-1, self.dz)[2 * o + rf])], 1))
        c, s = pr["cand"], pr["step"]
        parts = [self.ln_m(tm["m"][c]), tm["feats"][c]]
        if up is not None:
            parts += [up[0][c], up[1][c]]
        kin = torch.cat(parts, 1)
        if self.head_kind == "mlp":
            corr = self.head(torch.cat([q[s], kin], 1))[:, 0]
        else:
            corr = (self.bil(q)[s] * self.key(kin)).sum(-1)
        logit = tm["a_rel"][c] / t + corr
        return torch.where(pr["ok"], logit, torch.full_like(logit, NEG_INF))

    @staticmethod
    def pairs(tm: dict, g: torch.Tensor, go: torch.Tensor) -> dict:
        count = torch.where(go, tm["seg_count"][g.clamp_min(0)], torch.zeros_like(g))
        step, cand = expand_csr(tm["seg_start"][g.clamp_min(0)], count)
        ok = tm["valid"][cand]
        n_fin = torch.zeros(len(g), dtype=torch.long, device=g.device).index_add_(0, step, ok.long())
        return {"step": step, "cand": cand, "ok": ok, "n_fin": n_fin, "S": len(g)}

    @staticmethod
    def pick(logit: torch.Tensor, pr: dict, tm: dict) -> tuple:
        idx = segment_best(logit, pr["step"], pr["S"], tm["u"][pr["cand"]])
        found = idx >= 0
        if len(pr["cand"]) == 0:
            return torch.full_like(idx, -1), found, idx
        return torch.where(found, pr["cand"][idx.clamp_min(0)], torch.full_like(idx, -1)), found, idx

    @staticmethod
    def pin_count(pr: dict, tm: dict, N: int) -> torch.Tensor:
        ok = pr["ok"]
        key = torch.unique(pr["step"][ok] * N + tm["u"][pr["cand"][ok]]) // N
        return torch.zeros(pr["S"], dtype=torch.long, device=key.device).index_add_(0, key, torch.ones_like(key))

    @staticmethod
    def paths(lab) -> dict:
        length = lab.tok_ptr[1:] - lab.tok_ptr[:-1]
        P, T = len(length), int(length.max()) if len(length) else 0
        ar = torch.arange(T, device=lab.tok.device)
        ok = ar[None, :] < length[:, None]
        idx = (lab.tok_ptr[:-1][:, None] + ar[None, :]).clamp(0, max(len(lab.tok) - 1, 0))
        tok = torch.where(ok, lab.tok[idx], torch.full_like(idx, -1))
        rf = torch.where(ok, lab.tok_rf[idx], torch.zeros_like(idx))
        return {"tok": tok, "rf": rf, "length": length, "P": P, "T": T, "ok": ok}

    def forced_steps(self, wi: WalkIndex, tm: dict, pt: dict) -> dict:
        tok, rf, ok = pt["tok"], pt["rf"], pt["ok"]
        P, T = pt["P"], pt["T"]
        dev = tok.device
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        if T < 2:
            return {"path": empty, "pos": empty, "o": empty, "g": empty, "pr": {"step": empty, "cand": empty, "ok": empty.bool(), "n_fin": empty, "S": 0},
                    "target": empty.bool(), "pin_target": empty.bool(), "jstar": empty, "delay_in": torch.zeros((P, T), device=dev), "fallback": 0}
        cur_tok, cur_rf = tok[:, :-1].reshape(-1), rf[:, :-1].reshape(-1)
        alive = ok[:, 1:].reshape(-1)
        s = walk_step(wi, cur_tok, cur_rf, alive)
        is_step = s["go"] & (s["g"] >= 0)
        rows = torch.nonzero(is_step, as_tuple=True)[0]
        path, pos = rows // (T - 1), rows % (T - 1)
        g, o = s["g"][rows], s["dc"][rows]
        pr = self.pairs(tm, g, torch.ones_like(g, dtype=torch.bool))
        nxt_tok, nxt_rf = tok[path, pos + 1], rf[path, pos + 1]
        u, frf = tm["u"][pr["cand"]], tm["frf"][pr["cand"]]
        pin_t = pr["ok"] & (u == nxt_tok[pr["step"]])
        exact_t = pin_t & (frf == nxt_rf[pr["step"]])
        has_exact = torch.zeros(pr["S"], dtype=torch.bool, device=dev).index_put_((pr["step"][exact_t],), torch.ones_like(exact_t[exact_t]))
        target = torch.where(has_exact[pr["step"]], exact_t, pin_t)
        a_t = torch.where(target, tm["a_rel"][pr["cand"]], torch.full_like(tm["a_rel"][pr["cand"]], NEG_INF))
        jt = segment_best(a_t, pr["step"], pr["S"], u)
        jstar = torch.where(jt >= 0, pr["cand"][jt.clamp_min(0)], torch.full_like(jt, -1))
        delay_in = torch.zeros((P, T), device=dev)
        delay_in[path, pos + 1] = torch.where(jstar >= 0, tm["delay"][jstar.clamp_min(0)], torch.zeros(len(jstar), device=dev))
        has_pin = torch.zeros(pr["S"], dtype=torch.bool, device=dev).index_put_((pr["step"][pin_t],), torch.ones_like(pin_t[pin_t]))
        return {"path": path, "pos": pos, "o": o, "g": g, "pr": pr, "target": target, "pin_target": pin_t, "jstar": jstar, "delay_in": delay_in,
                "fallback": int((has_pin & ~has_exact).sum())}

    def encode(self, out: dict, b, tok: torch.Tensor, rf: torch.Tensor, ok: torch.Tensor, delay_in: torch.Tensor) -> torch.Tensor:
        P, T = tok.shape
        pos = torch.arange(T, device=tok.device)[None, :].expand(P, T)
        x = self.token_embedding(out, b, tok, rf, pos, delay_in)
        mask = causal_window_mask(T, self.window, tok.device)[None, None] & ok[:, None, None, :]
        mask = mask | torch.eye(T, dtype=torch.bool, device=tok.device)[None, None]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_out(x)

    def chunked_encode(self, out, b, tok, rf, ok, delay_in) -> torch.Tensor:
        P, T = tok.shape
        per = max(1, MAX_PT2 // max(T * T, 1))
        if P <= per:
            return self.encode(out, b, tok, rf, ok, delay_in)
        return torch.cat([self.encode(out, b, tok[i:i + per], rf[i:i + per], ok[i:i + per], delay_in[i:i + per]) for i in range(0, P, per)], 0)

    def forward_seq(self, out: dict, b, lab, t: torch.Tensor, up: tuple | None = None, sample: torch.Generator | None = None) -> dict:
        wi = WalkIndex(out, b)
        tm = self.terms(out)
        pt = self.paths(lab)
        fs = self.forced_steps(wi, tm, pt)
        tok, rf, ok, delay_in = pt["tok"], pt["rf"], pt["ok"], fs["delay_in"]
        pr, path, pos = fs["pr"], fs["path"], fs["pos"]
        S = pr["S"]
        if S == 0:
            e = torch.zeros(0, device=tok.device)
            return {"logit": e, "pr": pr, "target": e.bool(), "pin_target": e.bool(), "seg": e.long(), "path": path, "fallback": 0, "tm": tm}
        if self.tf_mix > 0 and self.training:
            with torch.no_grad():
                h0 = self.chunked_encode(out, b, tok, rf, ok, delay_in)
                lg = self.logits(h0[path, pos], fs["o"], rf[path, pos], pr, tm, t, up)
                j, found, _ = self.pick(lg, pr, tm)
                swap = (torch.rand(S, device=tok.device, generator=sample) < self.tf_mix) & found
                jc = j.clamp_min(0)
                tok, rf, delay_in = tok.clone(), rf.clone(), delay_in.clone()
                tok[path, pos + 1] = torch.where(swap, tm["u"][jc], tok[path, pos + 1])
                rf[path, pos + 1] = torch.where(swap, tm["frf"][jc], rf[path, pos + 1])
                delay_in[path, pos + 1] = torch.where(swap, tm["delay"][jc], delay_in[path, pos + 1])
        h = self.chunked_encode(out, b, tok, rf, ok, delay_in)
        logit = self.logits(h[path, pos], fs["o"], pt["rf"][path, pos], pr, tm, t, up)
        return {"logit": logit, "pr": pr, "target": fs["target"], "pin_target": fs["pin_target"], "seg": fs["g"], "path": path,
                "fallback": fs["fallback"], "tm": tm}

    @torch.no_grad()
    def readout(self, out: dict, b, t: torch.Tensor, start_pin: torch.Tensor, start_rf: torch.Tensor, lab_tok=None, lab_start=None,
                lab_len=None, keep_tokens: bool = False, up: tuple | None = None) -> dict:
        wi = WalkIndex(out, b)
        tm = self.terms(out)
        tr = WalkTrace(start_pin, start_rf, lab_tok, lab_start, lab_len, keep_tokens)
        P, dev = tr.P, tr.dev
        if P == 0:
            return tr.result()
        steps = wi.levels + 2
        slots = self.window if self.window > 0 else steps + 1
        cache = [(torch.zeros((P, blk.h, slots, blk.dh), device=dev), torch.zeros((P, blk.h, slots, blk.dh), device=dev)) for blk in self.blocks]
        filled = torch.zeros(slots, dtype=torch.bool, device=dev)
        delay_in = torch.zeros(P, device=dev)
        for i in range(steps):
            if not bool(tr.alive.any()):
                break
            pos = torch.full((P,), i, dtype=torch.long, device=dev)
            x = self.token_embedding(out, b, tr.tok, tr.rf, pos, delay_in)[:, None, :]
            slot = i % slots
            filled[slot] = True
            for (K_, V_), blk in zip(cache, self.blocks):
                q, k, v = blk.split(x)
                K_[:, :, slot] = k[:, :, 0]; V_[:, :, slot] = v[:, :, 0]
                x = x + blk.proj(blk.attend(q, K_, V_, filled[None, None, None, :]))
                x = x + blk.ff(blk.ln2(x))
            h = self.ln_out(x[:, 0])
            s = walk_step(wi, tr.tok, tr.rf, tr.alive)
            go = s["go"] & (s["g"] >= 0)
            pr = self.pairs(tm, s["g"], go)
            logit = self.logits(h, s["dc"], tr.rf, pr, tm, t, up)
            j, found, _ = self.pick(logit, pr, tm)
            found = found & go
            jc = j.clamp_min(0)
            tr.emit(s["at_start"], s["dc"], found, tm["u"][jc], tm["frf"][jc], torch.where(found, j, torch.full_like(j, -1)))
            delay_in = torch.where(found, tm["delay"][jc], torch.zeros_like(delay_in))
        return tr.result()


def seq_losses(dec, out: dict, b, lab, pos: dict, up: tuple | None = None, sample: torch.Generator | None = None) -> tuple:
    t, _ = dec.temps()
    res = dec.walk.forward_seq(out, b, lab, t, up, sample)
    logit, pr, target, pin_t = res["logit"], res["pr"], res["target"], res["pin_target"]
    S = pr["S"]
    G = len(out["schedule"].seg_ft)
    seg_wrong = torch.zeros(G, dtype=torch.bool, device=b.level.device)
    base = {"decisions": pos["decisions"], "fallback": pos["fallback"], "unmatched": pos["unmatched"]}
    if S == 0:
        return logit.new_zeros(()), {"walk_sum": 0.0, "walk_n": 0.0, "dec_right": 0.0, "dec_n": 0.0, **base}, {"seg_wrong": seg_wrong}
    lse_all = segment_logsumexp(logit, pr["step"], S)
    lse_pos = segment_logsumexp(logit.masked_fill(~target, NEG_INF), pr["step"], S)
    sel = torch.isfinite(lse_pos) & (pr["n_fin"] >= 2)
    ce = (lse_all - lse_pos)[sel]
    loss = ce.mean() if len(ce) else logit.new_zeros(())
    with torch.no_grad():
        tm = res["tm"]
        j, found, idx = SeqPointer.pick(logit, pr, tm)
        n_pin = SeqPointer.pin_count(pr, tm, b.num_nodes)
        scored = (n_pin >= 2) & found
        right = pin_t[idx.clamp_min(0)] & scored
        seg_wrong[res["seg"][scored & ~right]] = True
    stats = {"walk_sum": float(ce.sum()), "walk_n": float(len(ce)), "dec_right": float(right.sum()), "dec_n": float(scored.sum()), **base}
    return loss, stats, {"seg_wrong": seg_wrong}


class EndpointReranker(nn.Module):

    def __init__(self, dz: int, config: dict):
        super().__init__()
        c = config
        d, L, H = int(c["ep_rerank_width"]), int(c["ep_rerank_layers"]), int(c["seq_heads"])
        self.dz, self.k, self.referee, self.rank_only = dz, int(c["ep_rerank_k"]), c["ep_referee"], bool(int(c["ep_referee_rank"]))
        if self.k <= 0 or L < 1:
            raise ValueError("ep_rerank_k > 0 and ep_rerank_layers >= 1")
        if self.referee not in ("none", "path"):
            raise ValueError(f"ep_referee {self.referee!r} is none or path")
        if self.rank_only and self.referee != "path":
            raise ValueError("ep_referee_rank 1 needs ep_referee path")
        self.tok = TokenEmbedding(dz, d, POS_MAX)
        self.blocks = nn.ModuleList([Block(d, H, 0.0) for _ in range(L)])
        self.ln_out, self.ln_e = nn.LayerNorm(d), nn.LayerNorm(dz)
        self.head = mlp(d + dz + N_EP_FEATS + (2 if self.referee == "path" else 0), 1, d, 2)
        _zero_last(self.head[-1])

    @staticmethod
    def select(score: torch.Tensor, graph: torch.Tensor, n_graph: int, k: int, priority: torch.Tensor | None) -> torch.Tensor:
        n = len(score)
        idx = torch.nonzero(torch.isfinite(score), as_tuple=True)[0]
        pr = (torch.arange(n, device=score.device) if priority is None else priority)[idx]
        o = idx[torch.argsort(pr, stable=True)]
        o = o[torch.argsort(-score[o], stable=True)]
        o = o[torch.argsort(graph[o], stable=True)]
        g = graph[o]
        pos = torch.arange(len(o), device=score.device)
        first = torch.full((n_graph,), len(o), dtype=torch.long, device=score.device).scatter_reduce(0, g, pos, "amin", include_self=True)
        return o[pos - first[g] < k]

    def forward(self, dec, out: dict, b, d: dict, cand_idx: torch.Tensor, cand_graph: torch.Tensor, n_graph: int,
                priority: torch.Tensor | None = None) -> dict:
        score = d["ep_score"][cand_idx]
        sel = self.select(score, cand_graph, n_graph, self.k, priority)
        sg = cand_graph[sel]
        ep = cand_idx[sel]
        rf = out["ep_rf"][ep]
        r = dec.decode(out, b, d["logit"], ep, rf, keep_tokens=True)
        S, T = r["tokens"].shape
        dev = ep.device
        ok = torch.arange(T, device=dev)[None, :] < r["n_tok"][:, None]
        msg = r["msg"]
        delay_in = torch.zeros((S, T), device=dev)
        if msg.shape[1]:
            delay_in[:, 1:] = torch.where(msg >= 0, out["msg_delay"].float()[msg.clamp_min(0)], torch.zeros_like(delay_in[:, 1:]))
        pos = torch.arange(T, device=dev)[None, :].expand(S, T)
        x = self.tok(out, b, r["tokens"], r["token_rf"], pos, delay_in)
        mask = ok[:, None, None, :] | torch.eye(T, dtype=torch.bool, device=dev)[None, None]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_out(x)
        pooled = (x * ok[..., None]).sum(1) / ok.sum(1, keepdim=True).clamp_min(1)
        sl = out["ep_slack"][ep].float()
        mn = torch.full((n_graph,), float("inf"), device=dev).scatter_reduce(0, sg, sl, "amin", include_self=True)
        rel = sl - mn[sg]
        feats = [squash(rel), squash(rel / 10.0), squash(out["slew"][ep, rf].float() / 10.0), (b.x_cat[ep, 0] == 1).float()]
        slack_ref = None
        if self.referee == "path":
            if dec.bank is None:
                raise RuntimeError("ep_referee path needs the arc bank: set decoder.bank (train_decoder / eval_encoder do)")
            arrival = referee_arrival(out, b, dec.bank, r)
            slack_ref = out["req"].reshape(-1)[2 * ep + rf].float() - arrival
            ref = torch.where(torch.isfinite(slack_ref), slack_ref, sl)
            mn_ref = torch.full((n_graph,), float("inf"), device=dev).scatter_reduce(0, sg, ref, "amin", include_self=True)
            feats += [squash(ref - sl), squash(ref - mn_ref[sg])]
        z = out["z"].reshape(-1, self.dz)[2 * ep + rf]
        delta = self.head(torch.cat([pooled, self.ln_e(z), torch.stack(feats, 1)], 1))[:, 0]
        score2 = -ref if self.rank_only else score[sel] + delta
        key = (torch.arange(len(cand_idx), device=dev) if priority is None else priority)[sel]
        ps = segment_best(score2, sg, n_graph, key)
        pick = torch.where(ps >= 0, sel[ps.clamp_min(0)], torch.full_like(ps, -1))
        return {"sel": sel, "sel_graph": sg, "score2": score2, "delta": delta, "slack_ref": slack_ref, "pick": pick, "paths": r}


def rerank_losses(rr: dict, worst_sel: torch.Tensor, n_graph: int, learn: bool = True) -> tuple:
    s2, sg = rr["score2"], rr["sel_graph"]
    cnt = torch.zeros(n_graph, dtype=torch.long, device=s2.device).index_add_(0, sg, torch.ones_like(sg))
    has = torch.zeros(n_graph, dtype=torch.bool, device=s2.device).index_put_((sg[worst_sel],), torch.ones_like(worst_sel[worst_sel]))
    lse_all = segment_logsumexp(s2, sg, n_graph)
    lse_pos = segment_logsumexp(s2.masked_fill(~worst_sel, NEG_INF), sg, n_graph)
    sel = torch.isfinite(lse_pos) & (cnt >= 2)
    ce = (lse_all - lse_pos)[sel]
    loss = ce.mean() if (len(ce) and learn) else s2.new_zeros(())
    pick = rr["pick"]
    with torch.no_grad():
        right = int((worst_sel.new_zeros(len(pick), dtype=torch.bool).index_put_((torch.nonzero(pick >= 0, as_tuple=True)[0],),
                     worst_from_pick(worst_sel, rr["sel"], pick))).sum())
    stats = {"rr_sum": float(ce.sum()), "rr_n": float(len(ce)), "rr_right": right, "rr_designs": int((cnt > 0).sum()),
             "rr_skipped": int(((cnt > 0) & ~has).sum()), "rr_in_k": int(has.sum())}
    return loss, stats


def worst_from_pick(worst_sel: torch.Tensor, sel: torch.Tensor, pick: torch.Tensor) -> torch.Tensor:
    look = torch.full((int(sel.max()) + 1 if len(sel) else 1,), False, dtype=torch.bool, device=sel.device)
    look[sel] = worst_sel
    p = pick[pick >= 0]
    return look[p]
