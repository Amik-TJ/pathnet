from __future__ import annotations

import math

import torch
from torch import nn

from cell_model.model import mlp, squash

from .arc_embedding import ArcEmbedding
from .encoder import LearnedMessage

NEG_INF = float("-inf")


def segment_softmax(logit: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    H = logit.shape[1]
    m = torch.full((n_seg, H), NEG_INF, dtype=logit.dtype, device=logit.device)
    m = m.scatter_reduce(0, seg[:, None].expand(-1, H), logit.detach(), "amax", include_self=True)
    e = torch.exp(logit - m[seg])
    s = torch.zeros((n_seg, H), dtype=logit.dtype, device=logit.device).index_add_(0, seg, e)
    return e / s.clamp_min(1e-30)[seg]


def segment_pairs(seg: torch.Tensor, n_seg: int):
    n = len(seg)
    c = torch.bincount(seg, minlength=n_seg)
    start = torch.cumsum(c, 0) - c
    rep = c[seg]
    i = torch.repeat_interleave(torch.arange(n, device=seg.device), rep)
    csum = torch.cumsum(rep, 0)
    off = torch.arange(int(csum[-1]) if n else 0, device=seg.device) - torch.repeat_interleave(csum - rep, rep)
    j = start[seg[i]] + off
    return i, j


def entropy_stats(w: torch.Tensor, seg: torch.Tensor, n_seg: int):
    k = torch.bincount(seg, minlength=n_seg)
    ent = torch.zeros((n_seg, w.shape[1]), dtype=w.dtype, device=w.device)
    ent = ent.index_add_(0, seg, -w * torch.log(w.clamp_min(1e-30)))
    mx = torch.zeros_like(ent).scatter_reduce(0, seg[:, None].expand_as(w), w, "amax", include_self=True)
    multi = torch.nonzero(k >= 2, as_tuple=True)[0]
    return (ent[multi] / torch.log(k[multi].to(w.dtype))[:, None]).mean(1), mx[multi].mean(1), multi


class SegmentAttention(nn.Module):

    def __init__(self, d_q: int, d_kv: int, d_out: int, heads: int):
        super().__init__()
        if d_out % heads:
            raise ValueError(f"d_out {d_out} is not a multiple of heads {heads}")
        self.heads, self.dh = heads, d_out // heads
        self.q, self.k, self.v, self.o = nn.Linear(d_q, d_out), nn.Linear(d_kv, d_out), nn.Linear(d_kv, d_out), nn.Linear(d_out, d_out)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, seg: torch.Tensor, n_seg: int):
        H, dh = self.heads, self.dh
        q = self.q(q_in).view(n_seg, H, dh)
        k = self.k(kv_in).view(-1, H, dh)
        v = self.v(kv_in).view(-1, H, dh)
        logit = (q[seg] * k).sum(-1) / math.sqrt(dh)
        w = segment_softmax(logit, seg, n_seg)
        out = torch.zeros((n_seg, H, dh), dtype=v.dtype, device=v.device).index_add_(0, seg, w[:, :, None] * v)
        return self.o(out.view(n_seg, H * dh)), w


class SetBlock(nn.Module):

    def __init__(self, d: int, heads: int, ff: int | None = None):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff or 4 * d), nn.ReLU(), nn.Linear(ff or 4 * d, d))

    def forward(self, x: torch.Tensor, seg: torch.Tensor, n_seg: int):
        n, H, dh = x.shape[0], self.heads, self.dh
        i, j = segment_pairs(seg, n_seg)
        q, k, v = self.q(x).view(n, H, dh), self.k(x).view(n, H, dh), self.v(x).view(n, H, dh)
        logit = (q[i] * k[j]).sum(-1) / math.sqrt(dh)
        w = segment_softmax(logit, i, n)
        att = torch.zeros((n, H, dh), dtype=x.dtype, device=x.device).index_add_(0, i, w[:, :, None] * v[j])
        h = self.ln1(x + self.o(att.view(n, H * dh)))
        return self.ln2(h + self.ff(h)), w, (i, j)


class AttentionMessage(LearnedMessage):
    def __init__(self, arc: ArcEmbedding, n_pin_names: int, dz: int = 64, width: int = 128, depth: int = 2,
                 correction: bool = True, heads: int = 4, net_pool: bool = True, fanin_self_attn: bool = False,
                 soft_arrival: bool = False, soft_tau: float = 1.0):
        super().__init__(arc, n_pin_names, dz, width, depth, correction)
        self.heads, self.net_pool, self.fanin_self_attn, self.soft_arrival = heads, net_pool, fanin_self_attn, soft_arrival
        self.pool = SegmentAttention(dz, dz + 3, dz, heads)
        self.ln_pool = nn.LayerNorm(dz)
        if net_pool:
            self.netpool = SegmentAttention(dz, dz, dz, heads)
            self.ctx_proj = nn.Linear(dz, dz)
            self.msg = mlp(dz + dz + arc.d_out + 2 + dz, dz + 2, width, depth)
            if arc.mode == "tables":
                with torch.no_grad():
                    self.msg[-1].weight[dz:].zero_(); self.msg[-1].bias[dz:].zero_()
        if fanin_self_attn:
            self.sab = SetBlock(dz, heads)
        if soft_arrival:
            self.log_tau = nn.Parameter(torch.tensor(math.log(soft_tau)))
        self._ctx = None
        self.monitor = False
        self.record = None
        self._stats = {}

    def _acc(self, name: str, value: torch.Tensor, n: torch.Tensor | int):
        s, c = self._stats.get(name, (0.0, 0.0))
        self._stats[name] = (s + value.detach(), c + (n.detach() if torch.is_tensor(n) else n))

    def pop_stats(self) -> dict:
        out = {k: (float(s) / max(float(c), 1e-30) if float(c) else float("nan")) for k, (s, c) in self._stats.items()}
        if self.soft_arrival:
            out["soft_tau"] = float(torch.exp(self.log_tau))
        self._stats = {}
        return out

    def prepare(self, b, sch, z0) -> dict:
        st = super().prepare(b, sch, z0)
        if self.net_pool:
            N, dz = b.num_nodes, self.dz
            z0f = z0.reshape(-1, dz)
            d, s = b.net
            fd = torch.stack([2 * d, 2 * d + 1], 1).reshape(-1)
            fs = torch.stack([2 * s, 2 * s + 1], 1).reshape(-1)
            ctx, w = self.netpool(z0f, z0f[fs], fd, 2 * N)
            self._ctx = ctx
            st["ctx_o"] = ctx[sch.inst_ft]
            if self.monitor and len(fs):
                ent, mx, multi = entropy_stats(w, fd, 2 * N)
                self._acc("net_entropy", ent.sum(), len(multi)); self._acc("net_maxw", mx.sum(), len(multi))
        return st

    def cell(self, st, lo, hi, s_in, z_in):
        if not self.net_pool:
            return super().cell(st, lo, hi, s_in, z_in)
        d_hat, s_hat, e = self.arc.read(st, lo, hi, s_in)
        g = torch.stack([squash(s_in.float() / 10.0), st["gv"][lo:hi]], 1)
        o = self.msg(torch.cat([self.ln_in(z_in), st["z0_o"][lo:hi], e, g, st["ctx_o"][lo:hi]], 1))
        z = self.ln_out(o[:, :self.dz])
        if self.correction or self.arc.mode == "key":
            d_hat = d_hat + self.corr_scale * o[:, self.dz]
            s_hat = s_hat + self.corr_scale * o[:, self.dz + 1]
        return d_hat, s_hat, z

    def wire(self, z_driver, z0_sink, fd=None, fs=None, **kw):
        h = z_driver + self.wire_proj(z0_sink)
        if self.net_pool:
            h = h + self.ctx_proj(self._ctx[fd])
        return self.ln_wire(h)

    def pool_z(self, z_self, zm, loc, n_seg, cand=None, delay=None, slew=None, tgt=None, level=None):
        if self.fanin_self_attn:
            zm, w_sab, _ = self.sab(zm, loc, n_seg)
        best = torch.full((n_seg,), NEG_INF, dtype=cand.dtype, device=cand.device)
        best = best.scatter_reduce(0, loc, cand.detach(), "amax", include_self=True)
        a_rel = torch.where(torch.isfinite(cand) & torch.isfinite(best[loc]), cand - best[loc], torch.full_like(cand, -1e4))
        feats = torch.stack([squash(a_rel.float() / 10.0), squash(delay.float() / 10.0), squash(slew.float() / 10.0)], 1)
        out, w = self.pool(z_self, torch.cat([zm, feats], 1), loc, n_seg)
        if self.record is not None:
            self.record.append((tgt.cpu(), loc.cpu(), w.detach().cpu(), cand.detach().cpu(), level))
        if self.monitor:
            ent, mx, multi = entropy_stats(w, loc, n_seg)
            self._acc("fanin_entropy", ent.sum(), len(multi)); self._acc("fanin_maxw", mx.sum(), len(multi))
            wm = w.mean(1)
            top = torch.zeros(n_seg, dtype=wm.dtype, device=wm.device).scatter_reduce(0, loc, wm, "amax", include_self=True)
            is_top = wm == top[loc]
            is_win = cand == best[loc]
            n_win = torch.zeros(n_seg, dtype=torch.long, device=wm.device).index_add_(0, loc, is_win.long())
            n_fin = torch.zeros(n_seg, dtype=torch.long, device=wm.device).index_add_(0, loc, torch.isfinite(cand).long())
            n_top = torch.zeros(n_seg, dtype=torch.long, device=wm.device).index_add_(0, loc, is_top.long())
            ok = (n_win == 1) & (n_fin >= 2)
            agree = torch.zeros(n_seg, dtype=torch.long, device=wm.device).index_add_(0, loc, (is_top & is_win).long())
            self._acc("fanin_agree", ((agree > 0) & (n_top == 1))[ok].sum(), ok.sum())
            self._acc("fanin_weight_ties", (n_top > 1)[ok].sum(), ok.sum())
        return self.ln_pool(z_self + out)

    def pool_arrival(self, a_self, cand, loc, n_seg):
        if not self.soft_arrival:
            return a_self.scatter_reduce(0, loc, cand, "amax", include_self=True)
        tau = torch.exp(self.log_tau).to(cand.dtype)
        m = a_self.detach().scatter_reduce(0, loc, cand.detach(), "amax", include_self=True)
        fin = torch.isfinite(m)
        m0 = torch.where(fin, m, torch.zeros_like(m))
        ok_c, ok_s = torch.isfinite(cand), torch.isfinite(a_self)
        c0 = torch.where(ok_c, cand, m0[loc])
        a0 = torch.where(ok_s, a_self, m0)
        e = torch.exp((c0 - m0[loc]) / tau) * ok_c.to(cand.dtype)
        e_self = torch.exp((a0 - m0) / tau) * ok_s.to(cand.dtype)
        s = e_self + torch.zeros_like(a_self).index_add_(0, loc, e)
        num = e_self * a0 + torch.zeros_like(a_self).index_add_(0, loc, e * c0)
        out = torch.where(fin, num / s.clamp_min(1e-30), a_self)
        if self.monitor:
            w_win = torch.zeros_like(a_self).index_add_(0, loc, torch.where(cand == m[loc], e, torch.zeros_like(e))) / s.clamp_min(1e-30)
            n_fin = torch.zeros(n_seg, dtype=torch.long, device=cand.device).index_add_(0, loc, torch.isfinite(cand).long())
            ok = n_fin >= 2
            self._acc("soft_winner_weight", w_win[ok].sum(), ok.sum())
        return out
