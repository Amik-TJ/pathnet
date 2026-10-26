from __future__ import annotations

import torch

from .encoder import instance_level_offset

NEG_INF = float("-inf")
CLOSE_PS = 1.0
MARGIN_DEFAULTS = {"w_margin": 0.0,
                   "margin_eps": 0.1,
                   "margin_transition": 1,
                   "margin_max_gap": 0.0,
                   "margin_log": 0}


@torch.no_grad()
def true_scores(out: dict, b, y_arc: torch.Tensor, m_arc: torch.Tensor) -> tuple:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    n = len(y_arc)
    fu, launch, search = sch.inst_fu[k:k + n], sch.inst_launch[k:k + n], sch.inst_search[k:k + n]
    clk = b.clk_arr.reshape(-1).double()[fu]
    y = b.y_arrival.reshape(-1).double()[fu]
    src = torch.where(launch, clk, y)
    ok = search & m_arc & torch.isfinite(src) & (launch | b.m_arr.reshape(-1)[fu])
    return torch.where(ok, src + y_arc[:, 0].double(), torch.full_like(src, NEG_INF)), ok


def margin_loss(out: dict, b, pos: dict, y_arc: torch.Tensor, m_arc: torch.Tensor, eps: float = 0.1,
                transition: bool = True, max_gap: float = 0.0) -> tuple:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    a_hat = out["msg_arrival"]
    n, G, N, dev = len(a_hat), len(sch.seg_ft), b.num_nodes, a_hat.device
    seg, u, fu = sch.inst_seg[k:k + n], sch.inst_u[k:k + n], sch.inst_fu[k:k + n]
    a_true, has_true = true_scores(out, b, y_arc, m_arc)
    cand = pos["lab_seg"][seg] & torch.isfinite(a_hat)
    idx = torch.nonzero(cand & has_true, as_tuple=True)[0]
    seg_i, own_i = seg[idx], pos["pin_pos"][idx]
    sub = torch.where(own_i, fu[idx], 2 * u[idx]) if transition else 2 * u[idx]
    key, inv = torch.unique(seg_i * (2 * N) + sub, return_inverse=True)
    P = len(key)
    g = key // (2 * N)
    s_hat = torch.full((P,), NEG_INF, dtype=a_hat.dtype, device=dev).scatter_reduce(0, inv, a_hat[idx], "amax", include_self=True)
    s_true = torch.full((P,), NEG_INF, dtype=torch.float64, device=dev).scatter_reduce(0, inv, a_true[idx], "amax", include_self=True)
    flag = lambda m: torch.zeros(P, dtype=torch.bool, device=dev).index_put((inv[m],), torch.ones((), dtype=torch.bool, device=dev))
    own = flag(own_i)
    win = flag(pos["pos"][idx]) if transition else own

    def top(score, m, dtype):
        return torch.full((G,), NEG_INF, dtype=dtype, device=dev).scatter_reduce(0, g[m], score[m], "amax", include_self=True)

    w_hat, w_true = top(s_hat, win, a_hat.dtype), top(s_true, win, torch.float64)
    riv = ~win & torch.isfinite(w_true[g])
    gr = g[riv]
    d_hat = w_hat[gr] - s_hat[riv]
    d_true = w_true[gr] - s_true[riv]
    if max_gap > 0:
        keep = d_true <= max_gap
        gr, d_hat, d_true = gr[keep], d_hat[keep], d_true[keep]
    weight = 1.0 / (d_true.clamp_min(0.0) + eps)
    pen = weight.to(a_hat.dtype) * (d_hat - d_true.to(a_hat.dtype)).abs()
    cnt = torch.zeros(G, dtype=a_hat.dtype, device=dev).index_add_(0, gr, torch.ones_like(pen))
    tot = torch.zeros(G, dtype=a_hat.dtype, device=dev).index_add_(0, gr, pen)
    sel = cnt > 0
    per, w = tot[sel] / cnt[sel], pos["w"][sel]
    loss = (per * w).sum() / w.sum() if len(per) else a_hat.new_zeros(())
    with torch.no_grad():
        other = ~own & torch.isfinite(w_true[g])
        best_hat, best_true = top(s_hat, other, a_hat.dtype), top(s_true, other, torch.float64)
        pin_hat, pin_true = top(s_hat, own, a_hat.dtype), top(s_true, own, torch.float64)
        missing = cand & ~has_true
        whole = torch.zeros(G, dtype=torch.long, device=dev).index_add_(0, seg[missing], torch.ones_like(seg[missing])) == 0
        both = torch.isfinite(pin_true) & torch.isfinite(best_true) & whole
        true_gap, pred_gap = (pin_true - best_true)[both], (pin_hat - best_hat)[both]
        pins = torch.unique(seg[cand] * N + u[cand]) // N
        n_pin = torch.zeros(G, dtype=torch.long, device=dev).index_add_(0, pins, torch.ones_like(pins))
        stats = {"margin_sum": float((per * w).sum()), "margin_n": float(w.sum()), "margin_decisions": int(sel.sum()),
                 "margin_rivals": int(len(gr)), "margin_not_max": int((true_gap < 0).sum()),
                 "margin_no_true": int(missing.sum()), "margin_skipped": int(((n_pin >= 2) & ~both).sum())}
        detail = {"seg": torch.nonzero(both, as_tuple=True)[0], "w": pos["w"][both], "true_gap": true_gap, "pred_gap": pred_gap.detach(),
                  "no_true_seg": torch.nonzero(pos["lab_seg"] & ~whole, as_tuple=True)[0]}
    return loss, stats, detail
