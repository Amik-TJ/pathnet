from __future__ import annotations

import torch

from cell_model.data import exact_lookup

NEG_INF = float("-inf")


def exact_setup(tables: torch.Tensor, axes: torch.Tensor, s_d: torch.Tensor) -> torch.Tensor:
    t, ax, s = tables.double(), axes.double(), s_d.double()
    cols = [exact_lookup(t[:, rf], ax[:, rf, 0], ax[:, rf, 1], s[:, rf], torch.zeros_like(s[:, rf])) for rf in (0, 1)]
    return torch.stack(cols, 1)


def endpoint_slack(b, arr: torch.Tensor, slew: torch.Tensor, setup_fn, knockout: bool = False) -> dict:
    N = b.num_nodes
    dt = arr.dtype
    period = b.period.to(dt)
    req = torch.full((N, 2), float("nan"), dtype=dt, device=arr.device)
    if b.chk.shape[1]:
        c, d = b.chk
        edge = b.clk_arr[c, 0].to(dt)
        ok = ~torch.isnan(edge)
        if ok.any():
            c, d, edge = c[ok], d[ok], edge[ok]
            capture = torch.where(edge == 0.0, period[b.node_graph[d]], edge)
            if knockout:
                setup = setup_fn(b.chk_tables[ok], b.chk_axes[ok], slew[d], knockout=True)
            else:
                setup = setup_fn(b.chk_tables[ok], b.chk_axes[ok], slew[d])
            req = req.index_put((d,), capture[:, None] - setup.to(dt))
    out_port = b.x_cat[:, 0] == 1
    od = b.x_num[:, 6].to(dt)
    req = torch.where(out_port[:, None], (period[b.node_graph] - od)[:, None].expand(N, 2), req)
    slack = req - arr
    masked = torch.where(torch.isnan(slack) | (arr == NEG_INF), torch.full_like(slack, float("inf")), slack)
    ep_rf = torch.argmin(masked, dim=1)
    ep_slack = masked.gather(1, ep_rf[:, None])[:, 0]
    ep_slack = torch.where(torch.isinf(ep_slack), torch.full_like(ep_slack, float("nan")), ep_slack)
    return {"req": req, "slack": slack, "ep_slack": ep_slack, "ep_rf": ep_rf}


def l1_masked(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
    ok = m & torch.isfinite(pred)
    left_out = int((m & ~torch.isfinite(pred)).sum())
    e = (pred[ok] - y[ok].to(pred.dtype)).abs()
    return e.sum(), int(ok.sum()), left_out


@torch.no_grad()
def arc_delay_targets(out: dict, b, bank) -> tuple:
    from .encoder import instance_level_offset
    sch = out["schedule"]
    k = instance_level_offset(sch)
    n = len(out["msg_delay"]) if "msg_delay" in out else len(sch.inst_e) - k
    e, trf, fu, o, launch = (sch.inst_e[k:k + n], sch.inst_trf[k:k + n], sch.inst_fu[k:k + n], sch.inst_o[k:k + n], sch.inst_launch[k:k + n])
    y_slew = b.y_slew.reshape(-1).double()
    s_in = torch.where(launch, torch.zeros_like(y_slew[fu]), torch.nan_to_num(y_slew[fu], nan=0.0))
    load = b.x_num.double()[:, 2:4][o, trf]
    y = bank.targets(b.arc_key[e], trf, s_in, load).float()
    mask = launch | b.m_slew.reshape(-1)[fu]
    return y, mask


def arc_losses(out: dict, y: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None, n_graph: int = 0) -> tuple:
    sd, nd, _ = l1_masked(out["msg_delay"], y[:, 0], mask)
    ss, ns, _ = l1_masked(out["msg_slew"], y[:, 1], mask)
    if graph is None:
        total = (sd / nd if nd else 0.0) + (ss / ns if ns else 0.0)
    else:
        total = l1_by_design(out["msg_delay"], y[:, 0], mask, graph, n_graph) + l1_by_design(out["msg_slew"], y[:, 1], mask, graph, n_graph)
    return total, {"arc_delay_sum": float(sd), "arc_slew_sum": float(ss), "arc_n": nd}


def l1_by_design(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, graph: torch.Tensor, n_graph: int):
    ok = m & torch.isfinite(pred)
    g = graph.expand_as(pred)[ok]
    e = (pred[ok] - y[ok].to(pred.dtype)).abs()
    s = torch.zeros(n_graph, dtype=e.dtype, device=e.device).index_add_(0, g, e)
    c = torch.zeros(n_graph, dtype=e.dtype, device=e.device).index_add_(0, g, torch.ones_like(e))
    has = c > 0
    return (s[has] / c[has]).mean() if bool(has.any()) else e.sum() * 0.0


def dense_terms(out: dict, b, per_design: bool = False) -> dict:
    B = len(b.names)
    terms = {}
    for name, pred, y, m, g in (("arr", out["arr"], b.y_arrival, b.m_arr, b.node_graph[:, None]),
                                ("slew", out["slew"], b.y_slew, b.m_slew, b.node_graph[:, None]),
                                ("slack", out["ep_slack"][b.ep_index], b.ep_slack, torch.isfinite(b.ep_slack), b.ep_graph)):
        if per_design:
            terms[name] = l1_by_design(pred, y, m, g.expand_as(pred), B)
        else:
            s, n, _ = l1_masked(pred, y, m)
            terms[name] = s / n if n else s * 0.0
    return terms


def dense_losses(out: dict, b, w_arr: float = 1.0, w_slew: float = 1.0, w_slack: float = 1.0) -> tuple:
    sa, na, ua = l1_masked(out["arr"], b.y_arrival, b.m_arr)
    ss, ns, _ = l1_masked(out["slew"], b.y_slew, b.m_slew)
    ep_pred = out["ep_slack"][b.ep_index]
    se, ne, ue = l1_masked(ep_pred, b.ep_slack, torch.isfinite(b.ep_slack))
    total = 0.0
    for w, s, n in ((w_arr, sa, na), (w_slew, ss, ns), (w_slack, se, ne)):
        if w and n:
            total = total + w * s / n
    stats = {"arr_sum": float(sa), "arr_n": na, "slew_sum": float(ss), "slew_n": ns, "slack_sum": float(se), "slack_n": ne,
             "arr_unreached": ua, "slack_unreached": ue}
    return total, stats
