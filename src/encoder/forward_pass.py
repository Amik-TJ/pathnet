from __future__ import annotations

import numpy as np
import torch

from cell_model.data import Bank, featurise
from graph.nldm import lookup
from graph.vocab import RF_PAIRS
from walk.oracle import OracleScorer

NEG_INF = float("-inf")


def exact_cell_fn(bank: Bank):
    def fn(key, trf, s, c):
        y = bank.targets(key, trf, s, c)
        return y[:, 0], y[:, 1]
    return fn


def model_cell_fn(bank: Bank, model):
    @torch.no_grad()
    def fn(key, trf, s, c):
        y = model(featurise(bank, key, trf, s, c))[0].double()
        return y[:, 0], y[:, 1]
    return fn


def clock_network(data) -> np.ndarray:
    N = int(data.num_nodes)
    role = data.x_cat[:, 0].numpy()
    period = float(data.g[0, 0])
    level = data.level.numpy()
    net_src, net_dst = data.edge_index_net.numpy()
    arc_src, arc_dst = data.edge_index_arc.numpy()
    arc_mask = data.arc_rf_mask.numpy(); arc_on = data.arc_enabled.numpy(); arc_type = data.arc_type.numpy()
    clk = np.full((N, 2), np.nan)
    for v in np.flatnonzero(data.x_num[:, 8].numpy() > 0):
        clk[v] = (0.0, period / 2.0)
    net_by_dst, arcs_by_dst = {}, {}
    for e, d in enumerate(net_dst):
        net_by_dst.setdefault(int(d), []).append(int(net_src[e]))
    for e, d in enumerate(arc_dst):
        arcs_by_dst.setdefault(int(d), []).append(e)
    for v in np.argsort(level, kind="stable"):
        v = int(v)
        if role[v] in (1, 2, 4, 5):
            for d in net_by_dst.get(v, []):
                clk[v] = clk[d]
        elif role[v] == 3:
            for e in arcs_by_dst.get(v, []):
                if arc_type[e] == 1 or not arc_on[e]:
                    continue
                u = int(arc_src[e])
                for i, (frf, trf) in enumerate(RF_PAIRS):
                    if arc_mask[e, i] and not np.isnan(clk[u, frf]):
                        t = clk[u, frf]
                        clk[v, trf] = t if np.isnan(clk[v, trf]) else max(clk[v, trf], t)
    return clk


def forward_pass(data, cell_fn, device="cpu", clk_arr: np.ndarray | None = None) -> dict:
    dev = torch.device(device)
    f64 = dict(dtype=torch.float64, device=dev)
    N = int(data.num_nodes)
    xn = data.x_num.double()
    role = data.x_cat[:, 0].numpy()
    is_const = (xn[:, 7] > 0).to(dev)
    is_clk_port = xn[:, 8].numpy() > 0
    load = xn[:, 2:4].to(dev)
    period = float(data.g[0, 0])
    level = data.level.to(dev)
    if clk_arr is None:
        clk_arr = clock_network(data)
    clk = torch.tensor(clk_arr, **f64)

    arr = torch.full((N, 2), NEG_INF, **f64)
    slew = torch.zeros((N, 2), **f64)
    for v in np.flatnonzero(role == 0):
        if bool(is_const[v]):
            continue
        if is_clk_port[v]:
            arr[v, 0] = 0.0; arr[v, 1] = period / 2.0
        else:
            arr[v, :] = float(xn[v, 5])

    net = data.edge_index_net.to(dev)
    arc = data.edge_index_arc.to(dev)
    key = data.arc_key_id.to(dev)
    mask = data.arc_rf_mask.to(dev)
    launch = (data.arc_type == 1).to(dev)
    searchable = data.arc_enabled.to(dev) & ~is_const[arc[0]] & ~is_const[arc[1]]
    arc_delay = torch.full((arc.shape[1], 4), float("nan"), **f64)

    def by_level(dst):
        lv = level[dst]
        order = torch.argsort(lv, stable=True)
        counts = torch.bincount(lv, minlength=int(level.max()) + 1)
        return order, torch.cumsum(counts, 0) - counts, counts
    n_order, n_start, n_count = by_level(net[1])
    a_order, a_start, a_count = by_level(arc[1])

    for l in range(1, int(level.max()) + 1):
        if n_count[l]:
            e = n_order[n_start[l]:n_start[l] + n_count[l]]
            d, s = net[0, e], net[1, e]
            slew[s] = slew[d]
            ok = ~(is_const[s] | is_const[d])
            arr[s[ok]] = arr[d[ok]]
        if not a_count[l]:
            continue
        e_all = a_order[a_start[l]:a_start[l] + a_count[l]]
        parts = []
        for i, (frf, trf) in enumerate(RF_PAIRS):
            e = e_all[mask[e_all, i]]
            if len(e):
                parts.append((e, torch.full_like(e, i), torch.full_like(e, frf), torch.full_like(e, trf)))
        if not parts:
            continue
        e, pi, frf, trf = (torch.cat([p[j] for p in parts]) for j in range(4))
        u, o = arc[0, e], arc[1, e]
        is_launch = launch[e]
        s_in = torch.where(is_launch, torch.zeros((), **f64), slew[u, frf])
        d, sl = cell_fn(key[e], trf, s_in, load[o, trf])
        arc_delay[e, pi] = d
        flat = o * 2 + trf
        slew.view(-1).scatter_reduce_(0, flat, sl, reduce="amax", include_self=True)
        from_arr = torch.where(is_launch, torch.nan_to_num(clk[u, frf], nan=NEG_INF), arr[u, frf])
        ok = searchable[e] & (from_arr > NEG_INF)
        arr.view(-1).scatter_reduce_(0, flat[ok], (from_arr + d)[ok], reduce="amax", include_self=True)

    arr_n, slew_n = arr.cpu().numpy(), slew.cpu().numpy()
    req = np.full((N, 2), np.nan)
    chk_src, chk_dst = data.edge_index_check.numpy()
    chk_tables = data.check_tables.double().numpy(); chk_axes = data.check_axes.double().numpy()
    has_clock = ~np.isnan(clk_arr[:, 0])
    for e, dpin in enumerate(chk_dst):
        clkpin = int(chk_src[e])
        if not has_clock[clkpin]:
            continue
        edge = clk_arr[clkpin, 0]
        capture = period if edge == 0.0 else edge
        for rf in (0, 1):
            req[dpin, rf] = capture - float(lookup(chk_tables[e, rf], chk_axes[e, rf, 0], chk_axes[e, rf, 1], slew_n[dpin, rf], 0.0))
    for v in np.flatnonzero(role == 1):
        req[v, :] = period - float(xn[v, 6])
    slack = req - arr_n
    masked = np.where(np.isnan(slack) | (arr_n == NEG_INF), np.inf, slack)
    ep_rf = np.argmin(masked, axis=1)
    ep_slack = masked[np.arange(N), ep_rf]
    ep_slack = np.where(np.isinf(ep_slack), np.nan, ep_slack)
    return {"arr": arr_n, "slew": slew_n, "req": req, "slack": slack, "ep_slack": ep_slack, "ep_rf": ep_rf,
            "clk_arr": clk_arr, "arc_delay": arc_delay.cpu().numpy()}


class StoredDelayScorer(OracleScorer):

    def __init__(self, g, res: dict, vocab):
        super().__init__(g, res, vocab)
        self.stored = res["arc_delay"]

    def arc_delay(self, e, u, o, frf, trf):
        return float(self.stored[e, 2 * frf + trf])
