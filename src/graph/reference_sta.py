from __future__ import annotations

import numpy as np

from .nldm import lookup
from .vocab import RF_PAIRS, Vocab

NEG_INF = -np.inf


def run_reference_sta(data, vocab: Vocab, slew_merge_all: bool = True) -> dict:
    N = int(data.num_nodes)
    role = data.x_cat[:, 0].numpy()
    xn = data.x_num.double().numpy()
    load = xn[:, 2:4]
    input_delay, output_delay, is_const, is_clk_port = xn[:, 5], xn[:, 6], xn[:, 7] > 0, xn[:, 8] > 0
    period = float(data.g[0, 0])
    level = data.level.numpy()
    slew_axis = vocab.slew_axis

    arr = np.full((N, 2), NEG_INF)
    slew = np.zeros((N, 2))
    pred_node = -np.ones((N, 2), dtype=np.int64)
    pred_rf = -np.ones((N, 2), dtype=np.int64)
    pred_arc = -np.ones((N, 2), dtype=np.int64)

    for v in np.flatnonzero(role == 0):
        if is_const[v]:
            continue
        if is_clk_port[v]:
            arr[v, 0] = 0.0; arr[v, 1] = period / 2.0
        else:
            arr[v, :] = input_delay[v]
        slew[v, :] = 0.0

    net_src, net_dst = data.edge_index_net.numpy()
    arc_src, arc_dst = data.edge_index_arc.numpy()
    arc_key = data.arc_key_id.numpy(); arc_mask = data.arc_rf_mask.numpy(); arc_on = data.arc_enabled.numpy()
    arc_type = data.arc_type.numpy()

    clk_arr = np.full((N, 2), np.nan)
    clk_ports = np.flatnonzero(is_clk_port)
    for v in clk_ports:
        clk_arr[v] = (0.0, period / 2.0)
    order = np.argsort(level, kind="stable")
    net_by_dst = {}
    for e, d in enumerate(net_dst):
        net_by_dst.setdefault(int(d), []).append(int(net_src[e]))
    arcs_by_dst = {}
    for e, d in enumerate(arc_dst):
        arcs_by_dst.setdefault(int(d), []).append(e)
    for v in order:
        v = int(v)
        if role[v] in (1, 2, 4, 5):
            for d in net_by_dst.get(v, []):
                clk_arr[v] = clk_arr[d]
        elif role[v] == 3:
            for e in arcs_by_dst.get(v, []):
                if arc_type[e] == 1 or not arc_on[e]:
                    continue
                u = int(arc_src[e])
                for pair_idx, (frf, trf) in enumerate(RF_PAIRS):
                    if arc_mask[e, pair_idx] and not np.isnan(clk_arr[u, frf]):
                        t = clk_arr[u, frf]
                        clk_arr[v, trf] = t if np.isnan(clk_arr[v, trf]) else max(clk_arr[v, trf], t)
    has_clock = ~np.isnan(clk_arr[:, 0])
    by_level_net = {}
    for e, d in enumerate(net_dst):
        by_level_net.setdefault(level[d], []).append(e)
    by_level_arc = {}
    for e, d in enumerate(arc_dst):
        by_level_arc.setdefault(level[d], []).append(e)
    bank, caps = vocab.arc_bank, vocab.arc_cap_axis

    for l in range(1, int(level.max()) + 1):
        for e in by_level_net.get(l, []):
            d, s = net_src[e], net_dst[e]
            slew[s] = slew[d]
            if is_const[s] or is_const[d]:
                continue
            arr[s] = arr[d]
            pred_node[s] = d; pred_rf[s] = np.array([0, 1]); pred_arc[s] = -1
        for e in by_level_arc.get(l, []):
            u, o, k = arc_src[e], arc_dst[e], arc_key[e]
            searchable = arc_on[e] and not is_const[u] and not is_const[o]
            launch = arc_type[e] == 1
            for pair_idx, (frf, trf) in enumerate(RF_PAIRS):
                if not arc_mask[e, pair_idx]:
                    continue
                s_in = 0.0 if launch else slew[u, frf]
                c = load[o, trf]
                from_arr = clk_arr[u, frf] if launch else arr[u, frf]
                if np.isnan(from_arr):
                    from_arr = NEG_INF
                if slew_merge_all or (searchable and from_arr != NEG_INF):
                    sl = float(lookup(bank[k, 2 if trf == 0 else 3], slew_axis, caps[k], s_in, c))
                    if sl > slew[o, trf]:
                        slew[o, trf] = sl
                if not searchable or from_arr == NEG_INF:
                    continue
                d = float(lookup(bank[k, 0 if trf == 0 else 1], slew_axis, caps[k], s_in, c))
                cand = from_arr + d
                if cand > arr[o, trf]:
                    arr[o, trf] = cand; pred_node[o, trf] = u; pred_rf[o, trf] = frf; pred_arc[o, trf] = e

    req = np.full((N, 2), np.nan)
    chk_src, chk_dst = data.edge_index_check.numpy()
    chk_tables = data.check_tables.double().numpy(); chk_axes = data.check_axes.double().numpy()
    n_inverted_capture = 0
    for e, dpin in enumerate(chk_dst):
        clkpin = int(chk_src[e])
        if not has_clock[clkpin]:
            continue
        edge = clk_arr[clkpin, 0]
        capture = period if edge == 0.0 else edge
        if edge != 0.0:
            n_inverted_capture += 1
        for rf in (0, 1):
            setup = float(lookup(chk_tables[e, rf], chk_axes[e, rf, 0], chk_axes[e, rf, 1], slew[dpin, rf], 0.0))
            req[dpin, rf] = capture - setup
    for v in np.flatnonzero(role == 1):
        req[v, :] = period - output_delay[v]
    slack = req - arr
    masked = np.where(np.isnan(slack) | (arr == NEG_INF), np.inf, slack)
    ep_rf = np.argmin(masked, axis=1)
    ep_slack = masked[np.arange(N), ep_rf]
    ep_slack = np.where(np.isinf(ep_slack), np.nan, ep_slack)
    n_unclocked = int(sum(1 for c in chk_src if not has_clock[int(c)]))
    return {"arr": arr, "slew": slew, "req": req, "slack": slack, "ep_slack": ep_slack, "ep_rf": ep_rf,
            "pred_node": pred_node, "pred_rf": pred_rf, "pred_arc": pred_arc, "clk_arr": clk_arr,
            "n_registers": int(len(chk_src)), "n_unclocked_registers": n_unclocked,
            "n_inverted_capture": n_inverted_capture, "n_clock_network_pins": int(has_clock.sum())}


def backtrack(res: dict, node: int, rf: int, role: np.ndarray) -> list[tuple[int, int]]:
    path = [(node, rf)]
    seen = set()
    while role[node] not in (0, 6):
        p, prf = res["pred_node"][node, rf], res["pred_rf"][node, rf]
        if p < 0 or (node, rf) in seen:
            break
        seen.add((node, rf))
        node, rf = int(p), int(prf)
        path.append((node, rf))
    return path[::-1]


def compare_with_labels(data, res: dict, tol: float = 0.01, rel_tol: float = 2e-6) -> dict:
    role = data.x_cat[:, 0].numpy()
    out = {}
    if not hasattr(data, "y_ep_index"):
        return {"note": "no labels attached"}
    arr_scale = float(np.max(np.abs(data.y_ep_path_arrival.double().numpy()))) if data.y_ep_path_arrival.numel() else 0.0
    tol = max(tol, rel_tol * arr_scale)
    out["eff_tol"] = tol
    out["arr_scale"] = arr_scale
    ep = data.y_ep_index.numpy(); y_slack = data.y_ep_slack.double().numpy()
    calc = res["ep_slack"][ep]
    err = np.abs(calc - y_slack)
    out["endpoint_slack"] = {"n": int(len(ep)), "max_abs_err": float(np.nanmax(err)), "mean_abs_err": float(np.nanmean(err)),
                             "n_within_tol": int((err <= tol).sum()), "n_nan": int(np.isnan(err).sum())}
    nodes = data.y_ep_path.numpy(); rf = data.y_ep_path_rf.numpy(); y_arr = data.y_ep_path_arrival.double().numpy()
    calc_arr = res["arr"][nodes, rf]
    aerr = np.abs(calc_arr - y_arr)
    out["path_arrival"] = {"n": int(len(nodes)), "max_abs_err": float(np.max(aerr)), "mean_abs_err": float(np.mean(aerr)),
                           "n_within_tol": int((aerr <= tol).sum()), "n_minus_inf": int(np.isinf(calc_arr).sum())}
    worst_i = int(np.argmax(aerr))
    out["path_arrival"]["worst_example"] = (data.pin_names[nodes[worst_i]], "rise" if rf[worst_i] == 0 else "fall", float(y_arr[worst_i]), float(calc_arr[worst_i]))
    ws = res["ep_slack"].copy(); ws[~np.isin(np.arange(len(ws)), ep)] = np.nan
    worst = np.nanmin(ws)
    calc_set = set(np.flatnonzero(np.abs(ws - worst) < 1e-3).tolist())
    label_set = set(data.y_worst_endpoints.tolist())
    label_worst = float(data.y_path_slack)
    slack_of_label_ep = dict(zip(ep.tolist(), y_slack.tolist()))
    strict = calc_set == label_set
    loose = (all(abs(ws[e] - worst) < 1e-3 + tol for e in label_set)
             and all(abs(slack_of_label_ep.get(e, np.inf) - label_worst) < 1e-3 + tol for e in calc_set))
    out["worst_endpoints"] = {"label": sorted(label_set), "calc": sorted(calc_set), "match": strict or loose,
                              "exact_set_match": strict, "label_slack": label_worst, "calc_slack": float(worst)}
    e0 = int(data.y_path[-1]); path = backtrack(res, e0, int(res["ep_rf"][e0]), role)
    out["worst_path"] = {"label": data.y_path.tolist(), "calc": [n for n, _ in path], "match": [n for n, _ in path] == data.y_path.tolist(),
                         "label_rf": data.y_path_rf.tolist(), "calc_rf": [r for _, r in path]}
    ptr = data.y_ep_path_ptr.numpy(); ok = 0; ties = 0
    for i in range(len(ep)):
        lab = nodes[ptr[i]:ptr[i + 1]].tolist()
        if np.isnan(res["ep_slack"][ep[i]]):
            continue
        calc = [n for n, _ in backtrack(res, int(ep[i]), int(res["ep_rf"][ep[i]]), role)]
        if calc == lab:
            ok += 1
        elif np.all(aerr[ptr[i]:ptr[i + 1]] <= tol) and abs(res["ep_slack"][ep[i]] - y_slack[i]) <= tol:
            ties += 1
    out["per_endpoint_path_match"] = {"n": int(len(ep)), "match": int(ok), "tie_equivalent": int(ties)}
    return out
