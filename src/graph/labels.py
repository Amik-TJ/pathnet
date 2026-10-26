from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass, field

import numpy as np
import torch

TIE_TOL = 1e-3


class LabelError(ValueError):
    pass


@dataclass
class Sweep:
    design_dir: str
    endpoints: list[dict]
    points: dict[int, list[tuple[str, str, float]]]
    summary: dict
    sdc_path: str = ""
    problems: list[str] = field(default_factory=list)


def read_sweep(design_dir: str) -> Sweep:
    with open(os.path.join(design_dir, "endpoints.tsv")) as f:
        endpoints = list(csv.DictReader(f, delimiter="\t"))
    points: dict[int, list] = {}
    with gzip.open(os.path.join(design_dir, "points.tsv.gz"), "rt") as f:
        next(f)
        for line in f:
            pid, idx, pin, edge, arr = line.rstrip("\n").split("\t")
            points.setdefault(int(pid), []).append((pin, edge, float(arr)))
    with open(os.path.join(design_dir, "summary.tsv")) as f:
        summary = next(csv.DictReader(f, delimiter="\t"))
    return Sweep(design_dir, endpoints, points, summary, os.path.join(design_dir, "sta.sdc"))


def attach_labels(data, sweep: Sweep, *, strict: bool = True) -> dict:
    node_of = {n: i for i, n in enumerate(data.pin_names)}
    net = set(map(tuple, data.edge_index_net.t().tolist()))
    arc = set(map(tuple, data.edge_index_arc.t().tolist()))
    role = data.x_cat[:, 0].numpy()
    stats = {"paths": 0, "bad_steps": 0, "bad_step_examples": [], "unknown_pins": 0, "non_alternating": 0}

    def path_nodes(pid: int):
        pins = sweep.points[pid]
        nodes = []
        for pin, edge, arr in pins:
            if pin not in node_of:
                stats["unknown_pins"] += 1
                raise LabelError(f"path {pid}: pin {pin!r} is not a graph node")
            nodes.append(node_of[pin])
        for a, b in zip(nodes[:-1], nodes[1:]):
            if (a, b) not in net and (a, b) not in arc:
                stats["bad_steps"] += 1
                if len(stats["bad_step_examples"]) < 5:
                    stats["bad_step_examples"].append((data.pin_names[a], data.pin_names[b]))
        kinds = ["net" if (a, b) in net else "arc" for a, b in zip(nodes[:-1], nodes[1:])]
        for k1, k2 in zip(kinds[:-1], kinds[1:]):
            if k1 == "net" and k2 == "net":
                stats["non_alternating"] += 1
        rf = [0 if e == "^" else 1 for _, e, _ in pins]
        arr = [a for _, _, a in pins]
        return nodes, rf, arr

    rank0 = [r for r in sweep.endpoints if int(r["rank"]) == 0]
    if not rank0:
        raise LabelError("no rank-0 path in endpoints.tsv")
    ep_index, ep_slack, ep_arr, ep_req = [], [], [], []
    csr_nodes, csr_rf, csr_arr, ptr = [], [], [], [0]
    for r in rank0:
        pid = int(r["path_id"])
        nodes, rf, arr = path_nodes(pid)
        stats["paths"] += 1
        ep = node_of.get(r["endpoint"])
        if ep is None or ep != nodes[-1]:
            raise LabelError(f"endpoint {r['endpoint']} does not end path {pid}")
        ep_index.append(ep); ep_slack.append(float(r["slack_ps"])); ep_arr.append(float(r["arrival_ps"])); ep_req.append(float(r["required_ps"]))
        csr_nodes += nodes; csr_rf += rf; csr_arr += arr; ptr.append(len(csr_nodes))
        if role[nodes[0]] not in (0, 6):
            stats.setdefault("bad_start_roles", 0); stats["bad_start_roles"] += 1
        if role[nodes[-1]] not in (1, 5):
            stats.setdefault("bad_end_roles", 0); stats["bad_end_roles"] += 1
    if strict and (stats["bad_steps"] or stats["non_alternating"] or stats.get("bad_start_roles") or stats.get("bad_end_roles")):
        raise LabelError(f"label paths inconsistent with the graph: {stats}")

    worst = min(ep_slack)
    worst_eps = [e for e, s in zip(ep_index, ep_slack) if abs(s - worst) < TIE_TOL]
    first = rank0[0]
    nodes, rf, arr = path_nodes(int(first["path_id"]))

    data.y_path = torch.tensor(nodes, dtype=torch.long)
    data.y_path_rf = torch.tensor(rf, dtype=torch.long)
    data.y_path_arrival = torch.tensor(arr, dtype=torch.float32)
    data.y_path_slack = torch.tensor(float(first["slack_ps"]), dtype=torch.float32)
    data.y_worst_endpoints = torch.tensor(worst_eps, dtype=torch.long)
    data.y_ep_index = torch.tensor(ep_index, dtype=torch.long)
    data.y_ep_slack = torch.tensor(ep_slack, dtype=torch.float32)
    data.y_ep_arrival = torch.tensor(ep_arr, dtype=torch.float32)
    data.y_ep_required = torch.tensor(ep_req, dtype=torch.float32)
    data.y_ep_path = torch.tensor(csr_nodes, dtype=torch.long)
    data.y_ep_path_ptr = torch.tensor(ptr, dtype=torch.long)
    data.y_ep_path_rf = torch.tensor(csr_rf, dtype=torch.long)
    data.y_ep_path_arrival = torch.tensor(csr_arr, dtype=torch.float32)
    data.sweep_summary = dict(sweep.summary)
    stats["worst_slack"] = worst
    stats["tied_worst_endpoints"] = len(worst_eps)
    stats["summary_tied"] = int(sweep.summary.get("n_endpoints_tied_worst", -1))
    stats["summary_worst_slack"] = float(sweep.summary.get("worst_slack_ps", "nan"))
    return stats
