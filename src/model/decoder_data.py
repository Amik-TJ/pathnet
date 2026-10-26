from __future__ import annotations

import os
import time
from dataclasses import dataclass, fields

import torch

from .data import SIDECAR_VERSION, design_from_graph, graph_root, load_graph, sidecar_path


@dataclass
class PathLabels:
    ep_worst: torch.Tensor
    ep_rf: torch.Tensor
    tok: torch.Tensor
    tok_ptr: torch.Tensor
    dec_fu: torch.Tensor
    dec_ft: torch.Tensor
    dec_w: torch.Tensor
    tok_rf: torch.Tensor

    def to(self, device) -> "PathLabels":
        return PathLabels(**{f.name: getattr(self, f.name).to(device, non_blocking=True) for f in fields(self)})


def path_labels_from_graph(data, name: str = "") -> PathLabels:
    N = int(data.num_nodes)
    path, rf, ptr = data.y_ep_path.long(), data.y_ep_path_rf.long(), data.y_ep_path_ptr.long()
    ep = data.y_ep_index.long()
    n_ep, P = len(ep), len(path)
    length = ptr[1:] - ptr[:-1]
    if len(ptr) != n_ep + 1 or int(ptr[-1]) != P or (n_ep and int(length.min()) < 2):
        raise ValueError(f"{name}: the endpoint paths are not a CSR over the endpoints with at least 2 pins each")
    if not torch.equal(path[ptr[1:] - 1], ep):
        raise ValueError(f"{name}: a labelled path does not end at its endpoint")
    path_id = torch.repeat_interleave(torch.arange(n_ep), length)
    inside = path_id[:-1] == path_id[1:]
    net = data.edge_index_net.long()
    is_net = torch.isin(path[:-1] * N + path[1:], net[0] * N + net[1])
    arc_step = inside & ~is_net
    if bool(arc_step[ptr[:-1]].any()):
        raise ValueError(f"{name}: a path leaves its startpoint by an arc step")
    is_tok = torch.zeros(P, dtype=torch.bool)
    is_tok[:-1] = arc_step
    is_tok[ptr[1:] - 1] = True
    is_tok[ptr[:-1]] = True
    idx = torch.nonzero(is_tok, as_tuple=True)[0]
    pid = path_id[idx]
    order = torch.argsort(pid * P + (P - 1 - idx))
    tok = path[idx[order]]
    tok_ptr = torch.zeros(n_ep + 1, dtype=torch.long)
    tok_ptr[1:] = torch.cumsum(torch.bincount(pid, minlength=n_ep), 0)
    i = torch.nonzero(arc_step, as_tuple=True)[0]
    M = 2 * N
    key, count = torch.unique((2 * path[i + 1] + rf[i + 1]) * M + (2 * path[i] + rf[i]), return_counts=True)
    return PathLabels(ep_worst=torch.isin(ep, data.y_worst_endpoints.long()), ep_rf=rf[ptr[1:] - 1], tok=tok, tok_ptr=tok_ptr,
                      dec_fu=key % M, dec_ft=key // M, dec_w=count, tok_rf=rf[idx[order]])


def collate_labels(labels: list, b) -> PathLabels:
    off = b.node_ptr[:-1].tolist()
    if len(labels) != len(off):
        raise ValueError("collate_labels: one PathLabels per design of the batch")
    tok_off, ptrs = 0, [torch.zeros(1, dtype=torch.long)]
    for l in labels:
        ptrs.append(l.tok_ptr[1:] + tok_off)
        tok_off += int(l.tok_ptr[-1])
    out = PathLabels(ep_worst=torch.cat([l.ep_worst for l in labels]), ep_rf=torch.cat([l.ep_rf for l in labels]),
                     tok=torch.cat([l.tok + o for l, o in zip(labels, off)]), tok_ptr=torch.cat(ptrs),
                     dec_fu=torch.cat([l.dec_fu + 2 * o for l, o in zip(labels, off)]),
                     dec_ft=torch.cat([l.dec_ft + 2 * o for l, o in zip(labels, off)]),
                     dec_w=torch.cat([l.dec_w for l in labels]), tok_rf=torch.cat([l.tok_rf for l in labels]))
    if len(out.ep_worst) != len(b.ep_index):
        raise ValueError("collate_labels: the labels and the batch do not have the same endpoints")
    return out


def load_design_with_paths(gd: str, root: str | None = None, corner: str = "BC", stage: str = "synth") -> tuple:
    root = root or graph_root(corner)
    data = load_graph(gd, root, corner, stage)
    p = sidecar_path(root, gd, stage)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: build it with  python -m model.data build --stage {stage} --designs {gd}")
    side = torch.load(p, weights_only=True)
    if side["sidecar_version"] != SIDECAR_VERSION:
        raise ValueError(f"{p}: sidecar version {side['sidecar_version']}, expected {SIDECAR_VERSION}")
    return design_from_graph(data, side, gd), path_labels_from_graph(data, gd)


def load_designs_with_paths(names: list, root: str | None = None, corner: str = "BC", workers: int = 8,
                            log_every: int = 500, stage: str = "synth") -> tuple:
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    designs, labels = [None] * len(names), [None] * len(names)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, (d, l) in enumerate(ex.map(lambda n: load_design_with_paths(n, root, corner, stage), names)):
            designs[i], labels[i] = d, l
            if log_every and (i + 1) % log_every == 0:
                print(f"  loaded {i + 1}/{len(names)} designs, {time.time() - t0:.0f}s", flush=True)
    return designs, labels
