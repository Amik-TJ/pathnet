from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import zlib
from dataclasses import dataclass, fields, replace
from multiprocessing import Pool

import numpy as np
import torch

from paths import DATA

SIDECAR = "dense_labels.pt"
SIDECAR_VERSION = 1
LABEL_STAGES = ("synth", "place", "groute")
PATH_LABELS_VERSION = 1
PATH_LABEL_KEYS = ("y_path", "y_path_rf", "y_path_arrival", "y_path_slack", "y_worst_endpoints", "y_ep_index", "y_ep_slack",
                   "y_ep_arrival", "y_ep_required", "y_ep_path", "y_ep_path_ptr", "y_ep_path_rf", "y_ep_path_arrival")
VALUE_COLS = ("arrival_rise_ps", "arrival_fall_ps", "slew_rise_ps", "slew_fall_ps", "slack_ps", "slack_rise_ps", "slack_fall_ps")
ROLE_DIR = {0: "input", 1: "output", 2: "input", 3: "output", 4: "input", 5: "input", 6: "output"}
SPLITS = ("train", "val", "test", "excluded")


def graph_root(corner: str = "BC") -> str:
    return f"{DATA}/graphs_{corner}"


def default_split_file(corner: str = "BC") -> str:
    return f"{DATA}/splits/split_paper_{corner}.tsv"


def read_split(path: str) -> dict:
    out = {s: [] for s in SPLITS}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] not in out:
                raise ValueError(f"{path}: unknown split {r['split']!r} for {r['group']}/{r['design']}")
            out[r["split"]].append(f"{r['group']}/{r['design']}")
    return out


def select_designs(path: str, splits=("train", "val"), val_only=(), allow_test: bool = False) -> dict:
    s = read_split(path)
    if "excluded" in splits:
        raise ValueError("the excluded designs are used nowhere")
    if "test" in splits and not allow_test:
        raise ValueError("the test designs are report-only: pass allow_test=True")
    moved = [d for d in val_only if d in s["train"]]
    unknown = [d for d in val_only if d not in s["train"] and d not in s["val"]]
    if unknown:
        raise ValueError(f"val_only designs not in train or val: {unknown}")
    s["train"] = [d for d in s["train"] if d not in set(moved)]
    s["val"] = s["val"] + moved
    return {k: s[k] for k in splits}


def read_dense(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype={"name": str, "kind": str, "direction": str},
                     na_values={c: ["NA"] for c in VALUE_COLS}, keep_default_na=False)
    if tuple(df.columns[3:]) != VALUE_COLS:
        raise ValueError(f"{path}: unexpected header {list(df.columns)}")
    return list(df["name"]), df["kind"].to_numpy(), df["direction"].to_numpy(), df[list(VALUE_COLS)].to_numpy(np.float64)


def _csr(src: np.ndarray, dst: np.ndarray, n: int):
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, src + 1, 1)
    return np.cumsum(ptr), dst[order]


def _neighbours(ptr: np.ndarray, adj: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    cnt = ptr[nodes + 1] - ptr[nodes]
    if cnt.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.repeat(ptr[nodes], cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return adj[start + off]


def f1_cone(data, clk_arr: np.ndarray) -> np.ndarray:
    N = int(data.num_nodes)
    chk_clk = data.edge_index_check[0].numpy()
    unclocked = np.unique(chk_clk[np.isnan(clk_arr[chk_clk, 0])])
    a_src, a_dst = data.edge_index_arc.numpy()
    n_src, n_dst = data.edge_index_net.numpy()
    cone = np.zeros(N, dtype=bool)
    frontier = np.unique(a_dst[np.isin(a_src, unclocked)])
    if len(frontier) == 0:
        return cone
    ptr, adj = _csr(np.concatenate([n_src, a_src]), np.concatenate([n_dst, a_dst]), N)
    while len(frontier):
        cone[frontier] = True
        nxt = _neighbours(ptr, adj, frontier)
        frontier = np.unique(nxt[~cone[nxt]])
    return cone


def f3_pins(data) -> np.ndarray:
    N = int(data.num_nodes)
    outdeg = np.zeros(N, dtype=np.int64)
    np.add.at(outdeg, data.edge_index_net[0].numpy(), 1)
    np.add.at(outdeg, data.edge_index_arc[0].numpy(), 1)
    return (data.x_cat[:, 0].numpy() == 0) & (outdeg == 0)


def _check_stage(stage: str) -> str:
    if stage not in LABEL_STAGES:
        raise ValueError(f"unknown label stage {stage!r}; one of {LABEL_STAGES}")
    return stage


def sidecar_path(root: str, gd: str, stage: str = "synth") -> str:
    _check_stage(stage)
    return os.path.join(root, gd, SIDECAR if stage == "synth" else f"dense_labels_{stage}.pt")


def path_labels_path(root: str, gd: str, stage: str) -> str:
    if _check_stage(stage) == "synth":
        raise ValueError("the synth path labels are the y_* fields of graph.pt")
    return os.path.join(root, gd, f"path_labels_{stage}.pt")


def post_place_root(corner: str = "BC") -> str:
    return f"{DATA}/post_place_{corner}"


def build_path_labels(gd: str, stage: str, root: str | None = None, corner: str = "BC",
                      sweep_root: str | None = None, out_path: str | None = None) -> dict:
    from graph.labels import attach_labels, read_sweep
    if _check_stage(stage) == "synth":
        raise ValueError("synth path labels are graph.pt's own")
    root = root or graph_root(corner)
    sweep_dir = os.path.join(sweep_root or post_place_root(corner), stage, gd)
    if not os.path.exists(os.path.join(sweep_dir, "DONE")):
        raise FileNotFoundError(f"{sweep_dir}: no DONE")
    t0 = time.time()
    data = torch.load(os.path.join(root, gd, "graph.pt"), weights_only=False)
    synth_ep = data.y_ep_index.clone()
    synth_path, synth_slack = data.y_path.clone(), float(data.y_path_slack)
    stats = attach_labels(data, read_sweep(sweep_dir), strict=True)
    if not torch.equal(torch.sort(synth_ep).values, torch.sort(data.y_ep_index).values):
        raise ValueError(f"{gd} {stage}: labelled endpoints differ from graph.pt ({len(synth_ep)} vs {len(data.y_ep_index)})")
    with open(os.path.join(sweep_dir, "endpoints.tsv"), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    side = {"path_labels_version": PATH_LABELS_VERSION, "design": gd, "stage": stage, "source": sweep_dir,
            "endpoints_md5": md5, "num_nodes": int(data.num_nodes), "sweep_summary": dict(data.sweep_summary)}
    side.update({k: getattr(data, k) for k in PATH_LABEL_KEYS})
    out = out_path or path_labels_path(root, gd, stage)
    tmp = out + f".tmp{os.getpid()}"
    torch.save(side, tmp)
    os.replace(tmp, out)
    return {"design": gd, "stage": stage, "endpoints": int(len(data.y_ep_index)), "path_len": int(len(data.y_path)),
            "worst_slack_synth": round(synth_slack, 4), "worst_slack": round(float(data.y_path_slack), 4),
            "same_worst_path_as_synth": bool(torch.equal(synth_path, data.y_path)),
            "tied_worst": int(len(data.y_worst_endpoints)), "bad_steps": stats["bad_steps"],
            "seconds": round(time.time() - t0, 2), "error": ""}


def load_graph(gd: str, root: str | None = None, corner: str = "BC", stage: str = "synth"):
    root = root or graph_root(corner)
    data = torch.load(os.path.join(root, gd, "graph.pt"), weights_only=False)
    if _check_stage(stage) != "synth":
        p = path_labels_path(root, gd, stage)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}: build it with  python -m model.data build-path --stage {stage} --designs {gd}")
        side = torch.load(p, weights_only=False)
        if side["path_labels_version"] != PATH_LABELS_VERSION or int(side["num_nodes"]) != int(data.num_nodes):
            raise ValueError(f"{p}: version {side['path_labels_version']} / {side['num_nodes']} pins, "
                             f"expected {PATH_LABELS_VERSION} / {int(data.num_nodes)}")
        for k in PATH_LABEL_KEYS:
            setattr(data, k, side[k])
        data.label_stage = stage
    return data


def build_sidecar(gd: str, root: str | None = None, dump_root: str | None = None, corner: str = "BC",
                  out_path: str | None = None) -> dict:
    from encoder.forward_pass import clock_network
    root = root or graph_root(corner)
    dump_root = dump_root or f"{DATA}/sta_dense_{corner}"
    t0 = time.time()
    data = torch.load(os.path.join(root, gd, "graph.pt"), weights_only=False)
    dump = os.path.join(dump_root, gd, "pins.tsv.gz")
    names, kind, direction, V = read_dense(dump)
    N = int(data.num_nodes)
    import pandas as pd
    index = pd.Index(names)
    if not index.is_unique:
        raise ValueError(f"{gd}: duplicate rows in {dump}")
    perm = index.get_indexer(list(data.pin_names))
    if (perm < 0).any() or len(names) != N:
        raise ValueError(f"{gd}: {int((perm < 0).sum())} pins without a row, {len(names) - N + int((perm < 0).sum())} rows without a pin")
    role = data.x_cat[:, 0].numpy()
    want_kind = np.where(np.isin(role, (0, 1)), "port", "pin")
    want_dir = np.array([ROLE_DIR[int(r)] for r in role])
    if (kind[perm] != want_kind).any() or (direction[perm] != want_dir).any():
        raise ValueError(f"{gd}: kind/direction disagree with the pin role")
    D = V[perm]
    clk = clock_network(data)
    cone = f1_cone(data, clk)
    f3 = f3_pins(data)
    y_arr, y_slew, y_slack = D[:, 0:2], D[:, 2:4], D[:, 5:7]
    m_arr = ~np.isnan(y_arr) & ~cone[:, None]
    m_slew = ~np.isnan(y_slew) & ~cone[:, None]
    m_slack = ~np.isnan(y_slack) & ~f3[:, None]
    is_const = data.x_num[:, 7].numpy() > 0
    arr_na = np.isnan(y_arr).any(axis=1)
    with open(dump, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    f32 = lambda x: torch.tensor(x, dtype=torch.float32)
    side = {"sidecar_version": SIDECAR_VERSION, "design": gd, "source": dump, "source_md5": md5, "num_nodes": N,
            "y_arrival": f32(y_arr), "y_slew": f32(y_slew), "y_slack_rf": f32(y_slack),
            "y_mask_arrival": torch.tensor(m_arr), "y_mask_slew": torch.tensor(m_slew), "y_mask_slack": torch.tensor(m_slack),
            "f1_cone": torch.tensor(cone), "f3_pin": torch.tensor(f3), "clk_arr": torch.tensor(clk, dtype=torch.float64)}
    out = out_path or sidecar_path(root, gd)
    tmp = out + f".tmp{os.getpid()}"
    torch.save(side, tmp)
    os.replace(tmp, out)
    return {"design": gd, "nodes": N, "levels": int(data.level.max()) + 1,
            "arcs": int(data.edge_index_arc.shape[1]), "inst": int(data.arc_rf_mask.sum()),
            "endpoints": int(len(data.y_ep_index)),
            "arr_labels": int(m_arr.sum()), "slew_labels": int(m_slew.sum()), "slack_labels": int(m_slack.sum()),
            "arr_na_pins": int(arr_na.sum()), "constant_pins": int(is_const.sum()),
            "arr_na_not_constant_outside_f1": int((arr_na & ~is_const & ~cone).sum()),
            "constant_with_arr": int((~arr_na & is_const).sum()),
            "f1_cone_pins": int(cone.sum()), "f3_pins": int(f3.sum()),
            "clock_network_pins": int((~np.isnan(clk[:, 0])).sum()), "seconds": round(time.time() - t0, 2), "error": ""}


@dataclass
class Design:
    name: str
    num_nodes: int
    levels: int
    n_inst: int
    period: float
    x_cat: torch.Tensor
    x_num: torch.Tensor
    level: torch.Tensor
    clk_arr: torch.Tensor
    net: torch.Tensor
    arc: torch.Tensor
    arc_key: torch.Tensor
    arc_mask: torch.Tensor
    arc_type: torch.Tensor
    arc_sense: torch.Tensor
    arc_enabled: torch.Tensor
    chk: torch.Tensor
    chk_tables: torch.Tensor
    chk_axes: torch.Tensor
    ep_index: torch.Tensor
    ep_slack: torch.Tensor
    y_arrival: torch.Tensor | None = None
    y_slew: torch.Tensor | None = None
    y_slack_rf: torch.Tensor | None = None
    m_arr: torch.Tensor | None = None
    m_slew: torch.Tensor | None = None
    m_slack: torch.Tensor | None = None


def design_from_graph(data, side: dict | None = None, name: str | None = None) -> Design:
    from encoder.forward_pass import clock_network
    N = int(data.num_nodes)
    if side is not None and int(side["num_nodes"]) != N:
        raise ValueError(f"{name}: sidecar has {side['num_nodes']} pins, graph {N}")
    clk = side["clk_arr"] if side is not None else torch.tensor(clock_network(data), dtype=torch.float64)
    lab = {}
    if side is not None:
        lab = dict(y_arrival=side["y_arrival"], y_slew=side["y_slew"], y_slack_rf=side["y_slack_rf"],
                   m_arr=side["y_mask_arrival"], m_slew=side["y_mask_slew"], m_slack=side["y_mask_slack"])
    chk_axes = data.check_axes.float()
    if data.edge_index_check.shape[1] == 0:
        chk_axes = chk_axes.reshape(0, 2, 2, 7)
    return Design(name=name or f"{data.group}/{data.design}", num_nodes=N, levels=int(data.level.max()) + 1,
                  n_inst=int(data.arc_rf_mask.sum()), period=float(data.g[0, 0]),
                  x_cat=data.x_cat.long(), x_num=data.x_num.float(), level=data.level.long(), clk_arr=clk,
                  net=data.edge_index_net.long(), arc=data.edge_index_arc.long(), arc_key=data.arc_key_id.long(),
                  arc_mask=data.arc_rf_mask.bool(), arc_type=data.arc_type.long(), arc_sense=data.arc_sense.long(),
                  arc_enabled=data.arc_enabled.bool(), chk=data.edge_index_check.long(),
                  chk_tables=data.check_tables.float(), chk_axes=chk_axes,
                  ep_index=data.y_ep_index.long(), ep_slack=data.y_ep_slack.float(), **lab)


def load_design(gd: str, root: str | None = None, corner: str = "BC", labels: bool = True, stage: str = "synth") -> Design:
    root = root or graph_root(corner)
    data = load_graph(gd, root, corner, stage if labels else "synth")
    side = None
    if labels:
        p = sidecar_path(root, gd, stage)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}: build it with  python -m model.data build --stage {stage} --designs {gd}")
        side = torch.load(p, weights_only=True)
        if side["sidecar_version"] != SIDECAR_VERSION:
            raise ValueError(f"{p}: sidecar version {side['sidecar_version']}, expected {SIDECAR_VERSION}")
    return design_from_graph(data, side, gd)


def load_designs(names: list, root: str | None = None, corner: str = "BC", labels: bool = True, workers: int = 8,
                 log_every: int = 500, stage: str = "synth") -> list:
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    out = [None] * len(names)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, d in enumerate(ex.map(lambda n: load_design(n, root, corner, labels, stage), names)):
            out[i] = d
            if log_every and (i + 1) % log_every == 0:
                print(f"  loaded {i + 1}/{len(names)} designs, {time.time() - t0:.0f}s", flush=True)
    return out


def shuffle_labels(d: Design, seed: int = 0) -> Design:
    g = torch.Generator().manual_seed(zlib.crc32(f"{d.name}/{seed}".encode()))
    p = torch.randperm(d.num_nodes, generator=g)
    q = torch.randperm(len(d.ep_index), generator=g)
    return replace(d, y_arrival=d.y_arrival[p], y_slew=d.y_slew[p], y_slack_rf=d.y_slack_rf[p],
                   m_arr=d.m_arr[p], m_slew=d.m_slew[p], m_slack=d.m_slack[p], ep_slack=d.ep_slack[q])


@dataclass
class Batch:
    names: list
    num_nodes: int
    node_ptr: torch.Tensor
    node_graph: torch.Tensor
    period: torch.Tensor
    x_cat: torch.Tensor
    x_num: torch.Tensor
    level: torch.Tensor
    clk_arr: torch.Tensor
    net: torch.Tensor
    arc: torch.Tensor
    arc_key: torch.Tensor
    arc_mask: torch.Tensor
    arc_type: torch.Tensor
    arc_sense: torch.Tensor
    arc_enabled: torch.Tensor
    chk: torch.Tensor
    chk_tables: torch.Tensor
    chk_axes: torch.Tensor
    ep_index: torch.Tensor
    ep_slack: torch.Tensor
    ep_graph: torch.Tensor
    arc_ptr: torch.Tensor
    y_arrival: torch.Tensor | None = None
    y_slew: torch.Tensor | None = None
    y_slack_rf: torch.Tensor | None = None
    m_arr: torch.Tensor | None = None
    m_slew: torch.Tensor | None = None
    m_slack: torch.Tensor | None = None
    knockout_tables: bool = False

    def to(self, device) -> "Batch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return Batch(**kw)

    @property
    def levels(self) -> int:
        return int(self.level.max()) + 1


def collate(designs: list, knockout_tables: bool = False) -> Batch:
    off = np.cumsum([0] + [d.num_nodes for d in designs])
    aoff = np.cumsum([0] + [d.arc.shape[1] for d in designs])
    cat = lambda k: torch.cat([getattr(d, k) for d in designs])
    shift = lambda k: torch.cat([getattr(d, k) + int(o) for d, o in zip(designs, off[:-1])], dim=-1)
    x_num = cat("x_num")
    if knockout_tables:
        x_num = x_num.clone()
        x_num[:, 0:4] = 0.0
    has = designs[0].y_arrival is not None and all(d.y_arrival is not None for d in designs)
    lab = {}
    if has:
        lab = {k: cat(k) for k in ("y_arrival", "y_slew", "y_slack_rf", "m_arr", "m_slew", "m_slack")}
    return Batch(names=[d.name for d in designs], num_nodes=int(off[-1]),
                 node_ptr=torch.tensor(off, dtype=torch.long),
                 node_graph=torch.repeat_interleave(torch.arange(len(designs)), torch.tensor([d.num_nodes for d in designs])),
                 period=torch.tensor([d.period for d in designs], dtype=torch.float64),
                 x_cat=cat("x_cat"), x_num=x_num, level=cat("level"), clk_arr=cat("clk_arr"),
                 net=shift("net"), arc=shift("arc"), arc_key=cat("arc_key"), arc_mask=cat("arc_mask"),
                 arc_type=cat("arc_type"), arc_sense=cat("arc_sense"), arc_enabled=cat("arc_enabled"),
                 chk=shift("chk"), chk_tables=cat("chk_tables"), chk_axes=cat("chk_axes"),
                 ep_index=shift("ep_index"), ep_slack=cat("ep_slack"),
                 ep_graph=torch.repeat_interleave(torch.arange(len(designs)), torch.tensor([len(d.ep_index) for d in designs])),
                 arc_ptr=torch.tensor(aoff, dtype=torch.long), knockout_tables=knockout_tables, **lab)


class DepthBatcher:

    def __init__(self, levels: list, n_inst: list, max_inst: int = 300_000, max_designs: int = 64,
                 deep_levels: int = 256, depth_ratio: float = 1.5, seed: int = 0):
        self.levels, self.n_inst = np.asarray(levels), np.asarray(n_inst)
        self.max_inst, self.max_designs, self.deep_levels, self.depth_ratio = max_inst, max_designs, deep_levels, depth_ratio
        self.rng = np.random.default_rng(seed)

    def epoch(self) -> list:
        n = len(self.levels)
        order = np.lexsort((self.rng.random(n), self.levels))
        batches, cur, inst = [], [], 0
        for i in order.tolist():
            alone = self.levels[i] > self.deep_levels or self.n_inst[i] >= self.max_inst
            if cur and (alone or inst + self.n_inst[i] > self.max_inst or len(cur) >= self.max_designs
                        or self.levels[i] > self.depth_ratio * self.levels[cur[0]]):
                batches.append(cur); cur, inst = [], 0
            cur.append(i); inst += int(self.n_inst[i])
            if alone:
                batches.append(cur); cur, inst = [], 0
        if cur:
            batches.append(cur)
        self.rng.shuffle(batches)
        return batches

    def cost(self, batches: list) -> int:
        return int(sum(self.levels[b].max() for b in batches))


_BUILD = {}


def _build_one(gd):
    try:
        if _BUILD.get("command") == "build-path":
            return build_path_labels(gd, _BUILD["stage"], _BUILD["root"], _BUILD["corner"])
        out = sidecar_path(_BUILD["root"], gd, _BUILD.get("stage", "synth"))
        return build_sidecar(gd, _BUILD["root"], _BUILD["dump_root"], _BUILD["corner"], out_path=out)
    except Exception as e:
        return {"design": gd, "stage": _BUILD.get("stage", "synth"), "error": f"{type(e).__name__}: {e}"[:300]}


def _init_build(cfg):
    torch.set_num_threads(1)
    _BUILD.update(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "build-path", "info"))
    ap.add_argument("--stage", default="synth", choices=LABEL_STAGES)
    ap.add_argument("--corner", default="BC")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--val-only", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    root = graph_root(a.corner)
    sf = a.split_file or default_split_file(a.corner)
    if a.designs:
        names = list(a.designs)
    else:
        sel = select_designs(sf, a.splits, a.val_only, a.allow_test)
        names = [d for s in a.splits for d in sel[s]]

    if a.command == "info":
        s = read_split(sf)
        print(f"{sf}: " + ", ".join(f"{k} {len(v)}" for k, v in s.items()))
        import pandas as pd
        bs = pd.read_csv(f"{root}/build_summary.tsv", sep="\t")
        bs = bs[bs.status == "ok"]
        sh = pd.read_csv(f"{root}/shape_table_bench.tsv", sep="\t")
        lv = {f"{r.group}/{r.design}": int(r.levels) for r in bs.itertuples()}
        lv.update({f"{r.group}/{r.design}": int(r.levels) for r in sh.itertuples()})
        inst = {f"{r.group}/{r.design}": int(r.arc_edges) for r in bs.itertuples()}
        inst.update({f"{r.group}/{r.design}": int(r.arc_edges) for r in sh.itertuples()})
        for split in a.splits:
            ds = select_designs(sf, [split], a.val_only, a.allow_test)[split] if not a.designs else names
            L = [lv[d] for d in ds]
            I = [inst[d] * 2 for d in ds]
            b = DepthBatcher(L, I)
            costs = [b.cost(b.epoch()) for _ in range(5)]
            print(f"{split}: {len(ds)} designs, levels median {np.median(L):.0f} max {max(L)}; "
                  f"sequential level-steps per epoch (batched, 5 draws) {min(costs)}-{max(costs)}; "
                  f"deepest: {sorted(zip(L, ds))[-3:]}")
        return 0

    if a.command == "build-path":
        if a.stage == "synth":
            ap.error("build-path needs --stage place or groute (synth path labels are graph.pt's own)")
        out = a.summary or f"{root}/path_labels_{a.stage}_summary.tsv"
        cfg = {"root": root, "corner": a.corner, "stage": a.stage, "command": "build-path"}
        keys = ["design", "stage", "endpoints", "path_len", "worst_slack_synth", "worst_slack", "same_worst_path_as_synth",
                "tied_worst", "bad_steps", "seconds", "error"]
        t0 = time.time()
        rows = []
        with Pool(a.workers, initializer=_init_build, initargs=(cfg,)) as pool, open(out, "w") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(pool.imap_unordered(_build_one, names, chunksize=4)):
                rows.append(r); w.writerow(r); f.flush()
                if r.get("error"):
                    print(f"ERROR {r['design']}: {r['error']}", flush=True)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(names)}, {time.time() - t0:.0f}s", flush=True)
        ok = [r for r in rows if not r.get("error")]
        print(f"path labels ({a.stage}) written: {len(ok)}/{len(names)} in {time.time() - t0:.0f}s -> {out}")
        print(f"  endpoints {sum(r['endpoints'] for r in ok)}; worst path same as synth on "
              f"{sum(r['same_worst_path_as_synth'] for r in ok)}/{len(ok)} designs")
        return 0 if len(ok) == len(names) else 1

    out = a.summary or (f"{root}/dense_labels_summary.tsv" if a.stage == "synth" else f"{root}/dense_labels_{a.stage}_summary.tsv")
    dump_root = f"{DATA}/sta_dense_{a.corner}" if a.stage == "synth" else f"{post_place_root(a.corner)}/{a.stage}"
    cfg = {"root": root, "dump_root": dump_root, "corner": a.corner, "stage": a.stage, "command": "build"}
    keys = ["design", "nodes", "levels", "arcs", "inst", "endpoints", "arr_labels", "slew_labels", "slack_labels",
            "arr_na_pins", "constant_pins", "arr_na_not_constant_outside_f1", "constant_with_arr", "f1_cone_pins",
            "f3_pins", "clock_network_pins", "seconds", "error"]
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init_build, initargs=(cfg,)) as pool, open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(pool.imap_unordered(_build_one, names, chunksize=4)):
            rows.append(r); w.writerow(r); f.flush()
            if r.get("error"):
                print(f"ERROR {r['design']}: {r['error']}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(names)}, {time.time() - t0:.0f}s", flush=True)
    ok = [r for r in rows if not r.get("error")]
    tot = lambda k: sum(r[k] for r in ok)
    print(f"sidecars written: {len(ok)}/{len(names)} in {time.time() - t0:.0f}s -> {out}")
    print(f"  pins {tot('nodes')}; labels arrival {tot('arr_labels')} slew {tot('slew_labels')} slack {tot('slack_labels')} (pin transitions)")
    print(f"  arrival NA pins {tot('arr_na_pins')} = constant pins {tot('constant_pins')}? "
          f"NA-not-constant outside F1 {tot('arr_na_not_constant_outside_f1')}, constant-with-arrival {tot('constant_with_arr')}")
    print(f"  F1 cone pins {tot('f1_cone_pins')} in {sum(r['f1_cone_pins'] > 0 for r in ok)} designs; F3 pins {tot('f3_pins')}")
    return 0 if len(ok) == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
