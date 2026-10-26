from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import torch

from paths import DATA

N_AXIS = 7
G_MIN, G_MAX = -1.0, 6.5


def family(cell: str) -> str:
    return re.sub(r"x[p\d]+.*$", "", re.sub(r"_ASAP7_75t_\w+$", "", cell))


def split_families(families: list, used: set, n_val: int, n_test: int, seed: int) -> dict:
    cand = sorted(f for f in set(families) if f in used)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    out = {f: "train" for f in set(families)}
    for f in cand[:n_test]:
        out[f] = "test"
    for f in cand[n_test:n_test + n_val]:
        out[f] = "val"
    return out


def grid_coord(x: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    r = x / a0
    i = torch.clamp(torch.floor(torch.log2(torch.clamp(r, min=1.0))), 0, N_AXIS - 2)
    return torch.where(r < 1.0, r - 1.0, i + r / torch.pow(2.0, i) - 1.0)


def from_grid_coord(g: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    i = torch.clamp(torch.floor(g), 0, N_AXIS - 2)
    return torch.where(g < 0.0, (g + 1.0) * a0, (g - i + 1.0) * torch.pow(2.0, i) * a0)


def exact_lookup(table: torch.Tensor, axis1: torch.Tensor, axis2: torch.Tensor, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    B = table.shape[0]
    idx = torch.arange(B, device=table.device)
    i = torch.clamp((axis1 <= s[:, None]).sum(-1) - 1, 0, N_AXIS - 2)
    j = torch.clamp((axis2 <= c[:, None]).sum(-1) - 1, 0, N_AXIS - 2)
    u = (s - axis1[idx, i]) / (axis1[idx, i + 1] - axis1[idx, i])
    v = (c - axis2[idx, j]) / (axis2[idx, j + 1] - axis2[idx, j])
    t00, t01, t10, t11 = table[idx, i, j], table[idx, i, j + 1], table[idx, i + 1, j], table[idx, i + 1, j + 1]
    return (1 - u) * ((1 - v) * t00 + v * t01) + u * ((1 - v) * t10 + v * t11)


@dataclass
class Bank:
    tables: torch.Tensor
    cap_axis: torch.Tensor
    slew_axis: torch.Tensor
    present: torch.Tensor
    key_split: list
    key_family: list
    family_split: dict
    meta: list

    def pairs(self, split: str) -> torch.Tensor:
        ks = torch.tensor([s == split for s in self.key_split], device=self.tables.device)
        out = []
        for trf in (0, 1):
            ok = ks & self.present[:, trf] & self.present[:, 2 + trf]
            k = torch.nonzero(ok).flatten()
            out.append(torch.stack([k, torch.full_like(k, trf)], 1))
        return torch.cat(out)

    def gather(self, key: torch.Tensor, trf: torch.Tensor):
        return torch.stack([self.tables[key, trf], self.tables[key, 2 + trf]], 1)

    def targets(self, key, trf, s, c) -> torch.Tensor:
        t = self.gather(key, trf)
        ax1 = self.slew_axis.expand(len(key), N_AXIS)
        ax2 = self.cap_axis[key]
        return torch.stack([exact_lookup(t[:, 0], ax1, ax2, s, c), exact_lookup(t[:, 1], ax1, ax2, s, c)], 1)


def load_bank(used_keys, graph_root: str | None = None, corner: str = "BC", n_val: int = 6, n_test: int = 6,
              seed: int = 0, device="cpu") -> Bank:
    from graph.vocab import load_vocab
    root = graph_root or f"{DATA}/graphs_{corner}"
    v = load_vocab(root, corner)
    ratio = v.arc_cap_axis[:, 1:] / v.arc_cap_axis[:, :-1]
    if not (np.allclose(ratio, 2.0, atol=1e-5) and np.allclose(v.slew_axis[1:] / v.slew_axis[:-1], 2.0)):
        raise ValueError("an axis does not double from point to point: the grid coordinate of data.py does not apply")
    fams = [family(m[0]) for m in v.arc_key_meta]
    used = {fams[int(k)] for k in used_keys}
    fsplit = split_families(fams, used, n_val, n_test, seed)
    f64 = dict(dtype=torch.float64, device=device)
    return Bank(torch.tensor(v.arc_bank, **f64), torch.tensor(v.arc_cap_axis, **f64), torch.tensor(v.slew_axis, **f64),
                torch.tensor(v.arc_table_present, device=device), [fsplit[f] for f in fams], fams, fsplit, v.arc_key_meta)


def table_norm(t: torch.Tensor):
    off = t[:, :, 0, 0]
    scale = t[:, :, :4, :4].reshape(t.shape[0], 2, -1).std(-1).clamp_min(1e-3)
    return (t - off[:, :, None, None]) / scale[:, :, None, None], off, scale


class Sampler:

    def __init__(self, bank: Bank, split: str, queries: torch.Tensor | None, real_fraction: float = 0.5, seed: int = 0):
        self.bank, self.real_fraction = bank, real_fraction
        self.pairs = bank.pairs(split)
        self.gen = torch.Generator(device=bank.tables.device).manual_seed(seed)
        self.queries = None
        if queries is not None and len(queries):
            keep = torch.tensor([bank.key_split[int(k)] == split for k in queries[:, 0].tolist()], device=queries.device)
            self.queries = queries[keep]

    def synthetic(self, n: int):
        dev = self.bank.tables.device
        p = self.pairs[torch.randint(len(self.pairs), (n,), generator=self.gen, device=dev)]
        g = torch.rand((n, 2), generator=self.gen, device=dev, dtype=torch.float64) * (G_MAX - G_MIN) + G_MIN
        s = from_grid_coord(g[:, 0], self.bank.slew_axis[0].expand(n))
        c = from_grid_coord(g[:, 1], self.bank.cap_axis[p[:, 0], 0])
        return p[:, 0], p[:, 1], s, c

    def real(self, n: int, jitter: float = 0.1):
        dev = self.bank.tables.device
        q = self.queries[torch.randint(len(self.queries), (n,), generator=self.gen, device=dev)]
        f = 1.0 + jitter * (2 * torch.rand((n, 2), generator=self.gen, device=dev, dtype=torch.float64) - 1)
        return q[:, 0].long(), q[:, 1].long(), q[:, 2] * f[:, 0], q[:, 3] * f[:, 1]

    def batch(self, n: int):
        n_real = int(n * self.real_fraction) if self.queries is not None and len(self.queries) else 0
        parts = [self.synthetic(n - n_real)] + ([self.real(n_real)] if n_real else [])
        return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


def featurise(bank: Bank, key, trf, s, c) -> dict:
    t = bank.gather(key, trf)
    tn, off, scale = table_norm(t)
    y = bank.targets(key, trf, s, c)
    a0 = bank.cap_axis[key, 0]
    g = torch.stack([grid_coord(s, bank.slew_axis[0].expand(len(s))), grid_coord(c, a0)], 1)
    f = lambda x: x.float()
    return {"g": f(g), "tables_norm": f(tn), "offset": f(off), "scale": f(scale), "y": f(y), "y_norm": f((y - off) / scale),
            "tables_raw": f(t), "s": f(s), "c": f(c), "cap0": f(a0)}
