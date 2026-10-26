from __future__ import annotations

import torch
from torch import nn

from cell_model.data import N_AXIS, grid_coord, table_norm
from cell_model.model import mlp, squash

MODES = ("tables", "key")


def _sum_to_one(w: torch.Tensor) -> torch.Tensor:
    return w - w.mean(-1, keepdim=True) + 1.0 / w.shape[-1]


class ArcEmbedding(nn.Module):
    def __init__(self, arc_bank, cap_axis, slew_axis, mode: str = "tables", width: int = 512, depth: int = 4,
                 d_out: int = 32):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.mode, self.d_out = mode, d_out
        bank = torch.as_tensor(arc_bank, dtype=torch.float64)
        cap_axis = torch.as_tensor(cap_axis, dtype=torch.float64)
        slew_axis = torch.as_tensor(slew_axis, dtype=torch.float64)
        for ax in (cap_axis, slew_axis[None]):
            if not torch.allclose(ax[:, 1:] / ax[:, :-1], torch.full_like(ax[:, 1:], 2.0), atol=1e-5):
                raise ValueError("an axis does not double from point to point: the grid coordinate does not apply")
        K = bank.shape[0]
        t = torch.stack([torch.stack([bank[:, 0], bank[:, 2]], 1), torch.stack([bank[:, 1], bank[:, 3]], 1)], 1)
        tn, off, scale = table_norm(t.reshape(K * 2, 2, N_AXIS, N_AXIS))
        self.register_buffer("tn", tn.float().reshape(K, 2, 2, N_AXIS, N_AXIS), persistent=False)
        self.register_buffer("off", off.float().reshape(K, 2, 2), persistent=False)
        self.register_buffer("scale", scale.float().reshape(K, 2, 2), persistent=False)
        self.register_buffer("cap0", cap_axis[:, 0].float(), persistent=False)
        self.register_buffer("slew0", slew_axis[:1].float(), persistent=False)
        self.net_u = mlp(1, N_AXIS, width, depth, nn.ReLU)
        self.net_v = mlp(1, N_AXIS, width, depth, nn.ReLU)
        self.emb_type = nn.Embedding(4, 4)
        self.emb_sense = nn.Embedding(3, 4)
        self.emb_pair = nn.Embedding(4, 4)
        self.proj = nn.Linear(8 + 12, d_out)
        if mode == "key":
            self.emb_key = nn.Embedding(K * 2, d_out)

    def weights_u(self, gu: torch.Tensor) -> torch.Tensor:
        return _sum_to_one(self.net_u(gu[..., None]))

    def weights_v(self, gv: torch.Tensor) -> torch.Tensor:
        return _sum_to_one(self.net_v(gv[..., None]))

    def lookup(self, key, trf, s, c) -> torch.Tensor:
        st = self.static(key, trf, c)
        d, sl, _ = self.read(st, 0, len(key), s)
        return torch.stack([d, sl], 1)

    def static(self, key, trf, load, pair=None, arc_type=None, arc_sense=None, knockout: bool = False) -> dict:
        n = len(key)
        dev = self.tn.device
        pair = torch.zeros(n, dtype=torch.long, device=dev) if pair is None else pair
        arc_type = torch.zeros(n, dtype=torch.long, device=dev) if arc_type is None else arc_type
        arc_sense = torch.zeros(n, dtype=torch.long, device=dev) if arc_sense is None else arc_sense
        st = {"key2": key * 2 + trf,
              "cat": torch.cat([self.emb_type(arc_type), self.emb_sense(arc_sense), self.emb_pair(pair)], 1)}
        gv = grid_coord(load.float(), self.cap0[key])
        st["gv"] = gv
        if self.mode == "key":
            return st
        tn, off, scale = self.tn[key, trf], self.off[key, trf], self.scale[key, trf]
        if knockout:
            tn, off, scale = torch.zeros_like(tn), torch.zeros_like(off), torch.ones_like(scale)
        st.update(tv=(tn * self.weights_v(gv)[:, None, None, :]).sum(-1), off=off, scale=scale)
        return st

    def read(self, st: dict, lo: int, hi: int, s_in: torch.Tensor):
        gu = grid_coord(s_in.float(), self.slew0.expand(len(s_in)))
        gv = st["gv"][lo:hi]
        if self.mode == "key":
            e = self.emb_key(st["key2"][lo:hi])
            z = torch.zeros_like(gu)
            return z, z, e
        r = (st["tv"][lo:hi] * self.weights_u(gu)[:, None, :]).sum(-1)
        off, scale = st["off"][lo:hi], st["scale"][lo:hi]
        y = r * scale + off
        feats = torch.cat([r, squash(off / 10.0), torch.log(scale), gu[:, None], gv[:, None], st["cat"][lo:hi]], 1)
        return y[:, 0], y[:, 1], self.proj(feats)

    def setup(self, tables: torch.Tensor, axes: torch.Tensor, s_d: torch.Tensor, knockout: bool = False) -> torch.Tensor:
        tn, off, scale = table_norm(tables.float())
        if knockout:
            tn, off, scale = torch.zeros_like(tn), torch.zeros_like(off), torch.ones_like(scale)
        gu = grid_coord(s_d.float(), axes[:, :, 0, 0].float())
        gv = grid_coord(torch.zeros_like(gu), axes[:, :, 1, 0].float())
        w = self.weights_u(gu)[..., :, None] * self.weights_v(gv)[..., None, :]
        return (w * tn).sum((-1, -2)) * scale + off

    def load_step2(self, path: str) -> list:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("arm") != "kernel_sep" or ck.get("width") != self.net_u[0].out_features:
            raise ValueError(f"{path}: arm {ck.get('arm')} width {ck.get('width')}, need kernel_sep width {self.net_u[0].out_features}")
        sd = {k: v for k, v in ck["state_dict"].items() if k.startswith(("net_u.", "net_v."))}
        missing = self.load_state_dict(sd, strict=False)
        if missing.unexpected_keys or any(k.startswith(("net_u.", "net_v.")) for k in missing.missing_keys):
            raise ValueError(f"{path}: kernel networks do not match ({missing})")
        return sorted(sd)
