from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv, GINEConv

from cell_model.model import mlp, squash

from .arc_embedding import ArcEmbedding
from .encoder import NEG_INF, Schedule, build_schedule, instance_level_offset
from .heads import endpoint_slack

CONVS = ("gatv2", "gine", "gps")
GLOBAL = ("none", "linear", "multihead")
N_EDGE_TYPES = 4


def level_pe(level: torch.Tensor, dim: int) -> torch.Tensor:
    pos = level.float()[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=level.device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros((len(level), dim), device=level.device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe


@torch.no_grad()
def reach_mask(b, sch: Schedule) -> torch.Tensor:
    N, dev = b.num_nodes, b.level.device
    role, xn = b.x_cat[:, 0], b.x_num
    const = xn[:, 7] > 0
    R = ((role == 0) & ~const).repeat_interleave(2).long()
    clk_ok = ~torch.isnan(b.clk_arr).reshape(-1)
    for l in range(1, sch.L):
        c0, c1 = sch.neta_ptr[l], sch.neta_ptr[l + 1]
        if c1 > c0:
            R[sch.neta_fs[c0:c1]] = R[sch.neta_fd[c0:c1]]
        i0, i1 = sch.inst_ptr[l], sch.inst_ptr[l + 1]
        if i1 == i0:
            continue
        fu, ft, launch = sch.inst_fu[i0:i1], sch.inst_ft[i0:i1], sch.inst_launch[i0:i1]
        src_ok = torch.where(launch, clk_ok[fu], R[fu] > 0)
        R.index_add_(0, ft, (sch.inst_search[i0:i1] & src_ok).long())
    return R > 0


class LinearAttention(nn.Module):

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, ptr) -> torch.Tensor:
        n, H, dh = x.shape[0], self.heads, self.dh
        q, k, v = self.qkv(x).view(n, 3, H, dh).unbind(1)
        q, k = F.relu(q), F.relu(k)
        outs = []
        for lo, hi in zip(ptr[:-1], ptr[1:]):
            qg, kg, vg = q[lo:hi].transpose(0, 1), k[lo:hi].transpose(0, 1), v[lo:hi].transpose(0, 1)
            kv = kg.transpose(1, 2) @ vg
            den = (qg * kg.sum(1, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
            outs.append(((qg @ kv) / den).transpose(0, 1).reshape(hi - lo, H * dh))
        return self.o(torch.cat(outs) if outs else x.new_zeros(0, H * dh))


class FullAttention(nn.Module):

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, ptr) -> torch.Tensor:
        n, H, dh = x.shape[0], self.heads, self.dh
        q, k, v = self.qkv(x).view(n, 3, H, dh).unbind(1)
        outs = []
        for lo, hi in zip(ptr[:-1], ptr[1:]):
            a = F.scaled_dot_product_attention(q[lo:hi].transpose(0, 1)[None], k[lo:hi].transpose(0, 1)[None], v[lo:hi].transpose(0, 1)[None])
            outs.append(a[0].transpose(0, 1).reshape(hi - lo, H * dh))
        return self.o(torch.cat(outs) if outs else x.new_zeros(0, H * dh))


class GPSLayer(nn.Module):

    def __init__(self, d: int, d_edge: int, heads: int, global_attn: str):
        super().__init__()
        self.local = GINEConv(mlp(d, d, 2 * d, 1, nn.ReLU), edge_dim=d_edge, train_eps=True)
        self.attn = {"linear": LinearAttention, "multihead": FullAttention, "none": lambda d, h: None}[global_attn](d, heads)
        self.norm1, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d) if self.attn is not None else None
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x, ei, ea, ptr):
        h = self.norm1(x + self.local(x, ei, ea))
        if self.attn is not None:
            h = h + self.norm2(x + self.attn(x, ptr))
        return self.norm3(h + self.mlp(h))


class StoredMessage(nn.Module):
    exact = False
    dz = 0

    def __init__(self, arc_ref: list):
        super().__init__()
        self._arc = arc_ref
        self.delay = self.slew = None

    def init_z(self, b):
        return None

    def prepare(self, b, sch, z0) -> dict:
        if self.delay is None or len(self.delay) != len(sch.inst_e):
            raise RuntimeError("StoredMessage: set delay and slew for every arc instance of the schedule first")
        return {}

    def cell(self, st, lo, hi, s_in, z_in):
        return self.delay[lo:hi], self.slew[lo:hi], None

    def setup(self, tables, axes, s_d, knockout: bool = False):
        return self._arc[0].setup(tables, axes, s_d, knockout=knockout)


class LayeredEncoder(nn.Module):
    exact = False

    def __init__(self, arc: ArcEmbedding, n_pin_names: int, conv: str = "gatv2", d_model: int = 128, d_edge: int = 40,
                 layers: int = 8, heads: int = 4, level_pe: int = 0, global_attn: str = "none",
                 checkpoint_layers: bool = False, out_scale: float = 10.0, d_arc: int = 32,
                 arrival_from_delays: bool = False, delay_norm: str = "scale"):
        super().__init__()
        if delay_norm not in ("scale", "stats"):
            raise ValueError(f"delay_norm {delay_norm!r} not in ('scale', 'stats')")
        if conv not in CONVS or global_attn not in GLOBAL:
            raise ValueError(f"conv {conv!r} / global_attn {global_attn!r}")
        if d_model % 2 or d_model % heads or (level_pe and level_pe % 2):
            raise ValueError("d_model must be even and a multiple of heads; level_pe even")
        self._arc = [arc]
        self.conv, self.d, self.dz, self.n_layers, self.level_pe = conv, d_model, d_model // 2, layers, level_pe
        self.checkpoint_layers, self.out_scale, self.global_attn = checkpoint_layers, out_scale, global_attn
        self.emb_role, self.emb_pin = nn.Embedding(7, 8), nn.Embedding(n_pin_names, 8)
        self.node = mlp(8 + 8 + 9 + 4 + level_pe, d_model, d_model, 1)
        self.ln_node = nn.LayerNorm(d_model)
        n_feat = (2 * 2 * 7 + 4 + 4 + 2 if arc.mode == "tables" else 2 * d_arc) + 4 + 4 + 4 + 1
        self.emb_type, self.emb_sense, self.emb_pair = nn.Embedding(4, 4), nn.Embedding(3, 4), nn.Embedding(4, 4)
        self.edge = nn.Linear(n_feat, d_edge)
        self.emb_edge_type = nn.Embedding(N_EDGE_TYPES, d_edge)
        if conv == "gatv2":
            self.layers = nn.ModuleList([GATv2Conv(d_model, d_model // heads, heads=heads, concat=True, edge_dim=d_edge,
                                                   add_self_loops=False) for _ in range(layers)])
        elif conv == "gine":
            self.layers = nn.ModuleList([GINEConv(mlp(d_model, d_model, 2 * d_model, 1, nn.ReLU), edge_dim=d_edge, train_eps=True)
                                         for _ in range(layers)])
        else:
            self.layers = nn.ModuleList([GPSLayer(d_model, d_edge, heads, global_attn) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)]) if conv != "gps" else None
        self.arrival_from_delays, self.delay_norm = arrival_from_delays, delay_norm
        if not arrival_from_delays:
            self.pin_head = mlp(d_model, 4, d_model, 1)
        self.arc_hidden = mlp(2 * d_model + d_edge + 4, self.dz, d_model, 1)
        self.arc_out = nn.Linear(self.dz, 2)
        if delay_norm == "stats":
            self.register_buffer("arc_mu", torch.zeros(2))
            self.register_buffer("arc_sd", torch.full((2,), out_scale))
        if arrival_from_delays:
            from .encoder import LevelEncoder
            self._pass = [LevelEncoder(StoredMessage(self._arc))]
        self.monitor = False
        self._stats = {}

    @property
    def arc(self) -> ArcEmbedding:
        return self._arc[0]

    @torch.no_grad()
    def set_arc_stats(self, mu: torch.Tensor, sd: torch.Tensor):
        if self.delay_norm != "stats":
            raise ValueError("set_arc_stats needs delay_norm='stats'")
        self.arc_mu.copy_(mu); self.arc_sd.copy_(sd.clamp_min(1e-3))

    def _acc(self, name: str, value, n=1):
        s, c = self._stats.get(name, (0.0, 0.0))
        self._stats[name] = (s + (value.detach() if torch.is_tensor(value) else value), c + n)

    def pop_stats(self) -> dict:
        out = {k: (float(s) / c if c else float("nan")) for k, (s, c) in self._stats.items()}
        self._stats = {}
        return out

    def node_features(self, b) -> torch.Tensor:
        period = b.period.float()[b.node_graph]
        clk = b.clk_arr.float()
        fin = torch.isfinite(clk)
        clk_f = torch.cat([fin.float(), torch.where(fin, clk / period[:, None], torch.zeros_like(clk))], 1)
        parts = [self.emb_role(b.x_cat[:, 0]), self.emb_pin(b.x_cat[:, 2]), squash(b.x_num.float()), clk_f]
        if self.level_pe:
            parts.append(level_pe(b.level, self.level_pe))
        return self.ln_node(self.node(torch.cat(parts, 1)))

    def arc_features(self, b) -> torch.Tensor:
        arc = self.arc
        E = b.arc.shape[1]
        key, o = b.arc_key, b.arc[1]
        load = b.x_num[:, 2:4].float()
        cols = []
        for trf in (0, 1):
            t = torch.full((E,), trf, dtype=torch.long, device=key.device)
            st = arc.static(key, t, load[o, trf], knockout=b.knockout_tables)
            if arc.mode == "tables":
                cols += [st["tv"].reshape(E, -1), squash(st["off"] / 10.0), torch.log(st["scale"]), st["gv"][:, None]]
            else:
                cols.append(arc.emb_key(st["key2"]))
        cols += [self.emb_type(b.arc_type), self.emb_sense(b.arc_sense), b.arc_mask.float(), b.arc_enabled.float()[:, None]]
        return self.edge(torch.cat(cols, 1))

    def edges(self, b, ea_arc: torch.Tensor):
        dev = ea_arc.device
        E_n = b.net.shape[1]
        ea_net = self.edge.bias.new_zeros((E_n, ea_arc.shape[1]))
        ei = torch.cat([b.arc, b.arc.flip(0), b.net, b.net.flip(0)], 1)
        typ = torch.repeat_interleave(torch.arange(N_EDGE_TYPES, device=dev), torch.tensor([ea_arc.shape[0], ea_arc.shape[0], E_n, E_n], device=dev))
        ea = torch.cat([ea_arc, ea_arc, ea_net, ea_net], 0) + self.emb_edge_type(typ)
        return ei, ea

    def run_layer(self, i: int, h, ei, ea, ptr):
        layer = self.layers[i]
        if self.conv == "gps":
            return layer(h, ei, ea, ptr)
        if self.conv == "gatv2" and self.monitor:
            out, (_, alpha) = layer(h, ei, ea, return_attention_weights=True)
            from .attn_message import entropy_stats
            ent, mx, multi = entropy_stats(alpha, ei[1], h.shape[0])
            self._acc(f"gat_entropy", ent.sum(), int(len(multi))); self._acc(f"gat_maxw", mx.sum(), int(len(multi)))
        else:
            out = layer(h, ei, ea)
        return self.norms[i](h + F.relu(out))

    def forward(self, b, sch: Schedule | None = None, keep: bool = True, tf_prob: float = 0.0, generator=None) -> dict:
        if tf_prob > 0 and not self.arrival_from_delays:
            raise ValueError("scheduled sampling needs the schedule: only with arrival_from_delays (layered arm)")
        sch = sch or build_schedule(b)
        N, dev = b.num_nodes, b.level.device
        ptr = b.node_ptr.tolist()
        h = self.node_features(b)
        ea_arc = self.arc_features(b)
        ei, ea = self.edges(b, ea_arc)
        for i in range(self.n_layers):
            if self.checkpoint_layers and torch.is_grad_enabled():
                h_new = checkpoint(self.run_layer, i, h, ei, ea, ptr, use_reentrant=False)
            else:
                h_new = self.run_layer(i, h, ei, ea, ptr)
            if self.monitor:
                self._acc("layer_std", h_new.std(0).mean()); self._acc("layer_update", (h_new - h).norm() / h.norm().clamp_min(1e-6))
            h = h_new
        e, pair, u, o, fu, ft = sch.inst_e, sch.inst_pair, sch.inst_u, sch.inst_o, sch.inst_fu, sch.inst_ft
        hid = self.arc_hidden(torch.cat([h[u], h[o], ea_arc[e], self.emb_pair(pair)], 1))
        raw = self.arc_out(hid)
        ds = raw * self.arc_sd + self.arc_mu if self.delay_norm == "stats" else raw * self.out_scale
        if self.arrival_from_delays:
            msg = self._pass[0].message
            msg.delay, msg.slew = ds[:, 0], ds[:, 1]
            try:
                out = self._pass[0](b, sch, keep=keep, tf_prob=tf_prob, generator=generator)
            finally:
                msg.delay = msg.slew = None
            out["z"] = h.view(N, 2, self.dz)
            out["reach"] = torch.isfinite(out["arr"])
            if keep:
                out["msg_z"] = hid[instance_level_offset(sch):]
            return out
        pins = self.pin_head(h) * self.out_scale
        reach = reach_mask(b, sch).view(N, 2)
        arr = torch.where(reach, pins[:, 0:2], torch.full_like(pins[:, 0:2], NEG_INF))
        slew = pins[:, 2:4]
        delay, slew_j = ds[:, 0], ds[:, 1]
        A = arr.reshape(-1)
        clkA = torch.nan_to_num(b.clk_arr.float(), nan=NEG_INF).reshape(-1)
        from_arr = torch.where(sch.inst_launch, clkA[fu], A[fu])
        cand = torch.where(sch.inst_search & (from_arr > NEG_INF), from_arr + delay, torch.full_like(delay, NEG_INF))
        out = {"arr": arr, "slew": slew, "z": h.view(N, 2, self.dz), "schedule": sch, "reach": reach}
        out.update(endpoint_slack(b, arr, slew, self.arc.setup, knockout=b.knockout_tables))
        if keep:
            k = instance_level_offset(sch)
            out.update(msg_delay=delay[k:], msg_slew=slew_j[k:], msg_arrival=cand[k:], msg_z=hid[k:])
        return out
