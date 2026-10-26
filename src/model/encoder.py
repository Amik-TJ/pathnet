from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cell_model.model import mlp, squash
from graph.vocab import RF_PAIRS

from .arc_embedding import ArcEmbedding
from .heads import endpoint_slack, exact_setup

NEG_INF = float("-inf")
_RF = torch.tensor(RF_PAIRS)


@dataclass
class Schedule:
    L: int
    net_fs: torch.Tensor; net_fd: torch.Tensor; net_ptr: list
    neta_fs: torch.Tensor; neta_fd: torch.Tensor; neta_ptr: list
    inst_e: torch.Tensor; inst_pair: torch.Tensor; inst_frf: torch.Tensor; inst_trf: torch.Tensor
    inst_u: torch.Tensor; inst_o: torch.Tensor; inst_fu: torch.Tensor; inst_ft: torch.Tensor
    inst_launch: torch.Tensor; inst_search: torch.Tensor; inst_ptr: list
    seg_ft: torch.Tensor; inst_seg: torch.Tensor; seg_ptr: list


def _ptr(lv: torch.Tensor, L: int) -> list:
    c = torch.bincount(lv, minlength=L)
    return [0] + torch.cumsum(c, 0).tolist()


def build_schedule(b) -> Schedule:
    level = b.level
    N, L = b.num_nodes, b.levels
    const = b.x_num[:, 7] > 0
    d, s = b.net
    order = torch.argsort(level[s], stable=True)
    d, s = d[order], s[order]
    fd = torch.stack([2 * d, 2 * d + 1], 1).reshape(-1)
    fs = torch.stack([2 * s, 2 * s + 1], 1).reshape(-1)
    lvf = level[s].repeat_interleave(2)
    ok = (~(const[s] | const[d])).repeat_interleave(2)
    e, pair = torch.nonzero(b.arc_mask, as_tuple=True)
    rf = _RF.to(e.device)[pair]
    frf, trf = rf[:, 0], rf[:, 1]
    u, o = b.arc[0, e], b.arc[1, e]
    ft, fu = 2 * o + trf, 2 * u + frf
    lv = level[o]
    order = torch.argsort(lv * (2 * N) + ft, stable=True)
    e, pair, frf, trf, u, o, ft, fu, lv = (x[order] for x in (e, pair, frf, trf, u, o, ft, fu, lv))
    seg_ft, inst_seg = torch.unique_consecutive(ft, return_inverse=True)
    return Schedule(L=L, net_fs=fs, net_fd=fd, net_ptr=_ptr(lvf, L),
                    neta_fs=fs[ok], neta_fd=fd[ok], neta_ptr=_ptr(lvf[ok], L),
                    inst_e=e, inst_pair=pair, inst_frf=frf, inst_trf=trf, inst_u=u, inst_o=o, inst_fu=fu, inst_ft=ft,
                    inst_launch=b.arc_type[e] == 1, inst_search=b.arc_enabled[e] & ~const[u] & ~const[o],
                    inst_ptr=_ptr(lv, L), seg_ft=seg_ft, inst_seg=inst_seg, seg_ptr=_ptr(level[seg_ft // 2], L))


class ExactMessage(nn.Module):
    exact = True
    dz = 0

    def __init__(self, bank):
        super().__init__()
        self.bank = bank

    def init_z(self, b):
        return None

    def prepare(self, b, sch: Schedule, z0) -> dict:
        load = b.x_num.double()[:, 2:4]
        return {"key": b.arc_key[sch.inst_e], "trf": sch.inst_trf, "load": load[sch.inst_o, sch.inst_trf]}

    def cell(self, st, lo, hi, s_in, z_in):
        y = self.bank.targets(st["key"][lo:hi], st["trf"][lo:hi], s_in, st["load"][lo:hi])
        return y[:, 0], y[:, 1], None

    def setup(self, tables, axes, s_d, knockout: bool = False):
        if knockout:
            raise ValueError("the exact message has no knockout")
        return exact_setup(tables, axes, s_d)


class LearnedMessage(nn.Module):
    exact = False

    def __init__(self, arc: ArcEmbedding, n_pin_names: int, dz: int = 64, width: int = 128, depth: int = 2,
                 correction: bool = True):
        super().__init__()
        self.arc, self.dz, self.correction = arc, dz, correction
        self.emb_role = nn.Embedding(7, 8)
        self.emb_pin = nn.Embedding(n_pin_names, 8)
        self.emb_rf = nn.Embedding(2, dz)
        self.node = mlp(8 + 8 + 9, dz, width, 1)
        self.ln0 = nn.LayerNorm(dz)
        self.ln_in = nn.LayerNorm(dz)
        self.msg = mlp(dz + dz + arc.d_out + 2, dz + 2, width, depth)
        self.ln_out = nn.LayerNorm(dz)
        self.wire_proj = nn.Linear(dz, dz)
        self.ln_wire = nn.LayerNorm(dz)
        self.corr_scale = 1.0 if arc.mode == "tables" else 10.0
        if arc.mode == "tables":
            with torch.no_grad():
                self.msg[-1].weight[dz:].zero_(); self.msg[-1].bias[dz:].zero_()

    def init_z(self, b) -> torch.Tensor:
        x = torch.cat([self.emb_role(b.x_cat[:, 0]), self.emb_pin(b.x_cat[:, 2]), squash(b.x_num.float())], 1)
        return self.ln0(self.node(x)[:, None, :] + self.emb_rf.weight[None])

    def prepare(self, b, sch: Schedule, z0) -> dict:
        load = b.x_num[:, 2:4].float()[sch.inst_o, sch.inst_trf]
        st = self.arc.static(b.arc_key[sch.inst_e], sch.inst_trf, load, sch.inst_pair, b.arc_type[sch.inst_e],
                             b.arc_sense[sch.inst_e], knockout=b.knockout_tables)
        st["z0_o"] = z0.reshape(-1, self.dz)[sch.inst_ft]
        return st

    def cell(self, st, lo, hi, s_in, z_in):
        d_hat, s_hat, e = self.arc.read(st, lo, hi, s_in)
        g = torch.stack([squash(s_in.float() / 10.0), st["gv"][lo:hi]], 1)
        o = self.msg(torch.cat([self.ln_in(z_in), st["z0_o"][lo:hi], e, g], 1))
        z = self.ln_out(o[:, :self.dz])
        if self.correction or self.arc.mode == "key":
            d_hat = d_hat + self.corr_scale * o[:, self.dz]
            s_hat = s_hat + self.corr_scale * o[:, self.dz + 1]
        return d_hat, s_hat, z

    def wire(self, z_driver, z0_sink, **kw):
        return self.ln_wire(z_driver + self.wire_proj(z0_sink))

    def setup(self, tables, axes, s_d, knockout: bool = False):
        return self.arc.setup(tables, axes, s_d, knockout=knockout)


class LevelEncoder(nn.Module):

    def __init__(self, message: nn.Module):
        super().__init__()
        self.message = message
        self.monitor = False
        self._stats = {}

    @property
    def exact(self) -> bool:
        return getattr(self.message, "exact", False)

    def pop_stats(self) -> dict:
        out = {}
        if "z_norm_sum" in self._stats:
            out["z_norm_mean"] = float(self._stats["z_norm_sum"]) / max(float(self._stats["z_n"]), 1.0)
            out["z_norm_max"] = float(self._stats["z_norm_max"])
        self._stats = {}
        return out

    def forward(self, b, sch: Schedule | None = None, keep: bool = True, tf_prob: float = 0.0,
                generator: torch.Generator | None = None) -> dict:
        msg = self.message
        dt = torch.float64 if self.exact else torch.float32
        sch = sch or build_schedule(b)
        N, dev = b.num_nodes, b.level.device
        role, xn = b.x_cat[:, 0], b.x_num.to(dt)
        const, clk_port = xn[:, 7] > 0, xn[:, 8] > 0
        period = b.period.to(dt)[b.node_graph]
        A = torch.full((N, 2), NEG_INF, dtype=dt, device=dev)
        seed = (role == 0) & ~const
        A = torch.where((seed & ~clk_port)[:, None], xn[:, 5:6].expand(N, 2), A)
        A = torch.where((seed & clk_port)[:, None], torch.stack([torch.zeros_like(period), period / 2.0], 1), A)
        A = A.reshape(-1).clone()
        S = torch.zeros(2 * N, dtype=dt, device=dev)
        clkA = torch.nan_to_num(b.clk_arr.to(dt), nan=NEG_INF).reshape(-1)
        z0 = msg.init_z(b)
        dz = 0 if z0 is None else z0.shape[-1]
        z0f = None if z0 is None else z0.reshape(-1, dz)
        Z = None if z0 is None else z0f.clone()
        st = msg.prepare(b, sch, z0)
        use = None
        if tf_prob > 0:
            m = (b.m_arr & b.m_slew).reshape(-1)
            use = (torch.rand(2 * N, device=dev, generator=generator) < tf_prob) & m
            yA, yS = b.y_arrival.reshape(-1).to(dt), b.y_slew.reshape(-1).to(dt)
        zero = torch.zeros((), dtype=dt, device=dev)
        pool_a, pool_z = getattr(msg, "pool_arrival", None), getattr(msg, "pool_z", None)
        keep_d, keep_s, keep_a, keep_z = [], [], [], []
        for l in range(1, sch.L):
            a0, a1 = sch.net_ptr[l], sch.net_ptr[l + 1]
            if a1 > a0:
                fs, fd = sch.net_fs[a0:a1], sch.net_fd[a0:a1]
                S[fs] = S[fd]
                if Z is not None:
                    Z[fs] = msg.wire(Z[fd], z0f[fs], fd=fd, fs=fs)
            c0, c1 = sch.neta_ptr[l], sch.neta_ptr[l + 1]
            if c1 > c0:
                A[sch.neta_fs[c0:c1]] = A[sch.neta_fd[c0:c1]]
            i0, i1 = sch.inst_ptr[l], sch.inst_ptr[l + 1]
            if i1 == i0:
                continue
            fu, launch = sch.inst_fu[i0:i1], sch.inst_launch[i0:i1]
            a_src, s_src = A[fu], S[fu]
            if use is not None:
                u_tf = use[fu]
                a_src, s_src = torch.where(u_tf, yA[fu], a_src), torch.where(u_tf, yS[fu], s_src)
            s_in = torch.where(launch, zero, s_src)
            from_arr = torch.where(launch, clkA[fu], a_src)
            d, sl, zm = msg.cell(st, i0, i1, s_in, None if Z is None else Z[fu])
            d, sl = d.to(dt), sl.to(dt)
            cand = torch.where(sch.inst_search[i0:i1] & (from_arr > NEG_INF), from_arr + d, torch.full_like(d, NEG_INF))
            g0, g1 = sch.seg_ptr[l], sch.seg_ptr[l + 1]
            tgt, loc = sch.seg_ft[g0:g1], sch.inst_seg[i0:i1] - g0
            if pool_a is None:
                A[tgt] = A[tgt].scatter_reduce(0, loc, cand, "amax", include_self=True)
            else:
                A[tgt] = pool_a(A[tgt], cand, loc, g1 - g0)
            S[tgt] = S[tgt].scatter_reduce(0, loc, sl, "amax", include_self=True)
            if Z is not None:
                if pool_z is None:
                    Z[tgt] = Z[tgt].scatter_reduce(0, loc[:, None].expand(-1, dz), zm, "amax", include_self=True)
                else:
                    Z[tgt] = pool_z(Z[tgt], zm, loc, g1 - g0, cand=cand, delay=d, slew=sl, tgt=tgt, level=l)
            if self.monitor and Z is not None:
                zn = Z[tgt].detach().norm(dim=1)
                self._stats["z_norm_sum"] = self._stats.get("z_norm_sum", 0.0) + zn.sum()
                self._stats["z_n"] = self._stats.get("z_n", 0) + len(zn)
                self._stats["z_norm_max"] = torch.maximum(self._stats.get("z_norm_max", zn.new_zeros(())), zn.max())
            if keep:
                keep_d.append(d); keep_s.append(sl); keep_a.append(cand)
                if zm is not None:
                    keep_z.append(zm)
        arr, slew = A.view(N, 2), S.view(N, 2)
        out = {"arr": arr, "slew": slew, "z": None if Z is None else Z.view(N, 2, dz), "schedule": sch}
        out.update(endpoint_slack(b, arr, slew, msg.setup, knockout=b.knockout_tables))
        if keep:
            empty = torch.zeros(0, dtype=dt, device=dev)
            out["msg_delay"] = torch.cat(keep_d) if keep_d else empty
            out["msg_slew"] = torch.cat(keep_s) if keep_s else empty
            out["msg_arrival"] = torch.cat(keep_a) if keep_a else empty
            out["msg_z"] = torch.cat(keep_z) if keep_z else None
        return out


def instance_level_offset(sch: Schedule) -> int:
    return sch.inst_ptr[1]


def arc_delay(out: dict, num_arcs: int) -> torch.Tensor:
    sch = out["schedule"]
    k = instance_level_offset(sch)
    d = out["msg_delay"]
    ad = torch.full((num_arcs, 4), float("nan"), dtype=d.dtype, device=d.device)
    ad[sch.inst_e[k:k + len(d)], sch.inst_pair[k:k + len(d)]] = d
    return ad


class TimingEncoder(nn.Module):

    def __init__(self, vocab, config: dict):
        super().__init__()
        c = dict(config)
        self.config = c
        arch = c.get("arch", "mlp")
        if arch not in ARCHES:
            raise ValueError(f"arch {arch!r} not in {ARCHES}")
        self.arch = arch
        self.arc = ArcEmbedding(vocab.arc_bank, vocab.arc_cap_axis, vocab.slew_axis, mode=c["arc_mode"],
                                width=c["kernel_width"], depth=c["kernel_depth"], d_out=c["d_arc"])
        k = lambda name: c.get(name, ARM_DEFAULTS[name])
        if arch == "mlp":
            self.message = LearnedMessage(self.arc, n_pin_names=len(vocab.pin_name), dz=c["dz"], width=c["mlp_width"],
                                          depth=c["mlp_depth"], correction=c["correction"])
            self.encoder = LevelEncoder(self.message)
        elif arch == "attn":
            from .attn_message import AttentionMessage
            self.message = AttentionMessage(self.arc, n_pin_names=len(vocab.pin_name), dz=c["dz"], width=c["mlp_width"],
                                            depth=c["mlp_depth"], correction=c["correction"], heads=k("heads"),
                                            net_pool=bool(k("net_pool")), fanin_self_attn=bool(k("fanin_self_attn")),
                                            soft_arrival=bool(k("soft_arrival")), soft_tau=k("soft_tau"))
            self.encoder = LevelEncoder(self.message)
        else:
            from .layered import LayeredEncoder
            self.message = None
            self.encoder = LayeredEncoder(self.arc, n_pin_names=len(vocab.pin_name), conv=arch, d_model=k("d_model"),
                                          d_edge=k("d_edge"), layers=k("layers"), heads=k("heads"), level_pe=k("level_pe"),
                                          global_attn=k("global_attn"), checkpoint_layers=bool(k("checkpoint_layers")),
                                          out_scale=k("out_scale"), d_arc=c["d_arc"],
                                          arrival_from_delays=bool(k("arrival_from_delays")), delay_norm=k("delay_norm"))

    @property
    def dz(self) -> int:
        return self.encoder.dz if self.message is None else self.message.dz

    @property
    def layered(self) -> bool:
        return self.message is None

    def forward(self, b, **kw) -> dict:
        return self.encoder(b, **kw)


ARCHES = ("mlp", "attn", "gatv2", "gine", "gps")
DEFAULT_CONFIG = {"arc_mode": "tables", "kernel_width": 512, "kernel_depth": 4, "d_arc": 32, "dz": 64,
                  "mlp_width": 128, "mlp_depth": 2, "correction": True}
ARM_DEFAULTS = {"arch": "mlp",
                "heads": 4,
                "net_pool": 1,
                "fanin_self_attn": 0,
                "soft_arrival": 0,
                "soft_tau": 1.0,
                "layers": 8,
                "d_model": 128,
                "d_edge": 40,
                "level_pe": 0,
                "global_attn": "none",
                "checkpoint_layers": 0,
                "out_scale": 10.0,
                "arrival_from_delays": 0,
                "delay_norm": "scale"}
