from __future__ import annotations

import numpy as np

ROLE_IN_PORT, ROLE_OUT_PORT, ROLE_CELL_IN, ROLE_CELL_OUT, ROLE_REG_CLK, ROLE_REG_D, ROLE_REG_Q = range(7)
START_ROLES = (ROLE_REG_Q, ROLE_IN_PORT)
END_ROLES = (ROLE_REG_D, ROLE_OUT_PORT)
LOAD_ROLES = (ROLE_REG_D, ROLE_OUT_PORT, ROLE_CELL_IN)
IS_CONSTANT_COL = 7
EOS = -1


class WalkError(ValueError):
    pass


class WalkGraph:

    def __init__(self, data):
        self.data = data
        self.num_nodes = int(data.x_cat.shape[0])
        self.role = data.x_cat[:, 0].numpy().astype(np.int64)
        self.is_constant = data.x_num[:, IS_CONSTANT_COL].numpy() > 0
        self.pin_instance = data.pin_instance.numpy() if getattr(data, "pin_instance", None) is not None else None

        net_src, net_dst = data.edge_index_net.numpy()
        self.driver_of_net = -np.ones(self.num_nodes, dtype=np.int64)
        self.multi_driven = []
        for d, s in zip(net_src.tolist(), net_dst.tolist()):
            if self.driver_of_net[s] >= 0 and self.driver_of_net[s] != d:
                self.multi_driven.append(s)
                continue
            self.driver_of_net[s] = d
        self.net_edges = set(zip(net_src.tolist(), net_dst.tolist()))

        arc_src, arc_dst = data.edge_index_arc.numpy()
        enabled = getattr(data, "arc_enabled", None)
        enabled = np.ones(len(arc_src), dtype=bool) if enabled is None else enabled.numpy()
        preds_all: dict[int, set] = {}
        preds_on: dict[int, set] = {}
        self.arcs_of_pair: dict[tuple, list] = {}
        for e, (u, o) in enumerate(zip(arc_src.tolist(), arc_dst.tolist())):
            self.arcs_of_pair.setdefault((u, o), []).append(e)
            if self.is_constant[u]:
                continue
            preds_all.setdefault(o, set()).add(u)
            if enabled[e]:
                preds_on.setdefault(o, set()).add(u)
        self.arc_predecessors = {o: sorted(s) for o, s in preds_all.items()}
        self.arc_predecessors_enabled = {o: sorted(s) for o, s in preds_on.items()}
        self.endpoints = np.flatnonzero(np.isin(self.role, END_ROLES) & ~self.is_constant).tolist()

    def name(self, pin: int) -> str:
        names = getattr(self.data, "pin_names", None)
        return names[pin] if names is not None and 0 <= pin < len(names) else str(pin)


def as_walk_graph(g) -> WalkGraph:
    return g if isinstance(g, WalkGraph) else WalkGraph(g)


def decision_pin(g, token: int) -> int:
    g = as_walk_graph(g)
    r = g.role[token]
    if r in LOAD_ROLES:
        return int(g.driver_of_net[token])
    if r == ROLE_CELL_OUT:
        return int(token)
    raise WalkError(f"pin {g.name(token)} (role {r}) has no decision pin")


def allowed_next(g, previous_token, enabled_only: bool = True) -> list:
    g = as_walk_graph(g)
    if previous_token is None:
        return list(g.endpoints)
    t = int(previous_token)
    r = g.role[t]
    if r in START_ROLES:
        return [EOS]
    if r == ROLE_REG_CLK:
        raise WalkError(f"a register CLK pin is never a token: {g.name(t)}")
    o = decision_pin(g, t)
    if o < 0:
        return []
    if g.role[o] in START_ROLES:
        return [o]
    preds = g.arc_predecessors_enabled if enabled_only else g.arc_predecessors
    return list(preds.get(o, []))
