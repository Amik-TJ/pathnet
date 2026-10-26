from __future__ import annotations

from .candidates import (END_ROLES, LOAD_ROLES, ROLE_CELL_OUT, START_ROLES, WalkError, as_walk_graph)


def _as_list(x) -> list:
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def tokens_from_path(g, path) -> list:
    g = as_walk_graph(g)
    rev = _as_list(path)[::-1]
    if not rev:
        raise WalkError("empty path")
    if g.role[rev[0]] not in END_ROLES:
        raise WalkError(f"path ends at {g.name(rev[0])} (role {g.role[rev[0]]}), not a register D pin or an output port")
    if g.role[rev[-1]] not in START_ROLES:
        raise WalkError(f"path starts at {g.name(rev[-1])} (role {g.role[rev[-1]]}), not a register Q/QN or an input port")
    if len(rev) < 2:
        raise WalkError("a path has at least an endpoint and a startpoint")
    tokens = [rev[0]]
    for prev, p in zip(rev[:-1], rev[1:]):
        if (p, prev) in g.net_edges:
            continue
        tokens.append(p)
    if (rev[-1], rev[-2]) not in g.net_edges:
        raise WalkError(f"the startpoint {g.name(rev[-1])} is not reached by a net step")
    tokens.append(rev[-1])
    return tokens


def pins_from_tokens(g, tokens) -> list:
    g = as_walk_graph(g)
    pins = []
    for t in _as_list(tokens):
        pins.append(t)
        if g.role[t] in LOAD_ROLES:
            d = int(g.driver_of_net[t])
            if d >= 0 and g.role[d] == ROLE_CELL_OUT:
                pins.append(d)
    return pins


def token_count_rule(g, path) -> dict:
    g = as_walk_graph(g)
    if g.pin_instance is None:
        raise WalkError("the graph has no pin_instance")
    p = _as_list(path)
    inst = [int(g.pin_instance[v]) for v in p]
    visits = 1 + sum(1 for i in range(len(p) - 1) if inst[i] != inst[i + 1] or (p[i], p[i + 1]) in g.net_edges)
    con_arcs = sum(1 for a, b in zip(p[:-1], p[1:])
                   if g.role[a] == ROLE_CELL_OUT and g.role[b] == ROLE_CELL_OUT and (a, b) not in g.net_edges)
    is_port = [g.role[v] in (0, 1) for v in p]
    distinct_cells = len({i for i, port in zip(inst, is_port) if not port})
    return {"expected": visits + con_arcs, "visits": visits, "con_arcs": con_arcs,
            "expected_doc": distinct_cells + (1 if is_port[0] else 0) + con_arcs,
            "port_start": bool(is_port[0]), "port_end": bool(is_port[-1]),
            "repeated_instance": visits != len(set(inst))}
