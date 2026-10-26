from __future__ import annotations

import numpy as np

from walk.candidates import allowed_next

TIE_EPS = 1e-9


def tie_breaker(rng):
    def choose(scores):
        s = np.asarray(scores, dtype=np.float64)
        top = np.flatnonzero(s >= s.max() - TIE_EPS) if np.isfinite(s.max()) else np.array([0])
        return int(top[0] if len(top) == 1 else rng.choice(top))
    return choose


def labelled_paths(data):
    ep = data.y_ep_index.tolist(); ptr = data.y_ep_path_ptr.tolist()
    nodes = data.y_ep_path.tolist(); rfs = data.y_ep_path_rf.tolist()
    return [(ep[k], nodes[ptr[k]:ptr[k + 1]], rfs[ptr[k]:ptr[k + 1]]) for k in range(len(ep))]


def teacher_forced_steps(g, path, prf):
    rev, rev_rf = path[::-1], prf[::-1]
    steps = []
    n_forced = 0
    for i in range(len(rev) - 1):
        prev, nxt = rev[i], rev[i + 1]
        if (nxt, prev) in g.net_edges and g.role[nxt] == 3:
            continue
        if (nxt, prev) in g.net_edges:
            n_forced += 1
            continue
        tok = rev[i - 1] if i > 0 and (prev, rev[i - 1]) in g.net_edges else prev
        cands = allowed_next(g, tok)
        if len(cands) > 1:
            steps.append((tok, rev_rf[i], nxt, cands))
        else:
            n_forced += 1
    return steps, n_forced
