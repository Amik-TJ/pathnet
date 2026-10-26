from __future__ import annotations

import numpy as np

from graph.nldm import lookup
from graph.vocab import RF_PAIRS

from .candidates import EOS, LOAD_ROLES, START_ROLES, WalkError, allowed_next, as_walk_graph, decision_pin

NEG_INF = -np.inf


class OracleScorer:

    def __init__(self, g, res: dict, vocab):
        self.g = g = as_walk_graph(g)
        data = g.data
        self.arr, self.slew = res["arr"], res["slew"]
        self.load = data.x_num[:, 2:4].double().numpy()
        self.arc_key = data.arc_key_id.numpy()
        self.arc_mask = data.arc_rf_mask.numpy()
        self.arc_on = data.arc_enabled.numpy()
        self.bank, self.caps, self.slew_axis = vocab.arc_bank, vocab.arc_cap_axis, vocab.slew_axis

    def arc_delay(self, e: int, u: int, o: int, frf: int, trf: int) -> float:
        k = self.arc_key[e]
        return float(lookup(self.bank[k, 0 if trf == 0 else 1], self.slew_axis, self.caps[k],
                            self.slew[u, frf], self.load[o, trf]))

    def score(self, u: int, o: int, rf: int, from_rf: int | None = None):
        best = (NEG_INF, -1, -1)
        for e in self.g.arcs_of_pair.get((u, o), []):
            if not self.arc_on[e]:
                continue
            for pair_idx, (frf, trf) in enumerate(RF_PAIRS):
                if trf != rf or not self.arc_mask[e, pair_idx] or (from_rf is not None and frf != from_rf):
                    continue
                a = self.arr[u, frf]
                if a == NEG_INF:
                    continue
                d = self.arc_delay(e, u, o, frf, trf)
                if a + d > best[0]:
                    best = (a + d, frf, e)
        return best


def decide(g, scorer: OracleScorer, previous_token: int, rf: int, cache: dict | None = None, choose=None) -> dict:
    key = (int(previous_token), int(rf))
    if cache is not None and key in cache:
        return cache[key]
    cands = allowed_next(g, previous_token)
    o = decision_pin(g, previous_token)
    if cands and g.role[cands[0]] in START_ROLES and cands == [o]:
        out = {"o": o, "rf": rf, "candidates": cands, "scores": [None], "from_rf": [rf], "arcs": [-1],
               "choice": o, "choice_rf": rf, "startpoint": True}
    else:
        scored = [scorer.score(u, o, rf) for u in cands]
        best = int((choose or np.argmax)([s[0] for s in scored])) if scored else -1
        ok = best >= 0 and scored[best][0] != NEG_INF
        out = {"o": o, "rf": rf, "candidates": cands, "scores": [s[0] for s in scored],
               "from_rf": [s[1] for s in scored], "arcs": [s[2] for s in scored],
               "choice": cands[best] if ok else None, "choice_rf": scored[best][1] if ok else None,
               "startpoint": False}
    if cache is not None:
        cache[key] = out
    return out


def oracle_walk(g, res: dict, endpoint: int, rf: int, vocab=None, scorer: OracleScorer | None = None,
                cache: dict | None = None, choose=None) -> dict:
    g = as_walk_graph(g)
    if scorer is None:
        if vocab is None:
            raise WalkError("oracle_walk needs vocab or scorer (the arc tables)")
        scorer = OracleScorer(g, res, vocab)
    endpoint, rf = int(endpoint), int(rf)
    out = {"tokens": [], "token_rf": [], "pins": [], "pin_rf": [], "decisions": [], "complete": False, "error": ""}
    if endpoint not in set(allowed_next(g, None)):
        out["error"] = f"{g.name(endpoint)} is not an allowed endpoint"
        return out
    t = endpoint
    out["tokens"].append(t); out["token_rf"].append(rf); out["pins"].append(t); out["pin_rf"].append(rf)
    for _ in range(2 * g.num_nodes + 2):
        if allowed_next(g, t) == [EOS]:
            out["complete"] = True
            return out
        d = decide(g, scorer, t, rf, cache, choose)
        if d["choice"] is None:
            out["error"] = f"no candidate with an arrival after {g.name(t)} (decision pin {g.name(d['o']) if d['o'] >= 0 else None}, rf {rf})"
            return out
        out["decisions"].append(d)
        if g.role[t] in LOAD_ROLES and not d["startpoint"]:
            out["pins"].append(d["o"]); out["pin_rf"].append(rf)
        t, rf = d["choice"], d["choice_rf"]
        out["tokens"].append(t); out["token_rf"].append(rf); out["pins"].append(t); out["pin_rf"].append(rf)
    out["error"] = "the walk did not stop"
    return out
