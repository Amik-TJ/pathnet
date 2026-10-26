from __future__ import annotations

import ctypes as C
import json
import os
import time

import numpy as np

import torch
from torch_geometric.data import Data

from graph import SCHEMA_VERSION, boolfunc
from graph import build_dataset as old
from graph.build_pin_graph import ROLE_NAME, X_NUM_COLS, BuildError
from graph.labels import LabelError
from graph.liberty_nldm import CONSTRAINT_TABLES, LibertyError
from graph.sdc_three_commands import SdcError, parse_sdc
from graph.vocab import ARC_TYPE_ID, SENSE_ID, rf_mask
from graph.yosys_verilog import NetlistError

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "libfast_graph.so")

P = C.c_void_p
I64P = C.POINTER(C.c_int64)


class LibSpec(C.Structure):
    _fields_ = [(n, t) for n, t in [
        ("n_cells", C.c_int64), ("cell_names", P), ("cell_flags", P), ("pin_ptr", P), ("pin_names", P),
        ("pin_vocab", P), ("pin_dir", P), ("pin_role", P), ("pin_cap_rise", P), ("pin_cap_fall", P),
        ("pin_max_tr", P), ("pin_max_cap", P), ("pin_const_init", P), ("pin_func", P),
        ("arc_ptr", P), ("arc_from", P), ("arc_to", P), ("arc_type", P), ("arc_sense", P), ("arc_rf", P),
        ("arc_when_vocab", P), ("arc_key", P), ("arc_when_expr", P), ("arc_collapse_slot", P),
        ("chk_ptr", P), ("chk_clk", P), ("chk_d", P), ("chk_bank", P),
        ("n_expr", C.c_int64), ("expr_ptr", P), ("expr_code", P), ("expr_slot_ptr", P), ("expr_slot_pin", P)]]


class SdcSpec(C.Structure):
    _fields_ = [("top", C.c_char_p), ("clock_port", C.c_char_p), ("n_in", C.c_int64), ("in_names", C.c_char_p),
                ("in_vals", P), ("n_out", C.c_int64), ("out_names", C.c_char_p), ("out_vals", P)]


_TORCH = {np.dtype(np.int64): torch.long, np.dtype(np.float32): torch.float32, np.dtype(np.bool_): torch.bool}

SIZE_FIELDS = ("N", "n_ports", "n_inst", "E_net", "E_arc", "E_chk", "pin_blob", "inst_blob", "master_blob", "top_len",
               "n_partial", "n_pm", "has_labels", "n_path", "n_worst", "n_ep", "n_csr",
               "nets", "unloaded", "port_to_port", "when_disabled", "default_disabled", "collapsed",
               "escaped", "n_masters", "n_registers", "tie_cells", "n_const")
FP_OK, FP_UNSUPPORTED, FP_FALLBACK = 0, 1, 2

_so = None


def _load_so():
    global _so
    if _so is None:
        if not os.path.exists(SO):
            raise FileNotFoundError(f"{SO} missing; run {HERE}/build.sh")
        so = C.CDLL(SO)
        so.fp_lib_new.restype = P; so.fp_lib_new.argtypes = [C.POINTER(LibSpec)]
        so.fp_lib_free.argtypes = [P]
        so.fp_build.restype = P; so.fp_build.argtypes = [P, C.c_char_p, C.POINTER(SdcSpec), C.c_char_p]
        so.fp_status.restype = C.c_int; so.fp_status.argtypes = [P]
        so.fp_message.restype = C.c_int64; so.fp_message.argtypes = [P, C.c_char_p, C.c_int64]
        so.fp_sizes.argtypes = [P, I64P]
        so.fp_fill.argtypes = [P, C.POINTER(P)]
        so.fp_free.argtypes = [P]
        so.fp_np_sum.restype = C.c_double; so.fp_np_sum.argtypes = [P, C.c_int64]
        _so = so
    return _so


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(P)


def _blob(strings) -> bytes:
    return b"".join(s.encode() + b"\0" for s in strings)


class _Exprs:

    def __init__(self):
        self.ptr, self.code, self.slot_ptr, self.slot_pin = [0], [], [0], []
        self.cache = {}
        self.slots = []

    def compile(self, cell, expr: str):
        key = (cell.name, expr)
        if key in self.cache:
            return self.cache[key]
        try:
            ast = boolfunc.parse_function(expr)
        except boolfunc.FuncError:
            self.cache[key] = -2
            return -2
        names: list[str] = []
        code: list[int] = []
        ops = {"not": -1, "and": -2, "or": -3, "xor": -4}

        def emit(node):
            if node[0] == "var":
                if node[1] not in names:
                    names.append(node[1])
                code.append(names.index(node[1]))
            else:
                for ch in node[1:]:
                    emit(ch)
                code.append(ops[node[0]])
        emit(ast)
        assert len(code) < 256 and len(names) < 64, (cell.name, expr)
        pins = list(cell.pins)
        eid = len(self.ptr) - 1
        self.code += code; self.ptr.append(len(self.code))
        self.slot_pin += [pins.index(n) if n in cell.pins and cell.pins[n].direction == "input" else -1 for n in names]
        self.slot_ptr.append(len(self.slot_pin))
        self.slots.append(names)
        self.cache[key] = eid
        return eid


def flatten_library(lib, vocab) -> dict:
    ids = vocab.cell_master
    names = [None] * len(ids)
    for n, i in ids.items():
        names[i] = n
    assert set(names[1:]) == set(lib.cells), "vocab and Liberty disagree on the cell set"
    ex = _Exprs()
    F = {k: [] for k in ("cell_flags", "pin_ptr", "pin_names", "pin_vocab", "pin_dir", "pin_role", "pin_cap_rise",
                         "pin_cap_fall", "pin_max_tr", "pin_max_cap", "pin_const_init", "pin_func",
                         "arc_ptr", "arc_from", "arc_to", "arc_type", "arc_sense", "arc_rf", "arc_when_vocab",
                         "arc_key", "arc_when_expr", "arc_collapse_slot", "chk_ptr", "chk_clk", "chk_d", "chk_bank")}
    bank_t, bank_a = [], []
    n7 = len(vocab.slew_axis)
    F["pin_ptr"] += [0, 0]; F["arc_ptr"] += [0, 0]; F["chk_ptr"] += [0, 0]
    F["cell_flags"].append(0)
    for cid in range(1, len(names)):
        cname = names[cid]
        cell = lib.cells[cname]
        pins = list(cell.pins)
        broken = False
        ok, _ = lib.supported_master(cname)
        flags = (1 if ok else 0) | (8 if cell.is_tie else 0)
        role = {p: (2 if pin.direction == "input" else 3) for p, pin in cell.pins.items()}
        checks = list(cell.checks("setup_rising"))
        if checks:
            flags |= 2
            try:
                clk_pins = {c for c, d, _ in checks}
                d_pins = {d for c, d, _ in checks}
                for c in clk_pins:
                    if not cell.pins[c].is_clock:
                        broken = True
                    role[c] = 4
                for d in d_pins:
                    role[d] = 5
                for f, t, tg in cell.arcs():
                    if tg.timing_type == "rising_edge":
                        role[t] = 6
                if cell.ff is not None:
                    clocked_on, next_state = cell.ff
                    if clocked_on not in clk_pins or not any(d in next_state for d in d_pins):
                        broken = True
            except KeyError:
                broken = True
        for pname, pin in cell.pins.items():
            F["pin_names"].append(pname)
            F["pin_vocab"].append(vocab.pin_name.get(pname, -1))
            broken |= pname not in vocab.pin_name
            F["pin_dir"].append(0 if pin.direction == "input" else 1 if pin.direction == "output" else 2)
            F["pin_role"].append(role[pname])
            F["pin_cap_rise"].append(pin.cap_rise if pin.direction == "input" else 0.0)
            F["pin_cap_fall"].append(pin.cap_fall if pin.direction == "input" else 0.0)
            F["pin_max_tr"].append(pin.max_transition or 0.0)
            F["pin_max_cap"].append(pin.max_capacitance or 0.0)
            fn = pin.function
            F["pin_const_init"].append(int(fn.strip()) if pin.direction == "output" and fn is not None and fn.strip() in ("0", "1") else -1)
            F["pin_func"].append(-1 if fn is None else ex.compile(cell, fn))
        func_of = {p: F["pin_func"][len(F["pin_func"]) - len(pins) + k] for k, p in enumerate(pins)}
        for f, t, tg in cell.arcs():
            if f not in cell.pins or t not in cell.pins:
                broken = True
                continue
            w = tg.when if tg.when is not None else "<none>"
            key = (cname, f, t, tg.timing_type, w)
            if w not in vocab.when or key not in vocab.arc_key or tg.timing_type not in ARC_TYPE_ID:
                broken = True
                continue
            F["arc_from"].append(pins.index(f)); F["arc_to"].append(pins.index(t))
            F["arc_type"].append(ARC_TYPE_ID[tg.timing_type])
            F["arc_sense"].append(SENSE_ID.get(tg.timing_sense or "non_unate", 2))
            F["arc_rf"] += [int(b) for b in rf_mask(tg.timing_type, tg.timing_sense)]
            F["arc_when_vocab"].append(vocab.when[w])
            F["arc_key"].append(vocab.arc_key[key])
            F["arc_when_expr"].append(-1 if tg.when is None else ex.compile(cell, tg.when))
            slot = -1
            e = func_of[t]
            if e >= 0 and cell.pins[f].direction == "input" and f in ex.slots[e]:
                slot = ex.slots[e].index(f)
            F["arc_collapse_slot"].append(slot)
        for clk, d, tg in checks:
            tabs = [tg.tables.get(n) for n in CONSTRAINT_TABLES]
            if any(tb is None for tb in tabs) or clk not in cell.pins or d not in cell.pins:
                broken = True
                continue
            t_arr = np.stack([tb.values for tb in tabs])
            a_arr = np.stack([np.stack([tb.axis1, tb.axis2]) for tb in tabs])
            if t_arr.shape != (2, n7, n7) or a_arr.shape != (2, 2, n7):
                broken = True
                continue
            F["chk_clk"].append(pins.index(clk)); F["chk_d"].append(pins.index(d)); F["chk_bank"].append(len(bank_t))
            bank_t.append(t_arr); bank_a.append(a_arr)
        F["cell_flags"].append(flags | (4 if broken else 0))
        F["pin_ptr"].append(len(F["pin_names"])); F["arc_ptr"].append(len(F["arc_from"])); F["chk_ptr"].append(len(F["chk_clk"]))
    i64 = lambda k: np.asarray(F[k], dtype=np.int64)
    i8 = lambda k: np.asarray(F[k], dtype=np.int8)
    f64 = lambda k: np.asarray(F[k], dtype=np.float64)
    arr = {
        "cell_names": np.frombuffer(_blob(names), dtype=np.uint8).copy(), "cell_flags": i8("cell_flags"),
        "pin_ptr": i64("pin_ptr"), "pin_names": np.frombuffer(_blob(F["pin_names"]), dtype=np.uint8).copy(),
        "pin_vocab": i64("pin_vocab"), "pin_dir": i8("pin_dir"), "pin_role": i8("pin_role"),
        "pin_cap_rise": f64("pin_cap_rise"), "pin_cap_fall": f64("pin_cap_fall"), "pin_max_tr": f64("pin_max_tr"),
        "pin_max_cap": f64("pin_max_cap"), "pin_const_init": i8("pin_const_init"), "pin_func": i64("pin_func"),
        "arc_ptr": i64("arc_ptr"), "arc_from": i64("arc_from"), "arc_to": i64("arc_to"), "arc_type": i64("arc_type"),
        "arc_sense": i64("arc_sense"), "arc_rf": i8("arc_rf"), "arc_when_vocab": i64("arc_when_vocab"),
        "arc_key": i64("arc_key"), "arc_when_expr": i64("arc_when_expr"), "arc_collapse_slot": i64("arc_collapse_slot"),
        "chk_ptr": i64("chk_ptr"), "chk_clk": i64("chk_clk"), "chk_d": i64("chk_d"), "chk_bank": i64("chk_bank"),
        "expr_ptr": np.asarray(ex.ptr, np.int64), "expr_code": np.asarray(ex.code, np.int64),
        "expr_slot_ptr": np.asarray(ex.slot_ptr, np.int64), "expr_slot_pin": np.asarray(ex.slot_pin, np.int64),
    }
    arr["bank_tables"] = (np.stack(bank_t) if bank_t else np.zeros((0, 2, n7, n7))).astype(np.float32)
    arr["bank_axes"] = (np.stack(bank_a) if bank_a else np.zeros((0, 2, 2, n7))).astype(np.float32)
    arr["n_cells"] = len(names); arr["n_expr"] = len(ex.ptr) - 1; arr["cell_name_list"] = names
    for k in ("pin_ptr", "arc_ptr", "chk_ptr"):
        assert len(arr[k]) == len(names) + 1, k
    return arr


class FastBuilder:
    def __init__(self, lib, vocab):
        self.lib, self.vocab = lib, vocab
        self.so = _load_so()
        self.arr = flatten_library(lib, vocab)
        a = self.arr
        spec = LibSpec(n_cells=a["n_cells"], n_expr=a["n_expr"],
                       **{k: _ptr(a[k]) for k, _ in LibSpec._fields_ if k not in ("n_cells", "n_expr")})
        self.handle = self.so.fp_lib_new(C.byref(spec))
        self.n7 = len(vocab.slew_axis)

    def __del__(self):
        if getattr(self, "handle", None) and self.so is not None:
            self.so.fp_lib_free(self.handle)
            self.handle = None

    def build_graph(self, netlist_path, sdc, *, design, group, corner, label_dir=None, sweep_summary=None):
        so, vocab = self.so, self.vocab
        ins = list(sdc.input_delay.items()); outs = list(sdc.output_delay.items())
        in_vals = np.asarray([v for _, v in ins], np.float64); out_vals = np.asarray([v for _, v in outs], np.float64)
        spec = SdcSpec(top=None if sdc.top is None else sdc.top.encode(),
                       clock_port=None if sdc.clock_port is None else sdc.clock_port.encode(),
                       n_in=len(ins), in_names=_blob(p for p, _ in ins), in_vals=_ptr(in_vals),
                       n_out=len(outs), out_names=_blob(p for p, _ in outs), out_vals=_ptr(out_vals))
        res = so.fp_build(self.handle, netlist_path.encode(), C.byref(spec),
                          None if label_dir is None else label_dir.encode())
        try:
            st = so.fp_status(res)
            if st != FP_OK:
                n = so.fp_message(res, None, 0)
                buf = C.create_string_buffer(n)
                so.fp_message(res, buf, n)
                msg = buf.raw[:n].decode()
                if st == FP_UNSUPPORTED:
                    bad = msg.split("\0")[:-1]
                    raise BuildError("unsupported masters: " + ", ".join(f"{m} ({self.lib.supported_master(m)[1]})" for m in bad))
                return None, None, f"fallback:{msg}"
            sz = np.zeros(len(SIZE_FIELDS), np.int64)
            so.fp_sizes(res, sz.ctypes.data_as(I64P))
            S = dict(zip(SIZE_FIELDS, (int(x) for x in sz)))
            N, Ea, En, Ec = S["N"], S["E_arc"], S["E_net"], S["E_chk"]
            e = np.empty
            out = [e((N, 3), np.int64), e((N, 9), np.float32), e((N, 2), np.float32), e(N, np.int64), e(N, np.int64),
                   e((2, En), np.int64), e((2, Ea), np.int64), e(Ea, np.int64), e(Ea, np.int64), e((Ea, 4), np.bool_),
                   e(Ea, np.bool_), e(Ea, np.int64), e(Ea, np.int64), e(Ea, np.int64), e(Ea, np.bool_),
                   e((2, Ec), np.int64), e(Ec, np.int64),
                   e(S["pin_blob"], np.uint8), e(S["inst_blob"], np.uint8), e(S["master_blob"], np.uint8), e(S["top_len"], np.uint8),
                   e(S["n_partial"], np.int64), e(S["n_pm"], np.int64), e(S["n_pm"], np.int64),
                   e(S["n_path"], np.int64), e(S["n_path"], np.int64), e(S["n_path"], np.float32), e(S["n_worst"], np.int64),
                   e(S["n_ep"], np.int64), e(S["n_ep"], np.float32), e(S["n_ep"], np.float32), e(S["n_ep"], np.float32),
                   e(S["n_csr"], np.int64), e(S["n_ep"] + 1 if S["has_labels"] else 0, np.int64), e(S["n_csr"], np.int64),
                   e(S["n_csr"], np.float32), e(2, np.float64)]
            ptrs = (P * len(out))(*[o.ctypes.data for o in out])
            so.fp_fill(res, ptrs)
        finally:
            so.fp_free(res)
        (x_cat, x_num, x_erc, pin_instance, level, ei_net, ei_arc, a_type, a_sense, a_rf, a_wp, a_when, a_key, a_inst,
         a_en, ei_chk, c_bank, pin_blob, inst_blob, master_blob, top_b, partial, pm_m, pm_c,
         y_path, y_path_rf, y_path_arr, y_worst, y_ep_index, y_ep_slack, y_ep_arr, y_ep_req, y_ep_path, y_ep_ptr,
         y_ep_rf, y_ep_path_arr, dbl) = out

        def T(a, empty=None):
            if a.size:
                return torch.from_numpy(a)
            return empty() if empty is not None else torch.tensor(np.asarray([]), dtype=_TORCH[a.dtype])
        e_edges = lambda: torch.tensor(np.asarray(np.array([[], []]).reshape(2, -1)), dtype=torch.long)
        e_list = lambda dt: (lambda: torch.tensor([], dtype=dt))
        n7 = self.n7
        clock_is_virtual = sdc.clock_port is None
        data = Data(
            num_nodes=N,
            x_cat=T(x_cat), x_num=T(x_num), x_erc=T(x_erc),
            g=torch.tensor([[sdc.period]], dtype=torch.float32),
            pin_instance=T(pin_instance), level=T(level),
            edge_index_net=T(ei_net, e_edges), edge_index_arc=T(ei_arc, e_edges),
            arc_type=T(a_type), arc_sense=T(a_sense),
            arc_rf_mask=T(a_rf, lambda: torch.tensor(np.asarray(np.array([], dtype=bool).reshape(-1, 4)), dtype=torch.bool)),
            arc_when_present=T(a_wp, lambda: torch.tensor(np.asarray(np.array([]) != 0), dtype=torch.bool)),
            arc_when_id=T(a_when), arc_key_id=T(a_key), arc_instance=T(a_inst),
            arc_enabled=T(a_en),
            edge_index_check=T(ei_chk, e_edges),
            check_tables=T(self.arr["bank_tables"][c_bank]) if Ec else torch.tensor(np.zeros((0, 2, n7, n7)), dtype=torch.float32),
            check_axes=T(self.arr["bank_axes"][c_bank]) if Ec else torch.tensor(np.zeros((0, 2, n7)), dtype=torch.float32),
            slew_axis=torch.tensor(np.asarray(vocab.slew_axis), dtype=torch.float32),
        )
        data.pin_names = pin_blob.tobytes().decode().split("\0")[:-1]
        data.instance_names = inst_blob.tobytes().decode().split("\0")[:-1]
        data.instance_masters = master_blob.tobytes().decode().split("\0")[:-1]
        top = top_b.tobytes().decode()
        data.design = design; data.group = group; data.top = top; data.corner = corner
        data.lib_sha256 = list(vocab.lib_sha256); data.clock_is_virtual = clock_is_virtual
        data.schema_version = SCHEMA_VERSION
        data.x_num_cols = list(X_NUM_COLS)

        role = x_cat[:, 0]
        partial_names = [data.instance_names[i] for i in partial.tolist()]
        names = self.arr["cell_name_list"]
        counts = {
            "top": top, "ports": S["n_ports"],
            "input_ports": int((role == 0).sum()), "output_ports": int((role == 1).sum()),
            "instances": S["n_inst"], "masters": S["n_masters"], "registers": S["n_registers"],
            "escaped_instance_names": S["escaped"],
            "cell_pins": N - S["n_ports"], "cell_input_pins": int(np.isin(role, [2, 4, 5]).sum()),
            "cell_output_pins": int(np.isin(role, [3, 6]).sum()),
            "nodes": N, "nets": S["nets"], "unloaded_nets": S["unloaded"], "port_to_port_nets": S["port_to_port"],
            "net_edges": En, "arc_edges": Ea, "check_edges": Ec,
            "arc_edges_disabled": int(Ea - int(a_en.sum())), "when_arcs_disabled": S["when_disabled"],
            "default_arcs_disabled_by_true_when": S["default_disabled"],
            "collapsed_arcs_disabled": S["collapsed"], "tie_cells": S["tie_cells"],
            "constant_pins": S["n_const"], "partially_constant_cells": len(partial_names),
            "levels": int(level.max()) + 1, "max_fanout": int(x_num[:, 4].max()),
            "role_histogram": {ROLE_NAME[k]: int((role == k).sum()) for k in sorted(ROLE_NAME)},
            "period_ps": sdc.period, "clock_is_virtual": clock_is_virtual,
        }
        rep = {"counts": counts, "per_master_arcs": {names[m]: int(c) for m, c in zip(pm_m.tolist(), pm_c.tolist())},
               "partially_constant_cells": partial_names[:50], "warnings": []}
        if S["has_labels"]:
            L, F = e_list(torch.long), e_list(torch.float32)
            data.y_path = T(y_path, L)
            data.y_path_rf = T(y_path_rf, L)
            data.y_path_arrival = T(y_path_arr, F)
            data.y_path_slack = torch.tensor(float(dbl[1]), dtype=torch.float32)
            data.y_worst_endpoints = T(y_worst, L)
            data.y_ep_index = T(y_ep_index, L)
            data.y_ep_slack = T(y_ep_slack, F)
            data.y_ep_arrival = T(y_ep_arr, F)
            data.y_ep_required = T(y_ep_req, F)
            data.y_ep_path = T(y_ep_path, L)
            data.y_ep_path_ptr = T(y_ep_ptr, L)
            data.y_ep_path_rf = T(y_ep_rf, L)
            data.y_ep_path_arrival = T(y_ep_path_arr, F)
            data.sweep_summary = dict(sweep_summary)
            rep["labels"] = {"paths": S["n_ep"], "bad_steps": 0, "bad_step_examples": [], "unknown_pins": 0,
                             "non_alternating": 0, "worst_slack": float(dbl[0]), "tied_worst_endpoints": S["n_worst"],
                             "summary_tied": int(sweep_summary.get("n_endpoints_tied_worst", -1)),
                             "summary_worst_slack": float(sweep_summary.get("worst_slack_ps", "nan"))}
        return data, rep, "ok"

    def build_one(self, netlist_path, sdc_path, *, design, group, corner, label_dir=None, out_dir=None,
                  strict_labels=True, timings=None):
        if not strict_labels:
            return self._python(netlist_path, sdc_path, design, group, corner, label_dir, out_dir, strict_labels, "lenient labels", timings)
        t0 = time.time()
        tp = time.perf_counter()
        row = {"group": group, "design": design, "status": "ok", "reason": "", "seconds": 0.0, "engine": "fast"}
        try:
            try:
                sdc = parse_sdc(sdc_path)
                summary = None
                if label_dir is not None:
                    import csv
                    with open(os.path.join(label_dir, "summary.tsv")) as f:
                        summary = next(csv.DictReader(f, delimiter="\t"))
            except Exception:
                return self._python(netlist_path, sdc_path, design, group, corner, label_dir, out_dir, strict_labels, "sdc/summary", timings)
            data, rep, status = self.build_graph(netlist_path, sdc, design=design, group=group, corner=corner,
                                                 label_dir=label_dir, sweep_summary=summary)
            if data is None:
                return self._python(netlist_path, sdc_path, design, group, corner, label_dir, out_dir, strict_labels, status, timings)
            labels = rep.pop("labels", None)
            rep_dict = {**rep, "netlist": netlist_path, "sdc": sdc_path, "lib_files": self.vocab.lib_files,
                        "lib_sha256": self.vocab.lib_sha256}
            if labels is not None:
                rep_dict["labels"] = labels
            t_build = time.perf_counter() - tp
            t_save = t_rep = 0.0
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                t1 = time.perf_counter()
                torch.save(data, os.path.join(out_dir, "graph.pt"))
                t2 = time.perf_counter()
                with open(os.path.join(out_dir, "build_report.json"), "w") as f:
                    json.dump(rep_dict, f, indent=1, default=str)
                t_save, t_rep = t2 - t1, time.perf_counter() - t2
            if timings is not None:
                timings.update(build_s=t_build, save_s=t_save, report_s=t_rep)
            c = rep["counts"]
            row.update({k: c[k] for k in ("nodes", "instances", "registers", "net_edges", "arc_edges", "levels", "constant_pins")})
            if labels is not None:
                row["paths"] = labels["paths"]; row["tied"] = labels["tied_worst_endpoints"]
            row["seconds"] = round(time.time() - t0, 2)
            return data, rep_dict, row
        except (NetlistError, SdcError, BuildError, LabelError, LibertyError) as e:
            row.update({"status": "skipped", "reason": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 2)})
            if timings is not None:
                timings.update(build_s=time.perf_counter() - tp, save_s=0.0, report_s=0.0)
            return None, None, row

    def _python(self, netlist_path, sdc_path, design, group, corner, label_dir, out_dir, strict_labels, why, timings):
        t = time.perf_counter()
        data, rep, row = old.build_one(netlist_path, sdc_path, self.lib, self.vocab, design=design, group=group,
                                       corner=corner, label_dir=label_dir, out_dir=out_dir, strict_labels=strict_labels)
        row["engine"] = "python-fallback"
        row["fallback_reason"] = why
        if timings is not None:
            timings.update(build_s=time.perf_counter() - t, save_s=0.0, report_s=0.0)
        return data, rep, row


def np_sum_cpp(a: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return _load_so().fp_np_sum(_ptr(a), len(a))
