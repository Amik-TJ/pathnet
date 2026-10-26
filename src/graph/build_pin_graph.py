from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import Data

from . import SCHEMA_VERSION, boolfunc
from .liberty_nldm import CONSTRAINT_TABLES, Library
from .sdc_three_commands import Sdc
from .vocab import ARC_TYPE_ID, NO_WHEN, PORT_TOKEN, SENSE_ID, Vocab, rf_mask
from .yosys_verilog import Netlist

ROLE = {"in_port": 0, "out_port": 1, "cell_in": 2, "cell_out": 3, "reg_clk": 4, "reg_d": 5, "reg_q": 6}
ROLE_NAME = {v: k for k, v in ROLE.items()}
X_NUM_COLS = ("cap_rise", "cap_fall", "load_rise", "load_fall", "fanout", "input_delay", "output_delay",
              "is_constant", "is_clock_port")


class BuildError(ValueError):
    pass


@dataclass
class BuildReport:
    counts: dict = field(default_factory=dict)
    per_master_arcs: dict = field(default_factory=dict)
    unsupported_masters: dict = field(default_factory=dict)
    partially_constant_cells: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def build_graph(nl: Netlist, lib: Library, sdc: Sdc, vocab: Vocab, *, design: str = "", group: str = "",
                corner: str = "BC") -> tuple[Data, BuildReport]:
    rep = BuildReport()
    masters = sorted({i.master for i in nl.instances})
    for m in masters:
        ok, why = lib.supported_master(m)
        if not ok:
            rep.unsupported_masters[m] = why
    if rep.unsupported_masters:
        raise BuildError("unsupported masters: " + ", ".join(f"{m} ({w})" for m, w in rep.unsupported_masters.items()))
    if sdc.top is not None and sdc.top != nl.top:
        raise BuildError(f"SDC current_design {sdc.top} != netlist top {nl.top}")

    pin_names: list[str] = []
    node_of: dict[str, int] = {}
    role = []
    master_id, pinname_id, pin_instance = [], [], []
    cap_rise, cap_fall, max_tr, max_cap = [], [], [], []
    is_clock_pin = []
    n_inst = len(nl.instances)
    port_dir: dict[str, str] = {}
    for pi, port in enumerate(nl.ports):
        node_of[port.name] = len(pin_names); pin_names.append(port.name)
        role.append(ROLE["in_port"] if port.direction == "input" else ROLE["out_port"])
        master_id.append(0); pinname_id.append(0); pin_instance.append(n_inst + pi)
        cap_rise.append(0.0); cap_fall.append(0.0); max_tr.append(0.0); max_cap.append(0.0); is_clock_pin.append(False)
        port_dir[port.name] = port.direction
    inst_first_node = []
    for ii, inst in enumerate(nl.instances):
        cell = lib.cells[inst.master]
        inst_first_node.append(len(pin_names))
        extra = set(inst.pins) - set(cell.pins)
        if extra:
            raise BuildError(f"{inst.name} ({inst.master}) connects pins not in Liberty: {sorted(extra)}")
        missing = [p for p in cell.pins if p not in inst.pins]
        if missing:
            raise BuildError(f"{inst.name} ({inst.master}) leaves Liberty pins unconnected: {missing}")
        for pname, pin in cell.pins.items():
            full = f"{inst.name}/{pname}"
            node_of[full] = len(pin_names); pin_names.append(full)
            role.append(ROLE["cell_in"] if pin.direction == "input" else ROLE["cell_out"])
            master_id.append(vocab.cell_master[inst.master]); pinname_id.append(vocab.pin_name[pname])
            pin_instance.append(ii)
            cap_rise.append(pin.cap_rise if pin.direction == "input" else 0.0)
            cap_fall.append(pin.cap_fall if pin.direction == "input" else 0.0)
            max_tr.append(pin.max_transition or 0.0); max_cap.append(pin.max_capacitance or 0.0)
            is_clock_pin.append(pin.is_clock)
    N = len(pin_names)
    role = np.array(role, dtype=np.int64)

    registers = []
    for ii, inst in enumerate(nl.instances):
        cell = lib.cells[inst.master]
        checks = list(cell.checks("setup_rising"))
        if not checks:
            continue
        clk_pins = {c for c, d, _ in checks}
        d_pins = {d for c, d, _ in checks}
        for c in clk_pins:
            if not cell.pins[c].is_clock:
                raise BuildError(f"{inst.master}: setup related_pin {c} is not a clock pin")
            role[node_of[f"{inst.name}/{c}"]] = ROLE["reg_clk"]
        for d in d_pins:
            role[node_of[f"{inst.name}/{d}"]] = ROLE["reg_d"]
        for f, t, tg in cell.arcs():
            if tg.timing_type == "rising_edge":
                role[node_of[f"{inst.name}/{t}"]] = ROLE["reg_q"]
        if cell.ff is not None:
            clocked_on, next_state = cell.ff
            if clocked_on not in clk_pins:
                raise BuildError(f"{inst.master}: ff.clocked_on {clocked_on} != setup related_pin {clk_pins}")
            if not any(d in next_state for d in d_pins):
                raise BuildError(f"{inst.master}: ff.next_state {next_state!r} does not name the D pin {d_pins}")
        registers.append(ii)

    driver: dict[str, int] = {}
    sinks: dict[str, list[int]] = collections.defaultdict(list)
    for port in nl.ports:
        (driver.__setitem__ if port.direction == "input" else lambda n, v: sinks[n].append(v))(port.name, node_of[port.name])
    for inst in nl.instances:
        cell = lib.cells[inst.master]
        for pname, net in inst.pins.items():
            v = node_of[f"{inst.name}/{pname}"]
            if cell.pins[pname].direction == "output":
                if net in driver:
                    raise BuildError(f"net {net} has two drivers: {pin_names[driver[net]]} and {pin_names[v]}")
                driver[net] = v
            else:
                sinks[net].append(v)
    undriven = [n for n in sinks if n not in driver]
    if undriven:
        raise BuildError(f"{len(undriven)} nets without driver, e.g. {undriven[:3]}")
    net_of_node = {}
    for n, d in driver.items():
        net_of_node[d] = n
        for s in sinks[n]:
            net_of_node[s] = n
    unloaded = [n for n in driver if not sinks.get(n)]
    port_to_port = sum(1 for n, d in driver.items() if role[d] == ROLE["in_port"] and any(role[s] == ROLE["out_port"] for s in sinks[n]))

    src_net, dst_net = [], []
    for n, d in driver.items():
        for s in sinks.get(n, []):
            src_net.append(d); dst_net.append(s)

    load_rise = np.zeros(N); load_fall = np.zeros(N); fanout = np.zeros(N)
    cap_rise = np.array(cap_rise); cap_fall = np.array(cap_fall)
    for n, d in driver.items():
        ss = sinks.get(n, [])
        load_rise[d] = cap_rise[ss].sum(); load_fall[d] = cap_fall[ss].sum(); fanout[d] = len(ss)

    input_delay = np.zeros(N); output_delay = np.zeros(N); is_clock_port = np.zeros(N)
    clock_is_virtual = sdc.clock_port is None
    if not clock_is_virtual:
        if sdc.clock_port not in node_of or port_dir.get(sdc.clock_port) != "input":
            raise BuildError(f"SDC clock port {sdc.clock_port} is not an input port of the netlist")
        is_clock_port[node_of[sdc.clock_port]] = 1.0
    for p, v in sdc.input_delay.items():
        if port_dir.get(p) != "input":
            raise BuildError(f"SDC input delay on unknown/non-input port {p}")
        input_delay[node_of[p]] = v
    for p, v in sdc.output_delay.items():
        if port_dir.get(p) != "output":
            raise BuildError(f"SDC output delay on unknown/non-output port {p}")
        output_delay[node_of[p]] = v
    missing_in = [p.name for p in nl.ports if p.direction == "input" and p.name != sdc.clock_port and p.name not in sdc.input_delay]
    missing_out = [p.name for p in nl.ports if p.direction == "output" and p.name not in sdc.output_delay]
    if missing_in or missing_out:
        raise BuildError(f"ports without SDC delay: in {missing_in[:3]} out {missing_out[:3]}")

    const: dict[int, int] = {}
    for ii, inst in enumerate(nl.instances):
        cell = lib.cells[inst.master]
        for pname, pin in cell.pins.items():
            if pin.direction == "output" and pin.function is not None and pin.function.strip() in ("0", "1"):
                const[node_of[f"{inst.name}/{pname}"]] = int(pin.function.strip())
    partially = set()
    changed = True
    while changed:
        changed = False
        for n, d in driver.items():
            if d in const:
                for s in sinks.get(n, []):
                    if s not in const:
                        const[s] = const[d]; changed = True
        for ii, inst in enumerate(nl.instances):
            cell = lib.cells[inst.master]
            env = {}
            for pname, pin in cell.pins.items():
                if pin.direction == "input":
                    v = node_of[f"{inst.name}/{pname}"]
                    if v in const:
                        env[pname] = const[v]
            if not env:
                continue
            for pname, pin in cell.pins.items():
                if pin.direction != "output" or pin.function is None:
                    continue
                out = node_of[f"{inst.name}/{pname}"]
                if out in const:
                    continue
                try:
                    val = boolfunc.eval_function(pin.function, env)
                except boolfunc.FuncError:
                    val = None
                if val is not None:
                    const[out] = val; changed = True
                else:
                    partially.add(ii)
    is_constant = np.zeros(N)
    for v in const:
        is_constant[v] = 1.0
    rep.partially_constant_cells = [nl.instances[ii].name for ii in sorted(partially) if not all(
        node_of[f"{nl.instances[ii].name}/{p}"] in const for p, pin in lib.cells[nl.instances[ii].master].pins.items() if pin.direction == "output")]

    a_src, a_dst, a_type, a_sense, a_mask, a_when, a_key, a_inst, a_enabled = [], [], [], [], [], [], [], [], []
    per_master_arcs = collections.Counter()
    n_when_disabled = n_collapsed = 0
    n_default_disabled = 0
    for ii, inst in enumerate(nl.instances):
        cell = lib.cells[inst.master]
        env = {p: const[node_of[f"{inst.name}/{p}"]] for p, pin in cell.pins.items()
               if pin.direction == "input" and node_of[f"{inst.name}/{p}"] in const}
        pairs_with_true_when = set()
        if env:
            for f, t, tg in cell.arcs():
                if tg.when is not None:
                    try:
                        if boolfunc.eval_function(tg.when, env) == 1:
                            pairs_with_true_when.add((f, t))
                    except boolfunc.FuncError:
                        pass
        for f, t, tg in cell.arcs():
            u = node_of[f"{inst.name}/{f}"]; o = node_of[f"{inst.name}/{t}"]
            a_src.append(u); a_dst.append(o)
            a_type.append(ARC_TYPE_ID[tg.timing_type])
            a_sense.append(SENSE_ID.get(tg.timing_sense or "non_unate", 2))
            a_mask.append(rf_mask(tg.timing_type, tg.timing_sense))
            w = tg.when if tg.when is not None else NO_WHEN
            a_when.append(vocab.when[w])
            a_key.append(vocab.arc_key[(inst.master, f, t, tg.timing_type, w)])
            a_inst.append(ii)
            enabled = (u not in const) and (o not in const)
            if enabled and tg.when is not None and env:
                try:
                    if boolfunc.eval_function(tg.when, env) == 0:
                        enabled = False; n_when_disabled += 1
                except boolfunc.FuncError:
                    pass
            if enabled and tg.when is None and (f, t) in pairs_with_true_when:
                enabled = False; n_default_disabled += 1
            fn = cell.pins[t].function
            if enabled and env and fn is not None and cell.pins[f].direction == "input":
                try:
                    if f in boolfunc.variables(boolfunc.parse_function(fn)) and not boolfunc.depends_on(fn, f, env):
                        enabled = False; n_collapsed += 1
                except boolfunc.FuncError:
                    pass
            a_enabled.append(enabled)
            per_master_arcs[inst.master] += 1
    rep.per_master_arcs = dict(per_master_arcs)

    c_src, c_dst, c_tables, c_axes = [], [], [], []
    for ii in registers:
        inst = nl.instances[ii]; cell = lib.cells[inst.master]
        for clk, d, tg in cell.checks("setup_rising"):
            tabs, axes = [], []
            for name in CONSTRAINT_TABLES:
                tb = tg.tables.get(name)
                if tb is None:
                    raise BuildError(f"{inst.master}: setup_rising without {name}")
                tabs.append(tb.values); axes.append(np.stack([tb.axis1, tb.axis2]))
            c_src.append(node_of[f"{inst.name}/{clk}"]); c_dst.append(node_of[f"{inst.name}/{d}"])
            c_tables.append(np.stack(tabs)); c_axes.append(np.stack(axes))

    succ = [[] for _ in range(N)]
    indeg = np.zeros(N, dtype=np.int64)
    for u, v in zip(src_net + a_src, dst_net + a_dst):
        succ[u].append(v); indeg[v] += 1
    level = np.zeros(N, dtype=np.int64)
    queue = collections.deque(int(v) for v in np.flatnonzero(indeg == 0))
    seen = 0
    while queue:
        u = queue.popleft(); seen += 1
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1); indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if seen != N:
        raise BuildError(f"combinational loop: {N - seen} pins unreachable in topological order")

    def T(a, dtype):
        return torch.tensor(np.asarray(a), dtype=dtype)

    x_num = np.stack([cap_rise, cap_fall, load_rise, load_fall, fanout, input_delay, output_delay, is_constant, is_clock_port], 1)
    n7 = len(vocab.slew_axis)
    data = Data(
        num_nodes=N,
        x_cat=T(np.stack([role, master_id, pinname_id], 1), torch.long),
        x_num=T(x_num, torch.float32),
        x_erc=T(np.stack([max_tr, max_cap], 1), torch.float32),
        g=torch.tensor([[sdc.period]], dtype=torch.float32),
        pin_instance=T(pin_instance, torch.long),
        level=T(level, torch.long),
        edge_index_net=T(np.array([src_net, dst_net]).reshape(2, -1), torch.long),
        edge_index_arc=T(np.array([a_src, a_dst]).reshape(2, -1), torch.long),
        arc_type=T(a_type, torch.long), arc_sense=T(a_sense, torch.long),
        arc_rf_mask=T(np.array(a_mask, dtype=bool).reshape(-1, 4), torch.bool),
        arc_when_present=T(np.array(a_when) != 0, torch.bool), arc_when_id=T(a_when, torch.long),
        arc_key_id=T(a_key, torch.long), arc_instance=T(a_inst, torch.long),
        arc_enabled=T(a_enabled, torch.bool),
        edge_index_check=T(np.array([c_src, c_dst]).reshape(2, -1), torch.long),
        check_tables=T(np.stack(c_tables) if c_tables else np.zeros((0, 2, n7, n7)), torch.float32),
        check_axes=T(np.stack(c_axes) if c_axes else np.zeros((0, 2, n7)), torch.float32),
        slew_axis=T(vocab.slew_axis, torch.float32),
    )
    data.pin_names = pin_names
    data.instance_names = [i.name for i in nl.instances]
    data.instance_masters = [i.master for i in nl.instances]
    data.design = design; data.group = group; data.top = nl.top; data.corner = corner
    data.lib_sha256 = list(vocab.lib_sha256); data.clock_is_virtual = clock_is_virtual
    data.schema_version = SCHEMA_VERSION
    data.x_num_cols = list(X_NUM_COLS)

    rep.counts = {
        "top": nl.top, "ports": len(nl.ports),
        "input_ports": int((role == ROLE["in_port"]).sum()), "output_ports": int((role == ROLE["out_port"]).sum()),
        "instances": n_inst, "masters": len(masters), "registers": len(registers),
        "escaped_instance_names": sum(i.escaped for i in nl.instances),
        "cell_pins": N - len(nl.ports), "cell_input_pins": int(np.isin(role, [ROLE["cell_in"], ROLE["reg_clk"], ROLE["reg_d"]]).sum()),
        "cell_output_pins": int(np.isin(role, [ROLE["cell_out"], ROLE["reg_q"]]).sum()),
        "nodes": N, "nets": len(driver), "unloaded_nets": len(unloaded), "port_to_port_nets": port_to_port,
        "net_edges": len(src_net), "arc_edges": len(a_src), "check_edges": len(c_src),
        "arc_edges_disabled": int(len(a_enabled) - sum(a_enabled)), "when_arcs_disabled": n_when_disabled,
        "default_arcs_disabled_by_true_when": n_default_disabled,
        "collapsed_arcs_disabled": n_collapsed, "tie_cells": sum(1 for i in nl.instances if lib.cells[i.master].is_tie),
        "constant_pins": int(is_constant.sum()), "partially_constant_cells": len(rep.partially_constant_cells),
        "levels": int(level.max()) + 1, "max_fanout": int(fanout.max()),
        "role_histogram": {ROLE_NAME[k]: int((role == k).sum()) for k in sorted(ROLE_NAME)},
        "period_ps": sdc.period, "clock_is_virtual": clock_is_virtual,
    }
    return data, rep
