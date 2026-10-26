from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass, field

import numpy as np

from paths import LIBERTY

PARSER_VERSION = 3

ARC_TYPES = ("combinational", "combinational_rise", "combinational_fall", "rising_edge")
CHECK_TYPES_USED = ("setup_rising",)
PARSED_BUT_DROPPED = ("hold_rising", "min_pulse_width")
SUPPORTED_TIMING_TYPES = set(ARC_TYPES) | set(CHECK_TYPES_USED) | set(PARSED_BUT_DROPPED)
DELAY_TABLES = ("cell_rise", "cell_fall", "rise_transition", "fall_transition")
CONSTRAINT_TABLES = ("rise_constraint", "fall_constraint")
TABLE_GROUPS = set(DELAY_TABLES) | set(CONSTRAINT_TABLES)


class LibertyError(ValueError):
    pass


@dataclass
class Table:
    template: str
    axis1: np.ndarray
    axis2: np.ndarray
    values: np.ndarray


@dataclass
class TimingGroup:
    related_pin: str
    timing_type: str
    timing_sense: str | None
    when: str | None
    tables: dict[str, Table] = field(default_factory=dict)

    @property
    def is_arc(self) -> bool:
        return self.timing_type in ARC_TYPES


@dataclass
class Pin:
    name: str
    direction: str
    capacitance: float | None = None
    rise_capacitance: float | None = None
    fall_capacitance: float | None = None
    is_clock: bool = False
    max_transition: float | None = None
    max_capacitance: float | None = None
    function: str | None = None
    timing: list[TimingGroup] = field(default_factory=list)

    @property
    def cap_rise(self) -> float:
        v = self.rise_capacitance if self.rise_capacitance is not None else self.capacitance
        return 0.0 if v is None else v

    @property
    def cap_fall(self) -> float:
        v = self.fall_capacitance if self.fall_capacitance is not None else self.capacitance
        return 0.0 if v is None else v


@dataclass
class Cell:
    name: str
    area: float | None
    pins: dict[str, Pin]
    ff: tuple[str, str] | None
    has_latch: bool = False
    has_statetable: bool = False
    source: str = ""

    @property
    def output_pins(self) -> list[str]:
        return [p for p, pin in self.pins.items() if pin.direction == "output"]

    @property
    def input_pins(self) -> list[str]:
        return [p for p, pin in self.pins.items() if pin.direction == "input"]

    def arcs(self):
        for to_pin, pin in self.pins.items():
            if pin.direction != "output":
                continue
            for tg in pin.timing:
                if tg.is_arc:
                    yield tg.related_pin, to_pin, tg

    def checks(self, timing_type: str = "setup_rising"):
        for d_pin, pin in self.pins.items():
            for tg in pin.timing:
                if tg.timing_type == timing_type:
                    yield tg.related_pin, d_pin, tg

    def timing_types(self) -> set[str]:
        return {tg.timing_type for pin in self.pins.values() for tg in pin.timing}

    @property
    def is_tie(self) -> bool:
        return any(pin.direction == "output" and pin.function is not None and pin.function.strip() in ("0", "1")
                   for pin in self.pins.values())


@dataclass
class Library:
    cells: dict[str, Cell]
    templates: dict[str, dict]
    attrs: dict[str, str]
    files: list[str]
    sha256: list[str]
    slew_axis: np.ndarray | None = None
    parse_seconds: float = 0.0

    def supported_master(self, name: str) -> tuple[bool, str]:
        c = self.cells.get(name)
        if c is None:
            return False, "master not in library"
        if c.has_latch:
            return False, "latch"
        if c.has_statetable:
            return False, "statetable"
        bad = c.timing_types() - SUPPORTED_TIMING_TYPES
        if bad:
            return False, "timing_type " + ",".join(sorted(bad))
        return True, ""


_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_TOKEN = re.compile(r'\s*(?:("(?:[^"\\]|\\[\s\S])*")|([A-Za-z_][\w.\-\+\[\]/]*)|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|([(){}:;,\\]))')
_FLAG_ONLY = {"statetable", "latch"}

_KEEP = {
    "library": {"lu_table_template", "cell"},
    "cell": {"pin", "ff", "latch", "statetable", "bundle", "bus"},
    "pin": {"timing"},
    "timing": TABLE_GROUPS,
    "ff": set(), "latch": set(), "statetable": set(), "bundle": set(), "bus": set(),
    "lu_table_template": set(),
}
for _t in TABLE_GROUPS:
    _KEEP[_t] = set()


class _Stream:
    def __init__(self, text: str):
        self._it = _TOKEN.finditer(text)
        self._buf: list[str] = []
        self.count = 0

    def next(self) -> str | None:
        if self._buf:
            return self._buf.pop()
        for m in self._it:
            tok = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            if tok is None:
                continue
            if tok == "\\":
                continue
            self.count += 1
            return tok
        return None

    def push(self, tok: str):
        self._buf.append(tok)


def _unquote(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


def _skip_group(st: _Stream):
    depth = 1
    while depth:
        t = st.next()
        if t is None:
            raise LibertyError("unbalanced braces")
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1


def _parse_group(st: _Stream, name: str, args: list[str]) -> dict:
    node = {"name": name, "args": args, "attrs": {}, "groups": []}
    keep = _KEEP.get(name, set())
    while True:
        t = st.next()
        if t is None:
            raise LibertyError(f"EOF inside group {name}")
        if t == "}":
            return node
        key = t
        n = st.next()
        if n == ":":
            vals = [st.next()]
            if vals[0] is None:
                raise LibertyError("EOF in attribute")
            while True:
                v = st.next()
                if v is None:
                    raise LibertyError("EOF in attribute")
                if v == ";":
                    break
                if v == "}" or (v[0].isalpha() or v[0] == "_"):
                    st.push(v)
                    break
                vals.append(v)
            node["attrs"][key] = _unquote(vals[0]) if len(vals) == 1 else " ".join(vals)
        elif n == "(":
            gargs = []
            while True:
                v = st.next()
                if v is None:
                    raise LibertyError("EOF in args")
                if v == ")":
                    break
                if v != ",":
                    gargs.append(_unquote(v))
            after = st.next()
            if after == "{":
                if key in keep and key in _FLAG_ONLY:
                    node["groups"].append({"name": key, "args": gargs, "attrs": {}, "groups": []})
                    _skip_group(st)
                elif key in keep:
                    node["groups"].append(_parse_group(st, key, gargs))
                else:
                    _skip_group(st)
            elif after == ";":
                node["attrs"][key] = gargs
            else:
                raise LibertyError(f"unexpected token {after!r} after {key}(...)")
        else:
            raise LibertyError(f"unexpected token {n!r} after {key!r} in group {name}")


def _floats(s) -> np.ndarray:
    if isinstance(s, list):
        s = " ".join(s)
    return np.array([float(x) for x in re.split(r"[,\s]+", s.strip()) if x], dtype=np.float64)


def _read_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.read()
    with open(path) as f:
        return f.read()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _table(g: dict, templates: dict) -> Table:
    tname = g["args"][0] if g["args"] else "scalar"
    tpl = templates.get(tname, {})
    a1 = _floats(g["attrs"]["index_1"]) if "index_1" in g["attrs"] else tpl.get("index_1")
    a2 = _floats(g["attrs"]["index_2"]) if "index_2" in g["attrs"] else tpl.get("index_2")
    rows = g["attrs"].get("values")
    if rows is None:
        raise LibertyError(f"table {g['name']} without values")
    if isinstance(rows, str):
        rows = [rows]
    vals = np.array([_floats(r) for r in rows], dtype=np.float64)
    if tname == "scalar" or (a1 is None and a2 is None):
        vals = vals.reshape(1, 1)
        a1 = np.zeros(1); a2 = np.zeros(1)
    elif a2 is None:
        vals = vals.reshape(len(a1), 1); a2 = np.zeros(1)
    elif a1 is None:
        vals = vals.reshape(1, len(a2)); a1 = np.zeros(1)
    else:
        if vals.shape != (len(a1), len(a2)):
            raise LibertyError(f"table {g['name']} ({tname}) shape {vals.shape} != ({len(a1)}, {len(a2)})")
        v1, v2 = tpl.get("variable_1"), tpl.get("variable_2")
        if v1 and v2 and ("transition" in v2 and "capacitance" in v1):
            vals = vals.T; a1, a2 = a2, a1
    return Table(tname, a1, a2, vals)


def _bool(v) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def _float_or_none(v):
    return None if v is None else float(v)


def _extract(lib_node: dict, source: str, lib: "Library"):
    for g in lib_node["groups"]:
        if g["name"] == "lu_table_template":
            tname = g["args"][0]
            tpl = {"variable_1": g["attrs"].get("variable_1"), "variable_2": g["attrs"].get("variable_2")}
            for k in ("index_1", "index_2"):
                if k in g["attrs"]:
                    tpl[k] = _floats(g["attrs"][k])
            lib.templates[tname] = tpl
    for g in lib_node["groups"]:
        if g["name"] != "cell":
            continue
        cname = g["args"][0]
        pins: dict[str, Pin] = {}
        ff = None
        has_latch = has_state = False
        for cg in g["groups"]:
            if cg["name"] == "pin":
                for pname in cg["args"]:
                    a = cg["attrs"]
                    pin = Pin(
                        name=pname, direction=a.get("direction", "unknown"),
                        capacitance=_float_or_none(a.get("capacitance")),
                        rise_capacitance=_float_or_none(a.get("rise_capacitance")),
                        fall_capacitance=_float_or_none(a.get("fall_capacitance")),
                        is_clock=_bool(a.get("clock", "false")),
                        max_transition=_float_or_none(a.get("max_transition")),
                        max_capacitance=_float_or_none(a.get("max_capacitance")),
                        function=a.get("function"),
                    )
                    for tg in cg["groups"]:
                        if tg["name"] != "timing":
                            continue
                        ta = tg["attrs"]
                        rp = ta.get("related_pin")
                        if rp is None:
                            raise LibertyError(f"{cname}/{pname}: timing group without related_pin")
                        tables = {t["name"]: _table(t, lib.templates) for t in tg["groups"] if t["name"] in TABLE_GROUPS}
                        for rel in str(rp).split():
                            pin.timing.append(TimingGroup(
                                related_pin=rel,
                                timing_type=ta.get("timing_type", "combinational"),
                                timing_sense=ta.get("timing_sense"),
                                when=ta.get("when"),
                                tables=tables,
                            ))
                    pins[pname] = pin
            elif cg["name"] == "ff":
                ff = (cg["attrs"].get("clocked_on", "").strip('"'), cg["attrs"].get("next_state", "").strip('"'))
            elif cg["name"] == "latch":
                has_latch = True
            elif cg["name"] == "statetable":
                has_state = True
            elif cg["name"] in ("bundle", "bus"):
                raise LibertyError(f"{cname}: bundle/bus groups are not supported")
        lib.cells[cname] = Cell(cname, _float_or_none(g["attrs"].get("area")), pins, ff, has_latch, has_state, source)
    for k in ("time_unit", "capacitive_load_unit", "nom_voltage", "nom_temperature", "delay_model",
              "default_max_transition", "input_threshold_pct_rise", "slew_lower_threshold_pct_rise",
              "slew_upper_threshold_pct_rise", "slew_derate_from_library", "date", "revision"):
        if k in lib_node["attrs"] and k not in lib.attrs:
            v = lib_node["attrs"][k]
            lib.attrs[k] = " ".join(v) if isinstance(v, list) else v


def parse_liberty_text(text: str, source: str, lib: Library):
    text = _COMMENT.sub(" ", text)
    st = _Stream(text)
    t = st.next()
    if t != "library":
        raise LibertyError(f"{source}: expected `library`, got {t!r}")
    if st.next() != "(":
        raise LibertyError("expected ( after library")
    args = []
    while True:
        v = st.next()
        if v == ")":
            break
        if v != ",":
            args.append(v)
    if st.next() != "{":
        raise LibertyError("expected { after library(...)")
    node = _parse_group(st, "library", args)
    _extract(node, source, lib)
    return st.count


def parse_liberty_files(paths: list[str], cache_dir: str | None = None, verbose: bool = False) -> Library:
    resolved = [os.path.realpath(p) for p in paths]
    hashes = [_sha256(p) for p in resolved]
    key = hashlib.sha256(("|".join(hashes) + f"|v{PARSER_VERSION}").encode()).hexdigest()[:16]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, f"liberty_{key}.pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as f:
                lib = pickle.load(f)
            if verbose:
                print(f"[liberty] loaded cache {cpath} ({len(lib.cells)} cells)")
            return lib
    lib = Library(cells={}, templates={}, attrs={}, files=resolved, sha256=hashes)
    t0 = time.time()
    for p in resolved:
        t1 = time.time()
        n = parse_liberty_text(_read_text(p), p, lib)
        if verbose:
            print(f"[liberty] {os.path.basename(p)}: {n} tokens, {len(lib.cells)} cells so far, {time.time()-t1:.1f}s")
    lib.parse_seconds = time.time() - t0
    axes = {}
    for c in lib.cells.values():
        for _, _, tg in c.arcs():
            for name in DELAY_TABLES:
                tb = tg.tables.get(name)
                if tb is not None and tb.values.size > 1:
                    axes[tuple(tb.axis1.tolist())] = name
    if len(axes) != 1:
        raise LibertyError(f"delay tables use {len(axes)} different slew axes: {list(axes)[:3]}")
    lib.slew_axis = np.array(next(iter(axes)), dtype=np.float64)
    if cache_dir:
        with open(cpath, "wb") as f:
            pickle.dump(lib, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[liberty] cached -> {cpath}")
    return lib


def default_lib_files(corner: str = "BC", vt: str = "RVT", lib_dir: str | None = None) -> list[str]:
    tag = {"BC": "FF", "TC": "TT", "WC": "SS"}[corner]
    if lib_dir is None:
        lib_dir = LIBERTY
    return [os.path.join(lib_dir, f) for f in (
        f"asap7sc7p5t_AO_{vt}_{tag}_nldm_211120.lib.gz",
        f"asap7sc7p5t_INVBUF_{vt}_{tag}_nldm_220122.lib.gz",
        f"asap7sc7p5t_OA_{vt}_{tag}_nldm_211120.lib.gz",
        f"asap7sc7p5t_SIMPLE_{vt}_{tag}_nldm_211120.lib.gz",
        f"asap7sc7p5t_SEQ_{vt}_{tag}_nldm_220123.lib",
    )]
