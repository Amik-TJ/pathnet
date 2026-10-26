from __future__ import annotations

import re
from dataclasses import dataclass, field


class NetlistError(ValueError):
    pass


@dataclass
class Port:
    name: str
    direction: str
    bus: str | None
    index: int | None


@dataclass
class Instance:
    name: str
    master: str
    pins: dict[str, str] = field(default_factory=dict)
    escaped: bool = False


@dataclass
class Netlist:
    top: str
    ports: list[Port]
    instances: list[Instance]
    source: str
    n_attributes_stripped: int = 0

    @property
    def connections(self) -> list[tuple[str, str, str]]:
        return [(i.name, p, n) for i in self.instances for p, n in i.pins.items()]


_ATTR = re.compile(r"\(\*.*?\*\)", re.S)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_DECL = re.compile(r"^(input|output|inout)\s*(?:\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\])?\s*(.+)$", re.S)
_INST = re.compile(r"^([A-Za-z_][\w$]*)\s+(\\\S+\s|[A-Za-z_][\w$\.\[\]]*)\s*\((.*)\)\s*$", re.S)
_CONN = re.compile(r"\.([A-Za-z_]\w*)\s*\(\s*([^()]*?)\s*\)", re.S)
_LITERAL = re.compile(r"\b\d+'[bBhHdDoO]")


_ESC_BIT = re.compile(r"^(\S+)\s+\[(\d+)\]$")


def canon(name: str) -> str:
    name = name.strip()
    if name.startswith("\\"):
        name = name[1:].strip()
        m = _ESC_BIT.match(name)
        if m:
            name = f"{m.group(1)}[{m.group(2)}]"
    return name


def _split_names(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def parse_netlist(path: str) -> Netlist:
    raw = open(path).read()
    n_attr = len(_ATTR.findall(raw))
    text = _ATTR.sub(" ", raw)
    text = _BLOCK_COMMENT.sub(" ", text)
    text = _LINE_COMMENT.sub(" ", text)

    top = None
    ports: list[Port] = []
    port_dir: dict[str, str] = {}
    header_order: list[str] = []
    instances: list[Instance] = []
    seen_inst: set[str] = set()
    bus_range: dict[str, tuple[int, int]] = {}
    pending: list[tuple[Instance, str, str]] = []

    for stmt in text.split(";"):
        s = stmt.strip()
        if not s:
            continue
        if s.startswith("module"):
            m = re.match(r"module\s+(\S+)\s*\((.*)\)\s*$", s, re.S)
            if not m:
                raise NetlistError(f"cannot parse module header: {s[:80]!r}")
            top = canon(m.group(1))
            header_order = [canon(x) for x in _split_names(m.group(2))]
            continue
        if s.startswith("endmodule"):
            continue
        if s.startswith("wire") or s.startswith("reg"):
            m = re.match(r"^(?:wire|reg)\s*\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]\s*(.+)$", s, re.S)
            if m:
                for nm in _split_names(m.group(3)):
                    bus_range.setdefault(canon(nm), (min(int(m.group(1)), int(m.group(2))), max(int(m.group(1)), int(m.group(2)))))
            continue
        if s.startswith("assign"):
            raise NetlistError(f"`assign` present; not a flow 1_2_yosys.v netlist: {s[:80]!r}")
        m = _DECL.match(s)
        if m:
            direction, hi, lo, names = m.groups()
            if direction == "inout":
                raise NetlistError(f"inout port: {s[:80]!r}")
            for nm in _split_names(names):
                nm = canon(nm)
                if hi is None:
                    ports.append(Port(nm, direction, None, None))
                    port_dir[nm] = direction
                else:
                    a, b = int(hi), int(lo)
                    bus_range.setdefault(nm, (min(a, b), max(a, b)))
                    for i in range(min(a, b), max(a, b) + 1):
                        ports.append(Port(f"{nm}[{i}]", direction, nm, i))
                        port_dir[f"{nm}[{i}]"] = direction
            continue
        m = _INST.match(s)
        if not m:
            raise NetlistError(f"unrecognised statement: {s[:120]!r}")
        master, name_raw, body = m.groups()
        if "{" in body or "}" in body:
            raise NetlistError(f"concatenation in connection of {canon(name_raw)}")
        if _LITERAL.search(body):
            raise NetlistError(f"constant literal in connection of {canon(name_raw)} (expected tie cells)")
        inst = Instance(canon(name_raw), master, escaped=name_raw.startswith("\\"))
        if inst.name in seen_inst:
            raise NetlistError(f"duplicate instance name {inst.name}")
        seen_inst.add(inst.name)
        conns = _CONN.findall(body)
        rest = _CONN.sub("", body).replace(",", "").strip()
        if rest:
            raise NetlistError(f"unparsed text in connections of {inst.name}: {rest[:60]!r}")
        for pin, net in conns:
            net = canon(net)
            if not net:
                raise NetlistError(f"unconnected pin .{pin}() on {inst.name}")
            if pin in inst.pins:
                raise NetlistError(f"pin {pin} connected twice on {inst.name}")
            inst.pins[pin] = net
            pending.append((inst, pin, net))
        instances.append(inst)

    if top is None:
        raise NetlistError("no module statement")
    for inst, pin, net in pending:
        if "[" not in net and net in bus_range:
            lo, hi = bus_range[net]
            if lo != hi:
                raise NetlistError(f"{inst.name}.{pin} connects the whole {hi-lo+1}-bit bus {net}")
            inst.pins[pin] = f"{net}[{lo}]"
    declared_bases = {p.bus or p.name for p in ports}
    missing = [h for h in header_order if h not in declared_bases]
    if missing:
        raise NetlistError(f"ports in header without declaration: {missing[:5]}")
    return Netlist(top=top, ports=ports, instances=instances, source=path, n_attributes_stripped=n_attr)


def summarize(nl: Netlist) -> dict:
    n_in = sum(p.direction == "input" for p in nl.ports)
    return {
        "top": nl.top,
        "ports": len(nl.ports), "input_ports": n_in, "output_ports": len(nl.ports) - n_in,
        "instances": len(nl.instances),
        "masters": len({i.master for i in nl.instances}),
        "connections": sum(len(i.pins) for i in nl.instances),
        "escaped_instance_names": sum(i.escaped for i in nl.instances),
    }
