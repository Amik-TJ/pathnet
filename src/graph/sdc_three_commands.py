from __future__ import annotations

import re
from dataclasses import dataclass, field


class SdcError(ValueError):
    pass


@dataclass
class Sdc:
    top: str | None
    clock_name: str
    period: float
    clock_port: str | None
    input_delay: dict[str, float] = field(default_factory=dict)
    output_delay: dict[str, float] = field(default_factory=dict)
    source: str = ""


_CLOCK = re.compile(r"^create_clock\s+-name\s+(\S+)\s+-period\s+([\d.eE+-]+)(?:\s+\[get_ports\s+(?:\{\s*([^}]+?)\s*\}|(\S+?))\])?\s*$")
_DELAY = re.compile(r"^set_(input|output)_delay\s+([\d.eE+-]+)\s+-clock\s+\[get_clocks\s+(?:\{\s*([^}]+?)\s*\}|(\S+?))\]\s*(?:-add_delay\s+)?\[get_ports\s+(?:\{\s*([^}]+?)\s*\}|(\S+?))\]\s*$")


def _ports(text: str) -> list[str]:
    return text.split()


def parse_sdc(path: str) -> Sdc:
    top = None
    clocks = []
    in_d: dict[str, float] = {}
    out_d: dict[str, float] = {}
    clock_refs: set[str] = set()
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("current_design"):
            top = line.split()[1]
            continue
        m = _CLOCK.match(line)
        if m:
            clocks.append((m.group(1), float(m.group(2)), m.group(3) or m.group(4)))
            continue
        m = _DELAY.match(line)
        if m:
            kind, val, clk_b, clk_p, ports_b, ports_p = m.groups()
            clk = clk_b or clk_p
            ports = ports_b or ports_p
            clock_refs.add(clk)
            for p in _ports(ports):
                d = in_d if kind == "input" else out_d
                if p in d:
                    raise SdcError(f"{path}: port {p} has two {kind} delays")
                d[p] = float(val)
            continue
        raise SdcError(f"{path}: unsupported SDC line: {line!r}")
    if len(clocks) != 1:
        raise SdcError(f"{path}: expected exactly one create_clock, found {len(clocks)}")
    name, period, port = clocks[0]
    if clock_refs - {name}:
        raise SdcError(f"{path}: delays refer to clocks {sorted(clock_refs - {name})}, not {name}")
    return Sdc(top=top, clock_name=name, period=period, clock_port=port, input_delay=in_d, output_delay=out_d, source=path)
