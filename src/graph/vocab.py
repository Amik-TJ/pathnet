from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .liberty_nldm import DELAY_TABLES, Library

PORT_TOKEN = "<PORT>"
NO_WHEN = "<none>"
ARC_TYPE_ID = {"combinational": 0, "rising_edge": 1, "combinational_rise": 2, "combinational_fall": 3}
SENSE_ID = {"positive_unate": 0, "negative_unate": 1, "non_unate": 2}
RF_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


def rf_mask(timing_type: str, timing_sense: str | None) -> list[bool]:
    if timing_type == "rising_edge":
        pairs = {(0, 0), (0, 1)}
    elif timing_sense == "positive_unate":
        pairs = {(0, 0), (1, 1)}
    elif timing_sense == "negative_unate":
        pairs = {(0, 1), (1, 0)}
    else:
        pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if timing_type == "combinational_rise":
        pairs = {p for p in pairs if p[1] == 0}
    elif timing_type == "combinational_fall":
        pairs = {p for p in pairs if p[1] == 1}
    return [p in pairs for p in RF_PAIRS]


@dataclass
class Vocab:
    corner: str
    cell_master: dict[str, int]
    pin_name: dict[str, int]
    when: dict[str, int]
    arc_key: dict[tuple, int]
    arc_key_meta: list[tuple]
    arc_bank: np.ndarray
    arc_cap_axis: np.ndarray
    arc_table_present: np.ndarray
    slew_axis: np.ndarray
    lib_files: list[str] = field(default_factory=list)
    lib_sha256: list[str] = field(default_factory=list)


def build_vocab(lib: Library, corner: str) -> Vocab:
    cells = sorted(lib.cells)
    cell_master = {PORT_TOKEN: 0}
    for c in cells:
        cell_master[c] = len(cell_master)
    pin_name = {PORT_TOKEN: 0}
    for c in cells:
        for p in lib.cells[c].pins:
            if p not in pin_name:
                pin_name[p] = len(pin_name)
    when = {NO_WHEN: 0}
    arc_key: dict[tuple, int] = {}
    meta, bank, caps, present = [], [], [], []
    n = len(lib.slew_axis)
    for c in cells:
        for f, t, tg in lib.cells[c].arcs():
            w = tg.when if tg.when is not None else NO_WHEN
            if w not in when:
                when[w] = len(when)
            key = (c, f, t, tg.timing_type, w)
            if key in arc_key:
                continue
            arc_key[key] = len(meta)
            meta.append(key)
            tabs = np.zeros((4, n, n)); pres = np.zeros(4, dtype=bool); cap = None
            for k, name in enumerate(DELAY_TABLES):
                tb = tg.tables.get(name)
                if tb is None:
                    continue
                if tb.values.shape != (n, n):
                    raise ValueError(f"{key}: table {name} shape {tb.values.shape}")
                tabs[k] = tb.values; pres[k] = True
                if cap is None:
                    cap = tb.axis2
                elif not np.allclose(cap, tb.axis2):
                    raise ValueError(f"{key}: tables of one timing group have different cap axes")
            if cap is None:
                raise ValueError(f"{key}: no delay table")
            bank.append(tabs); caps.append(cap); present.append(pres)
    return Vocab(corner, cell_master, pin_name, when, arc_key, meta, np.stack(bank), np.stack(caps),
                 np.stack(present), np.asarray(lib.slew_axis, dtype=np.float64), list(lib.files), list(lib.sha256))


def save_vocab(v: Vocab, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"vocab_{v.corner}.json"), "w") as f:
        json.dump({
            "corner": v.corner, "cell_master": v.cell_master, "pin_name": v.pin_name, "when": v.when,
            "arc_key_meta": [list(m) for m in v.arc_key_meta], "lib_files": v.lib_files, "lib_sha256": v.lib_sha256,
            "arc_type_id": ARC_TYPE_ID, "sense_id": SENSE_ID, "rf_pairs": RF_PAIRS, "delay_tables": DELAY_TABLES,
        }, f, indent=1)
    torch.save({"arc_bank": torch.tensor(v.arc_bank, dtype=torch.float32),
                "arc_cap_axis": torch.tensor(v.arc_cap_axis, dtype=torch.float32),
                "arc_table_present": torch.tensor(v.arc_table_present),
                "slew_axis": torch.tensor(v.slew_axis, dtype=torch.float32)},
               os.path.join(out_dir, f"arc_bank_{v.corner}.pt"))


def load_vocab(out_dir: str, corner: str) -> Vocab:
    with open(os.path.join(out_dir, f"vocab_{corner}.json")) as f:
        j = json.load(f)
    bank = torch.load(os.path.join(out_dir, f"arc_bank_{corner}.pt"), weights_only=True)
    meta = [tuple(m) for m in j["arc_key_meta"]]
    return Vocab(corner, j["cell_master"], j["pin_name"], j["when"], {m: i for i, m in enumerate(meta)}, meta,
                 bank["arc_bank"].double().numpy(), bank["arc_cap_axis"].double().numpy(),
                 bank["arc_table_present"].numpy(), bank["slew_axis"].double().numpy(), j["lib_files"], j["lib_sha256"])


def attach_tables(data, vocab: Vocab):
    key = data.arc_key_id.long()
    data.arc_tables = torch.tensor(vocab.arc_bank, dtype=torch.float32)[key]
    data.arc_cap_axis = torch.tensor(vocab.arc_cap_axis, dtype=torch.float32)[key]
    return data
