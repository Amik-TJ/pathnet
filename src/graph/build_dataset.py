from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

import torch

from paths import DATA

from .build_pin_graph import BuildError, build_graph
from .labels import LabelError, attach_labels, read_sweep
from .liberty_nldm import LibertyError, default_lib_files, parse_liberty_files
from .sdc_three_commands import SdcError, parse_sdc
from .vocab import build_vocab, load_vocab, save_vocab
from .yosys_verilog import NetlistError, parse_netlist

DEFAULT_SWEEP = f"{DATA}/sta_post_synth_BC"
DEFAULT_OUT = f"{DATA}/graphs_BC"


def load_or_build_vocab(lib_files, corner, out_root, verbose):
    vpath = os.path.join(out_root, f"vocab_{corner}.json")
    lib = parse_liberty_files(lib_files, cache_dir=os.path.join(out_root, "cache"), verbose=verbose)
    if os.path.exists(vpath):
        vocab = load_vocab(out_root, corner)
        if vocab.lib_sha256 == lib.sha256:
            return lib, vocab
        print("[vocab] library hashes changed; rebuilding vocab", file=sys.stderr)
    vocab = build_vocab(lib, corner)
    save_vocab(vocab, out_root)
    if verbose:
        print(f"[vocab] {len(vocab.cell_master)-1} cells, {len(vocab.pin_name)-1} pin names, {len(vocab.when)-1} when strings, "
              f"{len(vocab.arc_key_meta)} arc keys -> {vpath}")
    return lib, vocab


def build_one(netlist_path, sdc_path, lib, vocab, *, design, group, corner, label_dir=None, out_dir=None, strict_labels=True):
    t0 = time.time()
    row = {"group": group, "design": design, "status": "ok", "reason": "", "seconds": 0.0}
    try:
        nl = parse_netlist(netlist_path)
        sdc = parse_sdc(sdc_path)
        data, rep = build_graph(nl, lib, sdc, vocab, design=design, group=group, corner=corner)
        rep_dict = {"counts": rep.counts, "per_master_arcs": rep.per_master_arcs,
                    "partially_constant_cells": rep.partially_constant_cells[:50], "warnings": rep.warnings,
                    "netlist": netlist_path, "sdc": sdc_path, "lib_files": vocab.lib_files, "lib_sha256": vocab.lib_sha256}
        if label_dir is not None:
            sweep = read_sweep(label_dir)
            rep_dict["labels"] = attach_labels(data, sweep, strict=strict_labels)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(data, os.path.join(out_dir, "graph.pt"))
            with open(os.path.join(out_dir, "build_report.json"), "w") as f:
                json.dump(rep_dict, f, indent=1, default=str)
        row.update({k: rep.counts[k] for k in ("nodes", "instances", "registers", "net_edges", "arc_edges", "levels", "constant_pins")})
        if "labels" in rep_dict:
            row["paths"] = rep_dict["labels"]["paths"]; row["tied"] = rep_dict["labels"]["tied_worst_endpoints"]
        row["seconds"] = round(time.time() - t0, 2)
        return data, rep_dict, row
    except (NetlistError, SdcError, BuildError, LabelError, LibertyError) as e:
        row.update({"status": "skipped", "reason": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 2)})
        return None, None, row
    except Exception as e:
        row.update({"status": "error", "reason": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 2)})
        row["traceback"] = traceback.format_exc()
        return None, None, row


def read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--netlist")
    ap.add_argument("--sdc")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--corner", default="BC", choices=["BC", "TC", "WC"])
    ap.add_argument("--lib-dir", default=None)
    ap.add_argument("--sweep-root", default=DEFAULT_SWEEP)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--lenient-labels", action="store_true")
    ap.add_argument("--raise-errors", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--engine", default="cpp", choices=["cpp", "python"])
    a = ap.parse_args(argv)
    out_root = a.out_root or f"{DATA}/graphs_{a.corner}"
    os.makedirs(out_root, exist_ok=True)
    verbose = not a.quiet
    lib, vocab = load_or_build_vocab(default_lib_files(a.corner, lib_dir=a.lib_dir), a.corner, out_root, verbose)
    fast = None
    if a.engine == "cpp":
        from fast_parser.fast_graph import FastBuilder
        fast = FastBuilder(lib, vocab)

    jobs = []
    if a.netlist:
        if not a.sdc:
            ap.error("--netlist needs --sdc")
        jobs.append(dict(netlist_path=a.netlist, sdc_path=a.sdc, design=a.name, group="custom",
                         label_dir=None if a.no_labels else a.labels, out_dir=os.path.join(out_root, "custom", a.name)))
    else:
        rows = read_manifest(a.manifest or os.path.join(a.sweep_root, "manifest.tsv"))
        by_key = {f"{r['group']}/{r['design']}": r for r in rows}
        keys = list(by_key) if a.all else a.designs
        keys = keys[a.offset:(a.offset + a.limit) if a.limit else None]
        for k in keys:
            r = by_key[k]
            ddir = os.path.join(a.sweep_root, r["group"], r["design"])
            jobs.append(dict(netlist_path=r["netlist"], sdc_path=os.path.join(ddir, "sta.sdc"), design=r["design"], group=r["group"],
                             label_dir=None if a.no_labels else ddir, out_dir=os.path.join(out_root, r["group"], r["design"])))
    summary_path = os.path.join(out_root, "build_summary.tsv")
    fields = ["group", "design", "status", "reason", "seconds", "nodes", "instances", "registers", "net_edges", "arc_edges", "levels", "constant_pins", "paths", "tied"]
    new_file = not os.path.exists(summary_path)
    n_ok = 0
    with open(summary_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if new_file:
            w.writeheader()
        for j in jobs:
            kw = dict(design=j["design"], group=j["group"], corner=a.corner, label_dir=j["label_dir"], out_dir=j["out_dir"],
                      strict_labels=not a.lenient_labels)
            if fast is not None:
                data, rep, row = fast.build_one(j["netlist_path"], j["sdc_path"], **kw)
            else:
                data, rep, row = build_one(j["netlist_path"], j["sdc_path"], lib, vocab, **kw)
            w.writerow(row); f.flush()
            n_ok += row["status"] == "ok"
            if verbose or row["status"] != "ok":
                print(f"[{row['status']:7s}] {j['group']}/{j['design']} {row['seconds']}s {row.get('reason','')}"
                      + (f" nodes={row['nodes']} arcs={row['arc_edges']} levels={row['levels']} paths={row.get('paths','-')}" if row["status"] == "ok" else ""))
            if row["status"] == "error" and a.raise_errors:
                print(row["traceback"], file=sys.stderr); sys.exit(1)
    print(f"done: {n_ok}/{len(jobs)} ok; summary {summary_path}")


if __name__ == "__main__":
    main()
