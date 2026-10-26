import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO, path))


with open(os.path.join(REPO, "configs", "paths.json")) as _f:
    _C = json.load(_f)

DATA = resolve(_C["data_root"])
LIBERTY = resolve(_C["liberty_dir"])
RUNS = resolve(_C["runs_root"])
