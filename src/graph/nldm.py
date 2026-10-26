from __future__ import annotations

import numpy as np


def bracket(axis: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = axis.shape[-1]
    if n < 2:
        return np.zeros(np.shape(x), dtype=np.int64)
    if axis.ndim == 1:
        i = np.searchsorted(axis, x, side="right") - 1
    else:
        i = (axis <= np.asarray(x)[..., None]).sum(-1) - 1
    return np.clip(i, 0, n - 2)


def lookup(table: np.ndarray, axis1: np.ndarray, axis2: np.ndarray, s, c) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    axis1 = np.asarray(axis1, dtype=np.float64)
    axis2 = np.asarray(axis2, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n1, n2 = table.shape[-2], table.shape[-1]
    if n1 == 1 and n2 == 1:
        return np.broadcast_to(table[..., 0, 0], np.broadcast(s, c).shape).copy()
    if table.ndim == 2:
        i = bracket(axis1, s)
        j = bracket(axis2, c)
        a1 = axis1
        a2 = axis2
        t00 = table[i, j]; t01 = table[i, j + 1]; t10 = table[i + 1, j]; t11 = table[i + 1, j + 1]
        u = (s - a1[i]) / (a1[i + 1] - a1[i]) if n1 > 1 else 0.0
        v = (c - a2[j]) / (a2[j + 1] - a2[j]) if n2 > 1 else 0.0
    else:
        E = table.shape[0]
        idx = np.arange(E)
        a1 = axis1 if axis1.ndim == 2 else np.broadcast_to(axis1, (E, n1))
        a2 = axis2 if axis2.ndim == 2 else np.broadcast_to(axis2, (E, n2))
        i = bracket(a1, s)
        j = bracket(a2, c)
        t00 = table[idx, i, j]; t01 = table[idx, i, j + 1]; t10 = table[idx, i + 1, j]; t11 = table[idx, i + 1, j + 1]
        u = (s - a1[idx, i]) / (a1[idx, i + 1] - a1[idx, i])
        v = (c - a2[idx, j]) / (a2[idx, j + 1] - a2[idx, j])
    if n1 == 1:
        return (1 - v) * t00 + v * t01
    if n2 == 1:
        return (1 - u) * t00 + u * t10
    return (1 - u) * ((1 - v) * t00 + v * t01) + u * ((1 - v) * t10 + v * t11)
