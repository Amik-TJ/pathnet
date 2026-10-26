from __future__ import annotations

import re

_TOK = re.compile(r"\s*(?:([A-Za-z_][\w\[\]]*)|([()!'*&^+|]))")


class FuncError(ValueError):
    pass


def tokenize(expr: str) -> list[str]:
    out, pos = [], 0
    while pos < len(expr):
        m = _TOK.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip() == "":
                break
            raise FuncError(f"cannot tokenize function {expr!r} at {pos}")
        out.append(m.group(1) or m.group(2))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        node = self.or_()
        if self.peek() is not None:
            raise FuncError(f"trailing tokens {self.t[self.i:]}")
        return node

    def or_(self):
        node = self.xor()
        while self.peek() in ("+", "|"):
            self.take()
            node = ("or", node, self.xor())
        return node

    def xor(self):
        node = self.and_()
        while self.peek() == "^":
            self.take()
            node = ("xor", node, self.and_())
        return node

    def and_(self):
        node = self.unary()
        while True:
            p = self.peek()
            if p in ("*", "&"):
                self.take()
                node = ("and", node, self.unary())
            elif p is not None and (p == "(" or p == "!" or re.match(r"[A-Za-z_]", p)):
                node = ("and", node, self.unary())
            else:
                return node

    def unary(self):
        p = self.peek()
        if p == "!":
            self.take()
            node = ("not", self.unary())
        elif p == "(":
            self.take()
            node = self.or_()
            if self.take() != ")":
                raise FuncError("missing )")
        elif p is not None and re.match(r"[A-Za-z_]", p):
            node = ("var", self.take())
        else:
            raise FuncError(f"unexpected token {p!r}")
        while self.peek() == "'":
            self.take()
            node = ("not", node)
        return node


_cache: dict[str, tuple] = {}


def parse_function(expr: str):
    if expr not in _cache:
        _cache[expr] = _Parser(tokenize(expr)).parse()
    return _cache[expr]


def variables(node, acc=None) -> set[str]:
    acc = set() if acc is None else acc
    if node[0] == "var":
        acc.add(node[1])
    else:
        for ch in node[1:]:
            variables(ch, acc)
    return acc


def evaluate(node, env: dict[str, int | None]):
    k = node[0]
    if k == "var":
        return env.get(node[1])
    if k == "not":
        v = evaluate(node[1], env)
        return None if v is None else 1 - v
    a = evaluate(node[1], env)
    b = evaluate(node[2], env)
    if k == "and":
        if a == 0 or b == 0:
            return 0
        if a is None or b is None:
            return None
        return 1
    if k == "or":
        if a == 1 or b == 1:
            return 1
        if a is None or b is None:
            return None
        return 0
    if k == "xor":
        if a is None or b is None:
            return None
        return a ^ b
    raise FuncError(k)


def eval_function(expr: str, env: dict[str, int | None]):
    return evaluate(parse_function(expr), env)


def depends_on(expr: str, var: str, env: dict[str, int | None]) -> bool:
    node = parse_function(expr)
    free = sorted(v for v in variables(node) if v != var and env.get(v) is None)
    if var not in variables(node):
        return False
    base = {k: v for k, v in env.items() if v is not None}
    for bits in range(1 << len(free)):
        e = dict(base)
        for i, v in enumerate(free):
            e[v] = (bits >> i) & 1
        e[var] = 0
        v0 = evaluate(node, e)
        e[var] = 1
        v1 = evaluate(node, e)
        if v0 != v1:
            return True
    return False
