"""Parser for the Python-ish concept expression syntax.

    Double(value=Last(Number(GivenBy(person=Greg, recipient=User))))
    MostAdorable(items) := Maximum(collection=items, by=AdorablenessScore())

Lowercase bare names parse as `Var`, capitalised bare names parse as a
zero-argument `Call`. That is the whole convention: concepts are capitalised,
bindings are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .expr import Arg, Call, Expr, Lit, Seq, Var

__all__ = ["parse", "parse_definition", "ParseError", "looks_like_expression"]


class ParseError(ValueError):
    pass


@dataclass
class _Token:
    kind: str
    value: str
    pos: int


_SYMBOLS = {
    "(": "lparen",
    ")": "rparen",
    "[": "lbracket",
    "]": "rbracket",
    ",": "comma",
    "=": "equals",
}


def _tokenize(src: str) -> List[_Token]:
    tokens: List[_Token] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src.startswith(":=", i):
            tokens.append(_Token("define", ":=", i))
            i += 2
            continue
        if c in _SYMBOLS:
            tokens.append(_Token(_SYMBOLS[c], c, i))
            i += 1
            continue
        if c in "\"'":
            quote = c
            i += 1
            buf = []
            while i < n and src[i] != quote:
                if src[i] == "\\" and i + 1 < n:
                    esc = src[i + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r"}.get(esc, esc))
                    i += 2
                    continue
                buf.append(src[i])
                i += 1
            if i >= n:
                raise ParseError("unterminated string")
            i += 1
            tokens.append(_Token("string", "".join(buf), i))
            continue
        if c.isdigit() or (c == "-" and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            while j < n and (src[j].isdigit() or src[j] == "." or src[j] == "_"):
                j += 1
            tokens.append(_Token("number", src[i:j].replace("_", ""), i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_.-"):
                j += 1
            tokens.append(_Token("name", src[i:j], i))
            i = j
            continue
        raise ParseError("unexpected character %r at %d" % (c, i))
    tokens.append(_Token("end", "", len(src)))
    return tokens


class _Parser:
    def __init__(self, tokens: List[_Token]) -> None:
        self.tokens = tokens
        self.i = 0

    @property
    def cur(self) -> _Token:
        return self.tokens[self.i]

    def eat(self, kind: str) -> _Token:
        tok = self.cur
        if tok.kind != kind:
            raise ParseError("expected %s but got %r" % (kind, tok.value or "end"))
        self.i += 1
        return tok

    def accept(self, kind: str) -> Optional[_Token]:
        if self.cur.kind == kind:
            self.i += 1
            return self.tokens[self.i - 1]
        return None

    def expr(self) -> Expr:
        tok = self.cur
        if tok.kind == "number":
            self.i += 1
            value = float(tok.value) if "." in tok.value else int(tok.value)
            return Lit(value)
        if tok.kind == "string":
            self.i += 1
            return Lit(tok.value)
        if tok.kind == "lbracket":
            self.i += 1
            items: List[Expr] = []
            while self.cur.kind != "rbracket":
                items.append(self.expr())
                if not self.accept("comma"):
                    break
            self.eat("rbracket")
            return Seq(tuple(items))
        if tok.kind == "name":
            self.i += 1
            name = tok.value
            if self.cur.kind == "lparen":
                self.i += 1
                args = self.arglist()
                self.eat("rparen")
                return Call(name, tuple(args))
            lowered = name.lower()
            if lowered in ("true", "false"):
                return Lit(lowered == "true")
            if lowered in ("none", "nothing", "null"):
                return Lit(None)
            if name[:1].isupper():
                return Call(name, ())
            return Var(name)
        raise ParseError("unexpected %r" % (tok.value or "end of input"))

    def arglist(self) -> List[Arg]:
        args: List[Arg] = []
        while self.cur.kind != "rparen":
            if self.cur.kind == "name" and self.tokens[self.i + 1].kind == "equals":
                name = self.eat("name").value
                self.eat("equals")
                args.append(Arg(name, self.expr()))
            else:
                args.append(Arg(None, self.expr()))
            if not self.accept("comma"):
                break
        return args


def parse(source: str) -> Expr:
    """Parse a single concept expression."""
    p = _Parser(_tokenize(source))
    expr = p.expr()
    if p.cur.kind != "end":
        raise ParseError("trailing input at %r" % p.cur.value)
    return expr


def parse_definition(source: str) -> Tuple[Expr, Expr]:
    """Parse `Head(params) := Body`, returning (head, body)."""
    if ":=" not in source:
        raise ParseError("not a definition, expected ':='")
    p = _Parser(_tokenize(source))
    head = p.expr()
    p.eat("define")
    body = p.expr()
    if p.cur.kind != "end":
        raise ParseError("trailing input at %r" % p.cur.value)
    if not isinstance(head, Call):
        raise ParseError("the left side of ':=' must be a concept call")
    return head, body


def looks_like_expression(source: str) -> bool:
    """Cheap check for whether a user typed raw concept syntax at us."""
    s = source.strip()
    if not s or " " in s.split("(")[0].strip():
        return False
    if "(" not in s or not s.endswith((")", "]")):
        return False
    head = s.split("(")[0]
    return bool(head) and head[0].isupper() and head.replace("_", "").isalnum()
