"""Concept expressions.

The whole system speaks in these. Ears produce them, realizations consume and
produce them, mouth renders them back into English. They look like Python calls
because that gives us binding, arguments, nesting and composition for free, but
a `Call` is an expression of *meaning* first and an executable thing only if
some realization happens to make it one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Expr",
    "Lit",
    "Var",
    "Arg",
    "Call",
    "Seq",
    "call",
    "num",
    "text",
    "truth",
    "nothing",
    "render",
    "walk",
    "substitute",
    "match",
    "concept_names",
]


class Expr:
    """Base class for every node in a concept expression."""

    __slots__ = ()

    def children(self) -> Sequence["Expr"]:
        return ()

    def __str__(self) -> str:
        return render(self, multiline=False)


@dataclass(frozen=True)
class Lit(Expr):
    """A literal value: number, text, truth, or nothing.

    Conceptually these are still concepts (`Number`, `Text`, `Truth`), we just
    do not make you spell `Six()` out longhand.
    """

    value: Any

    def kind(self) -> str:
        v = self.value
        if v is None:
            return "Nothing"
        if isinstance(v, bool):
            return "Truth"
        if isinstance(v, (int, float)):
            return "Number"
        return "Text"


@dataclass(frozen=True)
class Var(Expr):
    """A bound name. Created by `Let`, by pattern matching, or by the ears."""

    name: str


@dataclass(frozen=True)
class Arg:
    name: Optional[str]
    value: Expr

    def __str__(self) -> str:
        if self.name:
            return "%s=%s" % (self.name, self.value)
        return str(self.value)


@dataclass(frozen=True)
class Call(Expr):
    """A concept applied to arguments. `Double(value=6)`, `Greg()`, `Spooky()`."""

    concept: str
    args: Tuple[Arg, ...] = ()

    def children(self) -> Sequence[Expr]:
        return tuple(a.value for a in self.args)

    # -- argument access ---------------------------------------------------
    def get(self, name: str, default: Optional[Expr] = None) -> Optional[Expr]:
        for a in self.args:
            if a.name == name:
                return a.value
        return default

    def has(self, name: str) -> bool:
        return any(a.name == name for a in self.args)

    def positional(self) -> Tuple[Expr, ...]:
        return tuple(a.value for a in self.args if a.name is None)

    def pos(self, index: int, default: Optional[Expr] = None) -> Optional[Expr]:
        p = self.positional()
        if index < len(p):
            return p[index]
        return default

    def first(self, *names: str) -> Optional[Expr]:
        """The argument a realization means, however the ears labelled it.

        Tries the given keyword names, then the first positional argument,
        then falls back to the only argument if there is exactly one. A
        one-argument concept has no ambiguity to protect.
        """
        for n in names:
            v = self.get(n)
            if v is not None:
                return v
        p = self.positional()
        if p:
            return p[0]
        if len(self.args) == 1:
            return self.args[0].value
        return None

    def arg_names(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.args if a.name)

    def with_args(self, args: Sequence[Arg]) -> "Call":
        return Call(self.concept, tuple(args))


@dataclass(frozen=True)
class Seq(Expr):
    """An ordered collection literal."""

    items: Tuple[Expr, ...] = ()

    def children(self) -> Sequence[Expr]:
        return self.items


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def call(_concept: str, /, *positional: Expr, **kwargs: Expr) -> Call:
    args: List[Arg] = [Arg(None, p) for p in positional]
    args.extend(Arg(k, v) for k, v in kwargs.items())
    return Call(_concept, tuple(args))


def num(value: Any) -> Lit:
    return Lit(value)


def text(value: str) -> Lit:
    return Lit(value)


def truth(value: bool) -> Lit:
    return Lit(bool(value))


def nothing() -> Lit:
    return Lit(None)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_MAX_INLINE = 58


def _render_lit(v: Any) -> str:
    if v is None:
        return "Nothing"
    if v is True:
        return "True"
    if v is False:
        return "False"
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return repr(round(v, 10))
    if isinstance(v, int):
        return str(v)
    return '"%s"' % (
        str(v)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render(expr: Expr, multiline: bool = True, indent: int = 0) -> str:
    """Render an expression in the Python-ish concept syntax."""
    pad = "    " * indent
    if isinstance(expr, Lit):
        return pad + _render_lit(expr.value) if indent else _render_lit(expr.value)
    if isinstance(expr, Var):
        return (pad if indent else "") + expr.name
    if isinstance(expr, Seq):
        inner = ", ".join(render(i, multiline=False) for i in expr.items)
        flat = "[%s]" % inner
        if not multiline or len(flat) <= _MAX_INLINE:
            return (pad if indent else "") + flat
        lines = ["%s[" % (pad if indent else "")]
        for item in expr.items:
            lines.append(render(item, True, indent + 1) + ",")
        lines.append("%s]" % pad)
        return "\n".join(lines)
    if isinstance(expr, Call):
        head = (pad if indent else "") + expr.concept
        if not expr.args:
            return head + "()"
        flat = head + "(" + ", ".join(_flat_arg(a) for a in expr.args) + ")"
        if not multiline or len(flat) <= _MAX_INLINE:
            return flat
        lines = [head + "("]
        for a in expr.args:
            rendered = render(a.value, True, indent + 1)
            if a.name:
                stripped = rendered.lstrip()
                rendered = "    " * (indent + 1) + a.name + "=" + stripped
            lines.append(rendered + ",")
        lines.append(pad + ")")
        return "\n".join(lines)
    raise TypeError("cannot render %r" % (expr,))


def _flat_arg(a: Arg) -> str:
    v = render(a.value, multiline=False)
    return "%s=%s" % (a.name, v) if a.name else v


# ---------------------------------------------------------------------------
# traversal / transformation
# ---------------------------------------------------------------------------


def is_name(expr: Expr) -> bool:
    """Is this the name of a thing, rather than something being worked out?

    A bare `Dog()` is obviously a name. So is `Dog(quality=Lazy())`: putting
    an adjective in front of a noun does not turn it into a computation, and
    treating it as one is how "the lazy dog" came to be reported as a word
    soup needed taught.

    The distinction matters in two places that must agree. Realization uses
    it to decide that a name needs no further resolving, and `Question` uses
    it to decide that nothing was worked out and the thing is being asked
    about rather than evaluated.
    """
    if not isinstance(expr, Call):
        return False
    return all(arg.name == "quality" for arg in expr.args)


def quality_items(expr: Call) -> List[Expr]:
    """The adjectives on a noun phrase, flattened."""
    found: List[Expr] = []
    for arg in expr.args:
        if arg.name != "quality":
            continue
        if isinstance(arg.value, Seq):
            found.extend(arg.value.items)
        else:
            found.append(arg.value)
    return found


def walk(expr: Expr) -> Iterator[Expr]:
    """Depth-first walk, parents before children."""
    yield expr
    for child in expr.children():
        for sub in walk(child):
            yield sub


def concept_names(expr: Expr) -> List[str]:
    """Every concept name mentioned, in depth-first order, deduplicated."""
    seen: List[str] = []
    for node in walk(expr):
        if isinstance(node, Call) and node.concept not in seen:
            seen.append(node.concept)
    return seen


def substitute(expr: Expr, bindings: Dict[str, Expr]) -> Expr:
    """Replace free `Var`s with their bound expressions."""
    if isinstance(expr, Var):
        return bindings.get(expr.name, expr)
    if isinstance(expr, Seq):
        return Seq(tuple(substitute(i, bindings) for i in expr.items))
    if isinstance(expr, Call):
        return Call(
            expr.concept,
            tuple(Arg(a.name, substitute(a.value, bindings)) for a in expr.args),
        )
    return expr


def match(pattern: Expr, target: Expr, bindings: Optional[Dict[str, Expr]] = None):
    """Structurally match `pattern` against `target`.

    `Var`s in the pattern bind to whole subtrees. Returns the bindings dict, or
    None if the shapes disagree. A pattern keyword argument must be present in
    the target; extra target arguments are allowed so learned rules stay usable
    as expressions grow richer.
    """
    b: Dict[str, Expr] = dict(bindings) if bindings else {}

    if isinstance(pattern, Var):
        existing = b.get(pattern.name)
        if existing is not None and existing != target:
            return None
        b[pattern.name] = target
        return b

    if isinstance(pattern, Lit):
        return b if isinstance(target, Lit) and pattern.value == target.value else None

    if isinstance(pattern, Seq):
        if not isinstance(target, Seq) or len(pattern.items) != len(target.items):
            return None
        for p, t in zip(pattern.items, target.items):
            r = match(p, t, b)
            if r is None:
                return None
            b = r
        return b

    if isinstance(pattern, Call):
        if not isinstance(target, Call) or pattern.concept != target.concept:
            return None
        used: set = set()
        for i, a in enumerate(pattern.args):
            if a.name:
                tv = target.get(a.name)
                if tv is None:
                    return None
                r = match(a.value, tv, b)
                if r is None:
                    return None
                b = r
                used.add(a.name)
            else:
                tp = target.positional()
                idx = len([x for x in pattern.args[:i] if x.name is None])
                if idx >= len(tp):
                    return None
                r = match(a.value, tp[idx], b)
                if r is None:
                    return None
                b = r
        return b

    return None
