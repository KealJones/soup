"""The concepts Soup is born knowing, and the Python realizations for the ones
that bottom out in arithmetic, collections, logic or memory.

Plenty of concepts in here are realized by *rewrite rules* rather than code
(`Double(x) := Multiply(x, 2)`), because a realization is allowed to resolve
to more concepts. That is the whole point.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import datetime as datetime_mod
from datetime import datetime
from typing import List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, call, is_name, render, substitute, walk
from .knowledge import (
    HAS_PROPERTY,
    SYMMETRIC,
    TAXONOMIC,
    TRANSITIVE,
    Evidence,
    Knowledge,
)
from .parse import parse_definition
from .realize import Context, Gap, NATIVES, native, only_you_know

UNKNOWN_TOLERANT = {
    "Unknown",
    "Ask",
    "If",
    "Choose",
    "Select",
    "Pick",
    "Answer",
    "Assertion",
    "Explain",
    "Teach",
    "Remember",
    "Query",
    "Not",
    "Say",
    "Request",
    "Hear",
    "Speak",
    "Meaning",
    "Turn",
    "Think",
    "Consult",
    "Conversation",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _num(expr: Optional[Expr]) -> Optional[float]:
    if isinstance(expr, Lit) and isinstance(expr.value, (int, float)) and not isinstance(
        expr.value, bool
    ):
        return float(expr.value)
    return None


def _numbers(c: Call, *names: str) -> Optional[List[float]]:
    """The numbers in this call, if the call is one this native recognises.

    A word can mean more than one thing and the argument names are how the
    sentence said which. Scraping every number out regardless of its name is
    how `Power(work=100, time=4)` became a hundred million: the exponential
    reading claimed a shape that was never about exponents. So a named
    argument has to be a name this native knows, or this is somebody else's
    meaning of the word and the native declines.

    Pass no names to accept any of them, for the operations that genuinely
    do not care (Add of three things, Average of a pile).
    """
    allowed = set(names)
    out: List[float] = []
    for a in c.args:
        if names and a.name and a.name not in allowed:
            return None
        n = _num(a.value)
        if n is None:
            return None
        out.append(n)
    return out or None


def _pack(value: float) -> Lit:
    if isinstance(value, float) and value.is_integer():
        return Lit(int(value))
    return Lit(round(value, 10) if isinstance(value, float) else value)


def _bool(expr: Optional[Expr]) -> Optional[bool]:
    if isinstance(expr, Lit) and isinstance(expr.value, bool):
        return expr.value
    return None


def _items(expr: Optional[Expr]) -> Optional[List[Expr]]:
    if isinstance(expr, Seq):
        return list(expr.items)
    return None


def unknown(about: Expr) -> Call:
    return Call("Unknown", (Arg("about", about),))


def is_unknown(expr: Expr) -> bool:
    return isinstance(expr, Call) and expr.concept == "Unknown"


def _collection(ctx: Context, c: Call, *names: str):
    """Realize the collection argument of an operation that takes one.

    Returns (items, early). `early` is a result to hand straight back, which
    is how "the biggest of the pokemon" reports that it knows no pokemon
    instead of silently doing nothing.
    """
    target = c.first(*names)
    if target is None:
        return None, None
    value = ctx.realize(target)
    if is_unknown(value):
        return None, value
    return _items(value), None


def _apply_to(transform: Expr, item: Expr) -> Expr:
    """Apply a concept-as-operation to an item, keeping partial arguments."""
    if isinstance(transform, Call):
        return Call(transform.concept, transform.args + (Arg(None, item),))
    return item


def _attribute_question(ctx: Context, c: Call) -> Optional[Call]:
    """A list-op that is actually asking for an attribute of a named thing.

    `Minimum(collection=Players(), in=Chess())` is what the ears make of
    "the min number of players for chess". That is not a list. It is
    Players of Chess, and Minimum is a hint about which similarly-named
    Wikidata property to reach for.

    Leave real list operations alone: anything with a `by=`, and anything
    whose "collection" is already an operation soup can run (Files, Range).
    """
    if c.get("by") is not None:
        return None
    attr = c.first("collection", "of", "items")
    if not isinstance(attr, Call):
        return None
    thing = c.first("in", "for", "subject")
    if thing is None and attr.args:
        thing = attr.first("in", "for", "subject", "of")
        attr = Call(attr.concept, ())
    if not isinstance(thing, Call) or attr.args:
        return None
    if attr.concept in NATIVES:
        return None
    cd = ctx.knowledge.concept(attr.concept)
    if cd is not None and cd.kind == "operation":
        return None
    return Call(attr.concept, (Arg("subject", thing),))


def _realize_attribute(ctx: Context, expr: Call, specifically: str) -> Expr:
    source = next(
        (s for s in ctx.realizer.sources if hasattr(s, "specifically")), None
    )
    previous = getattr(source, "specifically", None)
    if source is not None:
        source.specifically = specifically
    try:
        return ctx.realize(expr)
    finally:
        if source is not None:
            source.specifically = previous


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


@native("List", special=True)
def _as_list(ctx: Context, c: Call) -> Optional[Expr]:
    """A list of things, or a scene that already is one.

    `List(subject=Activities(), given=[Read(Ann, Book), Cook(Margaret)])`
    is "give me the activities". The givens are the list. They are not
    verbs to go and perform, and Play is not a hole in the language.
    """
    given = c.get("given")
    if given is not None:
        return _seq_from_scene(given)
    inner = c.first("of", "collection", "items")
    if inner is None:
        return None
    items = _items(inner)
    if items is not None:
        return Seq(tuple(items))
    return inner


def _seq_from_scene(given: Expr) -> Optional[Expr]:
    items = _items(given) or ([given] if isinstance(given, Call) else [])
    kept = []
    for item in items:
        if not isinstance(item, Call) or item.concept == "Unknown":
            continue
        if item.args and all(
            isinstance(a.value, Call) and a.value.concept == "Unknown" for a in item.args
        ):
            continue
        kept.append(item)
    return Seq(tuple(kept)) if kept else None


@native("Add", "Plus", "Sum")
def _add(ctx: Context, c: Call) -> Optional[Expr]:
    ns = _numbers(c)
    if ns is None:
        # "the sum of those" totals a collection, but only when the collection
        # is the whole argument. `Add([1,2,3], 10)` means something else and
        # the distribution step in the realizer handles it.
        if len(c.args) == 1:
            items = _items(c.args[0].value)
            if items is not None:
                vals = [_num(i) for i in items]
                if vals and all(v is not None for v in vals):
                    return _pack(sum(vals))  # type: ignore[arg-type]
        return None
    return _pack(sum(ns))


@native("Multiply", "Times", "Product")
def _multiply(ctx: Context, c: Call) -> Optional[Expr]:
    ns = _numbers(c)
    if ns is None:
        return None
    total = 1.0
    for n in ns:
        total *= n
    return _pack(total)


@native("Subtract", "Minus", "Difference")
def _subtract(ctx: Context, c: Call) -> Optional[Expr]:
    left = _num(c.first("left", "from", "value"))
    right = _num(c.get("right") or c.get("by") or _second(c))
    if left is None or right is None:
        return None
    return _pack(left - right)


@native("Divide", "DividedBy")
def _divide(ctx: Context, c: Call) -> Optional[Expr]:
    left = _num(c.first("left", "value", "numerator", "of", "players"))
    right = _num(c.get("right") or c.get("by") or c.get("per") or c.get("denominator") or _second(c))
    if left is None or right is None:
        ns = _numbers(c)
        if ns and len(ns) >= 2:
            left, right = ns[0], ns[1]
    if left is None or right is None:
        return None
    if right == 0:
        return unknown(c)
    return _pack(left / right)


@native("Power", "ToThePowerOf")
def _power(ctx: Context, c: Call) -> Optional[Expr]:
    ns = _numbers(c, "base", "value", "of", "exponent", "by", "to")
    if not ns or len(ns) != 2:
        return None
    return _pack(ns[0] ** ns[1])


@native("Modulo", "Remainder")
def _modulo(ctx: Context, c: Call) -> Optional[Expr]:
    ns = _numbers(c)
    if not ns or len(ns) != 2 or ns[1] == 0:
        return None
    return _pack(ns[0] % ns[1])


@native("Negate")
def _negate(ctx: Context, c: Call) -> Optional[Expr]:
    n = _num(c.first("value", "of"))
    return None if n is None else _pack(-n)


@native("Round")
def _round(ctx: Context, c: Call) -> Optional[Expr]:
    n = _num(c.first("value", "of"))
    return None if n is None else _pack(round(n))


@native("Average", "Mean")
def _average(ctx: Context, c: Call) -> Optional[Expr]:
    items = _items(c.first("collection", "of", "items"))
    if items is None:
        ns = _numbers(c)
        if not ns:
            return None
        return _pack(sum(ns) / len(ns))
    vals = [_num(i) for i in items]
    if not vals or any(v is None for v in vals):
        return None
    return _pack(sum(vals) / len(vals))  # type: ignore[arg-type]


def _second(c: Call) -> Optional[Expr]:
    p = c.positional()
    return p[1] if len(p) > 1 else None


# ---------------------------------------------------------------------------
# comparison and logic
# ---------------------------------------------------------------------------


def _pair(c: Call):
    left = c.first("left", "value", "subject")
    right = c.get("right") or c.get("than") or c.get("to") or _second(c)
    return left, right


@native("GreaterThan", "MoreThan", "Bigger")
def _greater(ctx: Context, c: Call) -> Optional[Expr]:
    left, right = _pair(c)
    a, b = _num(left), _num(right)
    if a is None or b is None:
        return unknown(c) if _has_unresolved(left, right) else None
    return Lit(a > b)


@native("LessThan", "Smaller")
def _less(ctx: Context, c: Call) -> Optional[Expr]:
    left, right = _pair(c)
    a, b = _num(left), _num(right)
    if a is None or b is None:
        return unknown(c) if _has_unresolved(left, right) else None
    return Lit(a < b)


@native("EqualTo", "SameAs", "Equals")
def _equal(ctx: Context, c: Call) -> Optional[Expr]:
    left, right = _pair(c)
    if left is None or right is None:
        return None
    a, b = _num(left), _num(right)
    if a is not None and b is not None:
        return Lit(a == b)
    return Lit(left == right)


@native("NotEqualTo", "Different")
def _not_equal(ctx: Context, c: Call) -> Optional[Expr]:
    out = _equal(ctx, c)
    if isinstance(out, Lit) and isinstance(out.value, bool):
        return Lit(not out.value)
    return out


@native("Not")
def _not(ctx: Context, c: Call) -> Optional[Expr]:
    v = _bool(c.first("value", "of", "proposition"))
    if v is None:
        inner = c.first("value", "of", "proposition")
        return unknown(c) if inner is not None and is_unknown(inner) else None
    return Lit(not v)


@native("And", "Both")
def _and(ctx: Context, c: Call) -> Optional[Expr]:
    vals = [_bool(a.value) for a in c.args]
    if any(v is False for v in vals):
        return Lit(False)
    if all(v is True for v in vals) and vals:
        return Lit(True)
    return None


@native("Or", "Either")
def _or(ctx: Context, c: Call) -> Optional[Expr]:
    vals = [_bool(a.value) for a in c.args]
    if any(v is True for v in vals):
        return Lit(True)
    if all(v is False for v in vals) and vals:
        return Lit(False)
    return None


def _has_unresolved(*exprs: Optional[Expr]) -> bool:
    return any(e is not None and (is_unknown(e) or isinstance(e, Call)) for e in exprs)


# ---------------------------------------------------------------------------
# collections
# ---------------------------------------------------------------------------


# Map, Filter, Sort and the extremes take an *operation* as an argument, not a
# value. Realizing `Multiply(10)` before handing it over would collapse the
# partial application into the number 10, so these keep their arguments whole
# and realize only the collection.


@native("Map", "Each", "EveryOneOf", special=True)
def _map(ctx: Context, c: Call) -> Optional[Expr]:
    items, early = _collection(ctx, c, "collection", "over", "items")
    transform = c.get("transformation") or c.get("with") or c.get("by") or _second(c)
    if early is not None:
        return early
    if items is None or transform is None:
        return None
    return Seq(tuple(ctx.realize(_apply_to(transform, i)) for i in items))


@native("Filter", "Only", "Keep", special=True)
def _filter(ctx: Context, c: Call) -> Optional[Expr]:
    items, early = _collection(ctx, c, "collection", "over", "items")
    test = c.get("where") or c.get("test") or c.get("by") or _second(c)
    if early is not None:
        return early
    if items is None or test is None:
        return None
    kept = []
    for i in items:
        verdict = ctx.realize(_apply_to(test, i))
        if _bool(verdict) is True:
            kept.append(i)
    return Seq(tuple(kept))


@native("First", "Head")
def _first(ctx: Context, c: Call) -> Optional[Expr]:
    items = _items(c.first("collection", "of", "items"))
    if items is None:
        return None
    return items[0] if items else unknown(c)


@native("Last")
def _last(ctx: Context, c: Call) -> Optional[Expr]:
    items = _items(c.first("collection", "of", "items"))
    if items is None:
        return None
    return items[-1] if items else unknown(c)


@native("Nth", "ItemAt")
def _nth(ctx: Context, c: Call) -> Optional[Expr]:
    """The nth of them, counting the way people count: the first one is 1.

    Here so that an ordinal has somewhere to land. Without it "the fifth
    sister" has no composable reading, and asked what `Fifth(of)` means
    with nothing else to reach for, a model quite reasonably says a fifth
    of it, and then five sisters divided by five is one sister.
    """
    items = _items(c.first("collection", "of", "items"))
    index = _num(c.first("index", "position", "at", "n"))
    if items is None or index is None:
        return None
    i = int(index)
    if i < 0:
        i += len(items) + 1
    return items[i - 1] if 1 <= i <= len(items) else unknown(c)


@native("Count", "HowMany")
def _count(ctx: Context, c: Call) -> Optional[Expr]:
    peeled = _attribute_question(ctx, c)
    if peeled is not None:
        inner = ctx.realize(peeled)
    else:
        inner = c.first("collection", "of", "items")
    items = _items(inner)
    if items is None:
        # "how many players does chess have" arrives as Count(Players(...)),
        # and Players already came back 2. The ears wrap the question in
        # "how many" because the sentence says how many; when what is inside
        # is a quantity rather than a pile of things, it is already the
        # answer and counting it again gets you "how many of 2".
        if isinstance(inner, Lit) and isinstance(inner.value, (int, float)):
            return inner
        return None
    return Lit(len(items))


@native("Maximum", "Biggest", "Largest", "Max", special=True)
def _maximum(ctx: Context, c: Call) -> Optional[Expr]:
    return _extreme(ctx, c, biggest=True)


@native("Minimum", "Smallest", "Min", special=True)
def _minimum(ctx: Context, c: Call) -> Optional[Expr]:
    return _extreme(ctx, c, biggest=False)


def _extreme(ctx: Context, c: Call, biggest: bool) -> Optional[Expr]:
    peeled = _attribute_question(ctx, c)
    if peeled is not None:
        # "the min number of players for chess" is a property of chess, not
        # the smallest of a pile of players. Minimum is a special form, so
        # declining used to return the tree unchanged and mouth would read
        # it as "the smallest of Players".
        return _realize_attribute(
            ctx, peeled, "maximum" if biggest else "minimum"
        )
    items, early = _collection(ctx, c, "collection", "of", "items")
    if early is not None:
        return early
    if items is None:
        return None
    if not items:
        return unknown(c)
    by = c.get("by")
    scored = []
    for i in items:
        key = ctx.realize(_apply_to(by, i)) if by is not None else i
        n = _num(key)
        if n is None:
            return unknown(c) if by is not None else None
        scored.append((n, i))
    scored.sort(key=lambda p: p[0], reverse=biggest)
    return scored[0][1]


@native("Sort", "Ordered", special=True)
def _sort(ctx: Context, c: Call) -> Optional[Expr]:
    items, early = _collection(ctx, c, "collection", "of", "items")
    if early is not None:
        return early
    if items is None:
        return None
    by = c.get("by")
    scored = []
    for i in items:
        key = ctx.realize(_apply_to(by, i)) if by is not None else i
        n = _num(key)
        if n is None:
            return None
        scored.append((n, i))
    descending = _bool(c.get("descending")) is True
    scored.sort(key=lambda p: p[0], reverse=descending)
    return Seq(tuple(i for _, i in scored))


@native("Reverse")
def _reverse(ctx: Context, c: Call) -> Optional[Expr]:
    items = _items(c.first("collection", "of", "items"))
    if items is None:
        return None
    return Seq(tuple(reversed(items)))


@native("Range")
def _range(ctx: Context, c: Call) -> Optional[Expr]:
    start = _num(c.first("start", "from"))
    end = _num(c.get("end") or c.get("to") or _second(c))
    if start is None or end is None:
        return None
    step = 1 if end >= start else -1
    return Seq(tuple(Lit(int(v)) for v in _frange(int(start), int(end), step)))


def _frange(start: int, end: int, step: int):
    v = start
    while (step > 0 and v <= end) or (step < 0 and v >= end):
        yield v
        v += step


@native("Size", "Length")
def _size(ctx: Context, c: Call) -> Optional[Expr]:
    """The size of a number is the number. "is 5 bigger than 3" should just work."""
    target = c.first("subject", "of", "value")
    n = _num(target)
    if n is not None:
        return Lit(n if not float(n).is_integer() else int(n))
    items = _items(target)
    if items is not None:
        return Lit(len(items))
    return None


@native("Even")
def _even(ctx: Context, c: Call) -> Optional[Expr]:
    n = _num(c.first("value", "of"))
    return None if n is None else Lit(int(n) % 2 == 0)


@native("Odd")
def _odd(ctx: Context, c: Call) -> Optional[Expr]:
    n = _num(c.first("value", "of"))
    return None if n is None else Lit(int(n) % 2 == 1)


# ---------------------------------------------------------------------------
# binding and control (special forms: arguments stay unevaluated)
# ---------------------------------------------------------------------------


@native("Let", special=True)
def _let(ctx: Context, c: Call) -> Optional[Expr]:
    body = c.get("then") or c.get("in") or c.get("body")
    env = dict(ctx.env)
    for a in c.args:
        if a.name in (None, "then", "in", "body"):
            continue
        env[a.name] = ctx.realizer._realize(a.value, env, ctx.depth + 1, ctx.result)
    if body is None:
        return Lit(None)
    return ctx.realizer._realize(body, env, ctx.depth + 1, ctx.result)


@native("If", "Conditional", special=True)
def _if(ctx: Context, c: Call) -> Optional[Expr]:
    condition = c.first("condition", "if")
    then = c.get("then") or c.get("consequence")
    otherwise = c.get("otherwise") or c.get("else")
    verdict = ctx.realize(condition) if condition is not None else None
    truthy = _bool(verdict)
    if truthy is True:
        return ctx.realize(then) if then is not None else Lit(True)
    if truthy is False:
        return ctx.realize(otherwise) if otherwise is not None else Lit(None)
    return Call(
        "Conditional",
        (
            Arg("condition", verdict if verdict is not None else Lit(None)),
            Arg("then", then if then is not None else Lit(None)),
        ),
    )


@native("Quote", special=True)
def _quote(ctx: Context, c: Call) -> Optional[Expr]:
    inner = c.first("value", "of")
    return inner if inner is not None else Lit(None)


@native("Unknown", special=True)
def _unknown_form(ctx: Context, c: Call) -> Optional[Expr]:
    return c


# ---------------------------------------------------------------------------
# asking things of memory
# ---------------------------------------------------------------------------

_HOLES = ("Who", "What", "Which", "Where", "When", "Something", "Someone")


def _dehole(expr: Expr, found: List[str]) -> Expr:
    """Turn `Who()` / `What()` holes into fresh pattern variables."""
    if isinstance(expr, Call):
        if expr.concept in _HOLES and not expr.args:
            name = "_q%d" % len(found)
            found.append(name)
            return Var(name)
        return Call(expr.concept, tuple(Arg(a.name, _dehole(a.value, found)) for a in expr.args))
    if isinstance(expr, Seq):
        return Seq(tuple(_dehole(i, found) for i in expr.items))
    return expr


@native("Find", "Get", "Show", "Give")
def _find(ctx: Context, c: Call) -> Optional[Expr]:
    """"find the biggest of those" is just asking for the thing itself."""
    target = c.first("target", "of", "thing")
    return target


@native("Query", special=True)
def _query(ctx: Context, c: Call) -> Optional[Expr]:
    pattern = c.first("pattern", "of", "for")
    if pattern is None:
        return None
    holes: List[str] = []
    probe = _dehole(pattern, holes)
    hits = ctx.knowledge.query(probe)
    answers: List[Expr] = []
    for fact, bindings in hits:
        if not fact.truth:
            continue
        if holes:
            for h in holes:
                if h in bindings and bindings[h] not in answers:
                    answers.append(bindings[h])
        elif Lit(True) not in answers:
            answers.append(Lit(True))
    if not answers:
        return unknown(pattern)
    if len(answers) == 1:
        return answers[0]
    return Seq(tuple(answers))


@native("Ask", special=True)
def _ask(ctx: Context, c: Call) -> Optional[Expr]:
    asked = c.get("of")
    if isinstance(asked, Call) and is_name(asked) and c.get("about") is not None:
        # Asking somebody something, rather than finding out whether
        # something is so. A finished speech act: there is nothing to
        # realize, only words to choose for it.
        return c
    proposition = c.first("proposition", "of", "about")
    if proposition is None:
        return None
    verdict = ctx.realize(proposition)
    if _bool(verdict) is not None or is_unknown(verdict):
        return Call("Assertion", (Arg("proposition", proposition), Arg("truth", verdict)))
    if isinstance(verdict, Lit) or isinstance(verdict, Seq):
        return Call("Answer", (Arg("value", verdict), Arg("to", proposition)))
    return Call("Assertion", (Arg("proposition", proposition), Arg("truth", unknown(proposition))))


@native("Remember", "Believe", special=True)
def _remember(ctx: Context, c: Call) -> Optional[Expr]:
    proposition = c.first("proposition", "that", "of")
    if proposition is None:
        return None
    quoted = _quoted_in_thought(ctx, c)
    if quoted is not None:
        return quoted
    truth = _bool(c.get("truth"))
    settled = True if truth is None else truth
    ctx.knowledge.assert_fact(proposition, settled, Evidence(source="user"))
    ctx.effect("remembered %s" % render(proposition, False))
    # Carry the truth through, or "i can't swim" comes back as "okay, you can swim".
    return Call("Acknowledged", (Arg("about", proposition), Arg("truth", Lit(settled))))


@native("Recall", special=True)
def _recall(ctx: Context, c: Call) -> Optional[Expr]:
    subject = c.first("about", "of", "subject")
    if not isinstance(subject, Call):
        return None
    facts = [f.proposition for f in ctx.knowledge.facts_mentioning(subject.concept) if f.truth]
    if not facts:
        return unknown(subject)
    return Call(
        "Explanation",
        (Arg("concept", subject), Arg("facts", Seq(tuple(facts)))),
    )


# ---------------------------------------------------------------------------
# the clock
#
# World-facing, same family as Read/Write/Fetch below. "i don't know what
# the time is" is a silly thing for a program to say.
# ---------------------------------------------------------------------------


@native("Time", "Now")
def _time(ctx: Context, c: Call) -> Optional[Expr]:
    if c.args:
        return None
    ctx.effect("read the clock")
    return Lit(datetime.now().strftime("%I:%M %p").lstrip("0").lower())


@native("TwentyFourHourTime")
def _time_24(ctx: Context, c: Call) -> Optional[Expr]:
    """The other clock face, and the only other one there is.

    Deliberately a single concept rather than a list of the words people use
    for it. "military time", "army time", "iso time" and "24 hour time" are
    four names for this, and enumerating them here would mean the fifth name
    breaks. Mapping a phrase onto this concept is the ears' job, and being
    told that some new phrase means this is something to learn once and
    keep.
    """
    ctx.effect("read the clock")
    return Lit(datetime.now().strftime("%H:%M"))


@native("Date", "Today")
def _date(ctx: Context, c: Call) -> Optional[Expr]:
    if c.args:
        return None
    return Lit(datetime.now().strftime("%B %d, %Y").replace(" 0", " "))


@native("Day", "Weekday")
def _day(ctx: Context, c: Call) -> Optional[Expr]:
    if c.args:
        return None
    return Lit(datetime.now().strftime("%A"))


@native("Year")
def _year(ctx: Context, c: Call) -> Optional[Expr]:
    if c.args:
        return None
    return Lit(datetime.now().year)


@native("Timezone")
def _timezone(ctx: Context, c: Call) -> Optional[Expr]:
    """The clock's zone, not where someone lives."""
    subject = c.first("subject", "of", "in")
    if subject is not None and not (
        isinstance(subject, Call) and subject.concept in ("User", "Assistant", "Self")
    ):
        return None
    now = datetime.now().astimezone()
    name = getattr(now.tzinfo, "key", None) or now.tzname() or now.strftime("%Z")
    if not name:
        return None
    ctx.effect("read the clock")
    return Lit(name)


@native("Location", special=True)
def _location(ctx: Context, c: Call) -> Optional[Expr]:
    """Do not realize `attribute=Timezone()` first; that would turn the
    label into a clock reading and the LivesIn rule would steal the rest.
    """
    rewritten = ctx.realizer._apply_rules(c, ctx.result)
    if rewritten is None:
        return None
    return ctx.realize(rewritten)


# ---------------------------------------------------------------------------
# the world
#
# These bottom out the way Time does: they touch something that is not the
# store. Everything built on them (Create, Change, a Wikidata lookup) is a
# rule, not a new Python adapter.
# ---------------------------------------------------------------------------

_AGENT = "soup/0.1 (https://github.com/kealjones/soup) python-urllib"


def _as_text(expr: Optional[Expr]) -> Optional[str]:
    if isinstance(expr, Lit) and expr.value is not None:
        return str(expr.value)
    return None


def _quoted_in_thought(ctx: Context, expr: Expr) -> Optional[Expr]:
    """World-changing natives do not fire while thinking. They become meaning."""
    if not ctx.thinking:
        return None
    return Call("Meaning", (Arg("of", expr), Arg("as", Call("Thought", ()))))


def _as_path(expr: Optional[Expr]) -> Optional[str]:
    text = _as_text(expr)
    if text is not None:
        return text
    if isinstance(expr, Call):
        inner = expr.get("path") or expr.get("file") or expr.get("name")
        if inner is not None:
            return _as_path(inner)
    return None


def _from_json(value) -> Expr:
    if isinstance(value, dict):
        return Call("Object", tuple(Arg(str(k), _from_json(v)) for k, v in value.items()))
    if isinstance(value, list):
        return Seq(tuple(_from_json(item) for item in value))
    if isinstance(value, bool) or value is None:
        return Lit(value)
    if isinstance(value, float) and value.is_integer():
        return Lit(int(value))
    return Lit(value)


def _write_file(ctx: Context, path: str, text: str) -> Expr:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    ctx.effect("wrote %s" % path)
    return Call("Acknowledged", (Arg("about", Call("Write", (Arg("path", Lit(path)),))),))


def _query_string(expr: Optional[Expr]) -> str:
    if expr is None:
        return ""
    text = _as_text(expr)
    if text is not None:
        return text
    if isinstance(expr, Call):
        pairs = []
        for arg in expr.args:
            if not arg.name:
                continue
            value = _as_text(arg.value)
            if value is None:
                continue
            pairs.append((arg.name, value))
        return urllib.parse.urlencode(pairs)
    return ""


@native("File", "Directory", "Folder", "Object")
def _stands(ctx: Context, c: Call) -> Optional[Expr]:
    """Things that are themselves. A file is not a failed computation."""
    return c


@native("PathOf")
def _path_of(ctx: Context, c: Call) -> Optional[Expr]:
    path = _as_path(c.first("of", "value", "path", "file"))
    return Lit(path) if path is not None else None


_PY_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def _code_text(c: Call) -> Optional[str]:
    source = _as_text(c.first("code", "source", "command", "of"))
    if source is not None:
        return source
    for a in c.args:
        if a.name is None:
            source = _as_text(a.value)
            if source is not None:
                return source
    return None


def _from_python(value) -> Expr:
    if isinstance(value, (dict, list, bool)) or value is None:
        return _from_json(value)
    if isinstance(value, float) and value.is_integer():
        return Lit(int(value))
    if isinstance(value, (int, float, str)):
        return Lit(value)
    return Lit(str(value).strip())


@native("Python")
def _python(ctx: Context, c: Call) -> Optional[Expr]:
    """A method Soup was taught, until a native exists. The snippet is the rule."""
    source = _code_text(c)
    if not source:
        return None
    ns = {"datetime": datetime_mod, "os": os, "json": json, "time": time}
    for a in c.args:
        if a.name and a.name not in ("code", "source", "command", "of"):
            ns[a.name] = a.value.value if isinstance(a.value, Lit) else render(a.value, False)
    try:
        value = eval(source, {"__builtins__": _PY_BUILTINS}, ns)
    except Exception as exc:
        return Lit(str(exc))
    ctx.effect("ran python")
    return _from_python(value)


@native("Shell", "Bash")
def _shell(ctx: Context, c: Call) -> Optional[Expr]:
    cmd = _code_text(c)
    if not cmd:
        return None
    try:
        ran = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=8
        )
    except Exception as exc:
        return Lit(str(exc))
    ctx.effect("ran shell")
    return Lit((ran.stdout or ran.stderr or "").strip())


@native("Join")
def _join(ctx: Context, c: Call) -> Optional[Expr]:
    parts = [_as_path(a.value) for a in c.args]
    if any(p is None for p in parts) or not parts:
        return None
    return Lit(os.path.join(*parts))


@native("Concat")
def _concat(ctx: Context, c: Call) -> Optional[Expr]:
    parts = [_as_text(a.value) for a in c.args]
    if any(p is None for p in parts):
        return None
    return Lit("".join(parts))


@native("Replace")
def _replace(ctx: Context, c: Call) -> Optional[Expr]:
    positional = c.positional()
    text = _as_text(c.first("text", "value", "in"))
    old = _as_text(c.get("from") or c.get("old") or (positional[1] if len(positional) > 1 else None))
    new = _as_text(c.get("to") or c.get("new") or (positional[2] if len(positional) > 2 else None))
    if text is None or old is None or new is None:
        return None
    return Lit(text.replace(old, new))


@native("Lowercase")
def _lowercase(ctx: Context, c: Call) -> Optional[Expr]:
    text = _as_text(c.first("text", "value", "of"))
    return Lit(text.lower()) if text is not None else None


@native("GetProperty", "At", "Field")
def _get_property(ctx: Context, c: Call) -> Optional[Expr]:
    obj = c.first("of", "from", "object", "in")
    key_expr = c.get("key") or c.get("name") or _second(c)
    if obj is None or key_expr is None:
        return None
    key = _as_text(key_expr)
    if key is None and _num(key_expr) is not None:
        key = int(_num(key_expr))
    if isinstance(obj, Call) and isinstance(key, str):
        found = obj.get(key)
        return found if found is not None else unknown(c)
    if isinstance(obj, Seq):
        items = list(obj.items)
        if isinstance(key, int) and 0 <= key < len(items):
            return items[key]
        return unknown(c)
    return None


@native("HasKey", special=True)
def _has_key(ctx: Context, c: Call) -> Optional[Expr]:
    """Whether a JSON-like object contains a named field.

    Unlike `GetProperty`, absence is a useful false result here. This lets
    rules distinguish a Wikidata claim with no end-date qualifier from one
    whose `P582` qualifier is present without knowing anything about JSON's
    Python representation.
    """
    obj = c.first("object", "of", "in", "value")
    key_expr = c.get("key") or c.get("name") or _second(c)
    key = _as_text(key_expr)
    if obj is None or key is None:
        return None
    value = ctx.realize(obj)
    if is_unknown(value):
        return Lit(False)
    if isinstance(value, Call):
        return Lit(any(arg.name == key for arg in value.args))
    return Lit(False)


@native("Json")
def _json_parse(ctx: Context, c: Call) -> Optional[Expr]:
    text = _as_text(c.first("text", "of", "value"))
    if text is None:
        return None
    try:
        return _from_json(json.loads(text))
    except ValueError:
        return unknown(c)


@native("Read")
def _read(ctx: Context, c: Call) -> Optional[Expr]:
    path = _as_path(c.first("path", "file", "of"))
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            ctx.effect("read %s" % path)
            return Lit(fh.read())
    except OSError:
        return unknown(c)


@native("Write")
def _write(ctx: Context, c: Call) -> Optional[Expr]:
    positional = c.positional()
    path = _as_path(
        c.get("path") or c.get("file") or c.get("to") or (positional[0] if positional else None)
    )
    text = _as_text(
        c.get("contents")
        or c.get("content")
        or c.get("text")
        or (positional[1] if len(positional) > 1 else None)
    )
    if path is None or text is None:
        return None
    quoted = _quoted_in_thought(ctx, c)
    if quoted is not None:
        return quoted
    return _write_file(ctx, path, text)


@native("Files")
def _files(ctx: Context, c: Call) -> Optional[Expr]:
    path = _as_path(c.first("path", "folder", "of", "in"))
    if path is None:
        return None
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return unknown(c)
    ctx.effect("listed %s" % path)
    return Seq(tuple(Lit(name) for name in names))


@native("Delete")
def _delete(ctx: Context, c: Call) -> Optional[Expr]:
    path = _as_path(c.first("path", "file", "target", "of"))
    if path is None:
        return None
    quoted = _quoted_in_thought(ctx, c)
    if quoted is not None:
        return quoted
    try:
        os.remove(path)
    except IsADirectoryError:
        os.rmdir(path)
    except OSError:
        return unknown(c)
    ctx.effect("deleted %s" % path)
    return Call("Acknowledged", (Arg("about", Call("Delete", (Arg("path", Lit(path)),))),))


@native("Url")
def _url(ctx: Context, c: Call) -> Optional[Expr]:
    """A URL is structure, not a string you glue together.

    Fetch(Url(scheme, host, path, query=Object(...))) is the whole point.
    A lone string still works because people will say it that way, but
    soup should not be in the business of concatenating query strings.
    """
    if len(c.args) == 1 and not c.arg_names():
        text = _as_text(c.args[0].value)
        return Lit(text) if text is not None else None
    scheme = _as_text(c.first("scheme", "protocol")) or "https"
    host = _as_text(c.first("host", "domain"))
    if not host:
        return None
    path = _as_text(c.get("path")) or "/"
    if not path.startswith("/"):
        path = "/" + path
    query = _query_string(c.get("query"))
    return Lit(urllib.parse.urlunparse((scheme, host, path, "", query, "")))


@native("Fetch")
def _fetch(ctx: Context, c: Call) -> Optional[Expr]:
    url = _as_text(c.first("url", "from", "of"))
    if url is None:
        return None
    quoted = _quoted_in_thought(ctx, c)
    if quoted is not None:
        return quoted
    request = urllib.request.Request(
        url, headers={"User-Agent": _AGENT, "Accept": "*/*"}
    )
    try:
        with urllib.request.urlopen(request, timeout=8.0) as response:
            raw = response.read()
        ctx.effect("fetched %s" % url)
        try:
            return Lit(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return Lit(raw.decode("utf-8", "replace"))
    except TimeoutError:
        ctx.note("fetch timed out: %s" % url)
        return unknown(c)
    except (urllib.error.URLError, OSError) as problem:
        ctx.note("fetch failed: %s" % problem)
        return unknown(c)


@native("WebSearch", "SearchWeb", "DuckDuckGoSearch")
def _web_search(ctx: Context, c: Call) -> Optional[Expr]:
    query = _as_text(c.first("query", "text", "for", "about"))
    if query is None:
        return None
    count = _num(c.get("max_results"))
    try:
        from .web import web_search

        markdown = web_search(query, int(count) if count is not None else 5)
    except Exception as exc:
        ctx.note("web search failed: %s" % exc)
        return Lit("Web search failed: %s" % exc)
    ctx.effect("searched the web for %r" % query)
    return Lit(markdown)


@native("VisitWebpage", "BrowseWebpage", "ReadWebpage")
def _visit_webpage(ctx: Context, c: Call) -> Optional[Expr]:
    url = _as_text(c.first("url", "address", "page", "of"))
    if url is None:
        return None
    try:
        from .web import visit_webpage

        markdown = visit_webpage(url)
    except Exception as exc:
        ctx.note("web page visit failed: %s" % exc)
        return Lit("Could not read webpage: %s" % exc)
    ctx.effect("read webpage %s" % url)
    return Lit(markdown)


@native("WikipediaSearch", "SearchWikipedia")
def _wikipedia_search(ctx: Context, c: Call) -> Optional[Expr]:
    query = _as_text(c.first("query", "text", "topic", "for", "about"))
    if query is None:
        return None
    count = _num(c.get("max_results"))
    try:
        from .web import wikipedia_search

        markdown = wikipedia_search(query, int(count) if count is not None else 5)
    except Exception as exc:
        ctx.note("Wikipedia search failed: %s" % exc)
        return Lit("Wikipedia search failed: %s" % exc)
    ctx.effect("searched Wikipedia for %r" % query)
    return Lit(markdown)


@native("TranscribeAudio", "SpeechToText", "Transcribe")
def _transcribe_audio(ctx: Context, c: Call) -> Optional[Expr]:
    audio = c.first("audio", "file", "path", "url", "of")
    source = _as_path(audio) or _as_text(audio)
    if source is None:
        return None
    model = _as_text(c.get("model"))
    try:
        from .web import transcribe_audio

        text = transcribe_audio(source, model=model)
    except Exception as exc:
        ctx.note("audio transcription failed: %s" % exc)
        return Lit("Audio transcription failed: %s" % exc)
    ctx.effect("transcribed audio")
    return Lit(text)


@native("Hear")
def _hear(ctx: Context, c: Call) -> Optional[Expr]:
    """Turn text into meaning. The ears, callable from inside a tree.

    Hear("what is 2 times 3") realizes as if it was said to Soup.
    Hear(Read(path), as=Program()) understands without doing.
    """
    text = _as_text(c.first("text", "of", "value", "utterance", "source"))
    if text is None:
        return None
    mood = c.get("as")
    quoted = isinstance(mood, Call) and mood.concept in (
        "Program",
        "Document",
        "Code",
        "Thought",
        "Utterance",
    )
    meaning = _hear_text(ctx, text)
    if meaning is None:
        return unknown(c)
    ctx.effect("heard")
    if quoted:
        return Call("Meaning", (Arg("of", meaning), Arg("as", mood if mood is not None else Lit("program"))))
    return meaning


def _hear_text(ctx: Context, text: str) -> Optional[Expr]:
    from .discourse import Discourse
    from .ears import Ears

    discourse = getattr(ctx.realizer, "discourse", None) or Discourse()
    heard = Ears(ctx.knowledge, discourse, seat=ctx.realizer.seat).listen(text)
    ctx.result.heard = heard
    return heard.expr


@native("Speak")
def _speak(ctx: Context, c: Call) -> Optional[Expr]:
    """Say a finished answer out loud. The whole of talking, in one concept.

    Soup's answer is the concept expression; this only chooses words for it.
    That is a translation job, and the seat is better at it than a renderer
    in every direction that matters: it does not need a new branch every time
    a concept turns up in a shape nobody wrote one for, and a register or a
    different language costs a line of style rather than a rewrite.

    The model here is the embedded one, so this works with nothing running and
    nothing installed. It is handed the answer and told to add nothing to it:
    the thinking already happened, and a mouth that decides facts is a second
    brain disagreeing with the first.
    """
    target = c.first("text", "of", "value", "about")
    if target is None:
        return None
    # A style is a concept like everything else: Speak(of=X, style=Pirate())
    # or style=Swahili(). Nothing here has to know what any of them mean.
    style = c.get("style") or c.get("as")
    said = _say(ctx, target, render(style, False) if style is not None else "")
    ctx.effect("spoke")
    return Lit(said)


def _say(ctx: Context, target: Expr, style: str = "") -> str:
    if isinstance(target, Lit) and isinstance(target.value, str):
        # Already words. Saying them again only gives the chair a chance to
        # paraphrase away something that was already settled.
        return target.value
    expression = render(target, multiline=False)
    seat = ctx.realizer.seat
    speak = getattr(seat, "speak", None)
    if speak is None or not getattr(seat, "available", True):
        # Nothing in the chair to translate with, so the answer goes out as
        # it is. Honest rather than helpful, which is the right way round for
        # a case that only happens when Soup has been unplugged on purpose.
        return expression
    said = speak(expression, style)
    return _keep_the_letter(said, target) if said else expression


def _keep_the_letter(said: str, expr: Expr) -> str:
    """A multiple choice pick is a letter. Wording must not lose it.

    The one thing the words are not allowed to paraphrase away: everything
    else about a sentence is style, and this is the answer.
    """
    if not isinstance(expr, Call):
        return said
    letter = expr.get("letter")
    if not isinstance(letter, Lit) or not letter.value:
        return said
    tag = str(letter.value).strip().upper()[:1]
    if not tag:
        return said
    # Whatever tag the wording arrived with goes, and the real one goes on:
    # "a. 0" is the right answer written in a way that reads as no answer
    # at all, because the letter is the answer and its case is part of it.
    rest = re.sub(r"^\(?[A-Za-z]\)?\s*[.):]\s*", "", said.strip())
    return "%s. %s" % (tag, rest or said.strip())


_MOODS = ("Thought", "Program", "Document", "Code")
_AGENTS = ("Agent", "Subagent", "Other")
_PURE_SPEECH = ("Affirm", "Deny", "Unintelligible")


def _named(expr: Optional[Expr], names) -> bool:
    return isinstance(expr, Call) and expr.concept in names


def _as_meaning(value: Expr, mood: Optional[Expr]) -> Expr:
    if isinstance(value, Call) and value.concept == "Meaning":
        return value
    as_mood = mood if _named(mood, _MOODS) else Call("Thought", ())
    return Call("Meaning", (Arg("of", value), Arg("as", as_mood)))


def _unwrap_meaning(expr: Expr) -> Expr:
    if isinstance(expr, Call) and expr.concept == "Meaning":
        inner = expr.first("of", "value")
        return inner if inner is not None else expr
    return expr


@native("Conversation", "Think", "Consult", "Subagent", special=True)
def _conversation(ctx: Context, c: Call) -> Optional[Expr]:
    """A nested loop. Thinking is Conversation you are not addressed by.

    Turn speaks. This does not. World natives become Meaning instead of
    happening. Consult hears through the seat first; Think realizes what
    it is already holding. Subagent is the same loop with its own
    Discourse and, if named, its own seat.
    """
    about = c.first("about", "of", "topic", "text")
    if about is None:
        return None
    who = c.get("with")
    nested = c.concept == "Subagent" or _named(who, _AGENTS)
    budget = getattr(ctx.realizer, "think_budget", 3)
    if getattr(ctx.realizer, "_think_depth", 0) >= budget:
        return unknown(c)
    if nested:
        return _subagent(ctx, about, who, c.get("as"))
    if isinstance(about, Lit) and isinstance(about.value, str):
        heard = _hear_text(ctx, about.value)
        if heard is None:
            return unknown(c)
        about = heard
    ctx.realizer._think_depth = getattr(ctx.realizer, "_think_depth", 0) + 1
    previous = ctx.realizer._thinking
    ctx.realizer._thinking = True
    try:
        value = ctx.realize(about)
    finally:
        ctx.realizer._thinking = previous
        ctx.realizer._think_depth -= 1
    return _as_meaning(value, c.get("as"))


def _subagent(ctx: Context, about: Expr, who: Optional[Expr], mood: Optional[Expr]) -> Expr:
    from .discourse import Discourse
    from .realize import Realizer

    seat = ctx.realizer.seat
    name = _as_text(who.first("name", "of", "id")) if isinstance(who, Call) else None
    agents = getattr(ctx.realizer, "agents", None) or {}
    if name and name in agents:
        seat = agents[name]
    remaining = max(0, getattr(ctx.realizer, "think_budget", 3) - getattr(ctx.realizer, "_think_depth", 0))
    inner = Realizer(
        ctx.knowledge,
        seat=seat,
        sources=ctx.realizer.sources,
        discourse=Discourse(),
        agents=agents,
        think_budget=remaining,
    )
    if isinstance(about, Lit) and isinstance(about.value, str):
        from .ears import Ears

        heard = Ears(inner.knowledge, inner.discourse, seat=inner.seat).listen(about.value)
        about = heard.expr
    result = inner.realize(about, thinking=True)
    return _as_meaning(result.value, mood)


# Names that are not mysteries. Asking a model what "I" is helps nobody.
_SELF_EVIDENT = frozenset(
    {"User", "Self", "Assistant", "Agent", "Subagent", "Other", "Ref", "It"}
)


@native("Kind", "Classify", special=True)
def _kind(ctx: Context, c: Call) -> Optional[Expr]:
    """What sort of thing is this name?

    The question `Identity` cannot usefully answer. Asked who Chess is, a
    model writes a sentence, and a sentence is not something to reason
    with. Asked what kind of thing it is, it says `Game(players=2)`, and
    two is a number Soup can count against the people in the room.
    """
    target = c.first("of", "about", "subject")
    if not isinstance(target, Call):
        return None
    seat = ctx.realizer.seat
    if seat is None or not getattr(seat, "available", True):
        return None
    from .teacher import Teacher

    name = Call(target.concept, ())
    meaning = seat.define(render(name, False), list(ctx.knowledge.concepts))
    if meaning is None:
        return None
    taught = Teacher(ctx.knowledge).classify(name, meaning)
    if taught is None:
        return None
    ctx.effect("worked out what %s is" % target.concept)
    return taught


@native("Solve", "FigureOut", "WorkOut", "Reason", special=True)
def _solve(ctx: Context, c: Call) -> Optional[Expr]:
    """Think, and when it will not resolve, ask yourself why, then think again.

    One pass of realization either lands or reports a single hole. This
    keeps going. Each round takes the hole it just hit and puts it to
    itself as a question: an unknown verb is a definition to go and get, an
    unknown name is a kind to find out. Whatever survives the Teacher is
    filed as an ordinary rule or fact, so the next round has more to work
    with than the last one did, and everything learned outlives the
    question that prompted it.

    That is the whole trick, and why it is a conversation rather than a
    bigger prompt: the model is asked small, checkable things about
    meaning, one at a time, and Soup does the reasoning between the
    answers. Nothing here speaks, and nothing here asks the person. A hole
    only they can fill comes back Unknown and the public turn asks them.
    """
    about = c.first("about", "of", "topic", "text")
    if about is None:
        return None
    budget = getattr(ctx.realizer, "think_budget", 3)
    if getattr(ctx.realizer, "_think_depth", 0) >= budget:
        return unknown(c)
    # A scene is not a belief. "there are five sisters in a room" is true
    # for the length of the riddle and then it is not true of anything.
    assumed = _assume(ctx, c.get("given"))
    ctx.realizer._think_depth = getattr(ctx.realizer, "_think_depth", 0) + 1
    try:
        listing = _scene_listing(about, assumed)
        if listing is not None:
            # "what is each of them doing" with the scene already attached
            # is the scene. Thinking would spend its budget defining
            # Activities and then report it does not know.
            return listing
        return _reason(ctx, about, max(1, budget))
    finally:
        ctx.realizer._think_depth -= 1
        _forget_assumed(ctx, assumed)


def _reason(ctx: Context, about: Expr, rounds: int) -> Expr:
    from .discourse import Discourse
    from .realize import Realizer

    inner = Realizer(
        ctx.knowledge,
        seat=ctx.realizer.seat,
        sources=ctx.realizer.sources,
        discourse=Discourse(),
        agents=getattr(ctx.realizer, "agents", None),
        # The rounds are the loop. A thought inside one does not get to
        # start another.
        think_budget=0,
    )
    asked: set = set()
    attempt = inner.realize(about, thinking=True)
    for _ in range(rounds):
        if attempt.resolved and not is_unknown(attempt.value):
            break
        question = _open_question(ctx, attempt, asked, getattr(inner, "_declined", ()))
        if question is None:
            break
        answer = inner.realize(question, thinking=True).value
        _note_thought(ctx, inner, question, answer)
        if isinstance(answer, Call) and answer.concept == "Taught":
            attempt = inner.realize(about, thinking=True)
        # A question that taught us nothing costs a round and no more. It is
        # already marked asked, so the next round moves on to the next hole
        # rather than putting the same one to the same model again. Which
        # matters: a verb is often only unresolvable because of the noun
        # underneath it, and that noun is the round after.
    ctx.result.trace.extend(attempt.trace)
    ctx.result.learned.extend(attempt.learned)
    return attempt.value


def _open_question(ctx: Context, attempt, asked: set, declined) -> Optional[Expr]:
    """The one thing worth asking yourself next.

    A missing verb first, because that is the shape of the question; then a
    name we hold nothing about, because that is what the verb would have
    been working on.
    """
    from .teacher import Teacher

    for gap in sorted(attempt.gaps, key=lambda g: (-g.arity, g.concept)):
        if gap.concept in asked:
            continue
        asked.add(gap.concept)
        if gap.concept in declined:
            # Realization already put this exact signature to the seat on
            # the way here and came back with nothing.
            continue
        head = Teacher(ctx.knowledge).ask(gap).head
        # Where the word turned up, carried along, because the signature on
        # its own is not enough to read some words by.
        return Call("Teach", (Arg("concept", head), Arg("context", gap.expr)))
    name = _unclassified(ctx, attempt.value, asked)
    if name is None:
        return None
    asked.add(name.concept)
    return Call("Kind", (Arg("of", name),))


def _unclassified(ctx: Context, expr: Expr, asked: set) -> Optional[Call]:
    """A bare name in the unresolved part whose kind we do not know.

    Deliberately not "a name we hold no facts about". That chess is being
    played by Kate is a fact mentioning chess and tells us nothing
    whatsoever about what chess is, which is the thing standing between us
    and the answer.
    """
    from .expr import walk

    for node in walk(expr):
        if not isinstance(node, Call) or node.args:
            continue
        name = node.concept
        if name in asked or name in _SELF_EVIDENT or name in _HOLES:
            continue
        cd = ctx.knowledge.concept(name)
        if cd is not None and cd.kind not in ("concept", "entity", "thing"):
            # A seeded operation or attribute is not a mystery noun.
            continue
        if ctx.knowledge.ancestors(name):
            continue
        return node
    return None


def _note_thought(ctx: Context, inner, question: Expr, answer: Expr) -> None:
    """Both halves of a round, kept. `:why` should show you the thinking."""
    from .discourse import Turn as DiscourseTurn

    ctx.result.trace.append(
        "asked myself %s -> %s" % (render(question, False), render(answer, False))
    )
    inner.discourse.record(
        DiscourseTurn(
            heard=render(question, False),
            meaning=question,
            answer=answer,
            said="",
        )
    )


def _assume(ctx: Context, given: Optional[Expr]) -> List:
    """Believe the scene, but only for as long as the question lasts."""
    if given is None:
        return []
    items = _items(given) or ([given] if isinstance(given, Call) else [])
    added = []
    for item in items:
        if not isinstance(item, Call):
            continue
        if any(f.proposition == item for f in ctx.knowledge.facts):
            # Already believed for better reasons than this. Leave it be.
            continue
        added.append(
            ctx.knowledge.assert_fact(item, True, Evidence(source="given", confidence=0.5))
        )
    return added


def _forget_assumed(ctx: Context, assumed: List) -> None:
    if not assumed:
        return
    ctx.knowledge.facts = [f for f in ctx.knowledge.facts if f not in assumed]


def _scene_listing(about: Expr, assumed: List) -> Optional[Expr]:
    """The givens, when the question was about the group they describe.

    `Activities(subject=AllOf(Sisters(count=5)))` plus a list of who is
    doing what is not a missing verb. It is a request to read the scene
    back. A question about one of them (`Doing(subject=Fifth(...))`) still
    has to be worked out, because the scene does not name her.
    """
    if not assumed or not isinstance(about, Call):
        return None
    target = about.first("subject", "of", "collection", "about") or about
    if not _groupish(target):
        return None
    props = []
    for fact in assumed:
        proposition = getattr(fact, "proposition", fact)
        if isinstance(proposition, Call) and proposition.concept == "Unknown":
            continue
        props.append(proposition)
    return Seq(tuple(props)) if props else None


def _groupish(expr: Expr) -> bool:
    if not isinstance(expr, Call):
        return False
    if expr.concept in ("AllOf", "Each", "Every", "Everyone", "Them", "They"):
        return True
    return expr.get("count") is not None or expr.get("n") is not None


@native("Turn", special=True)
def _turn(ctx: Context, c: Call) -> Optional[Expr]:
    """One public turn: Hear, realize, Speak. Teacher steps in on a gap."""
    text = _as_text(c.first("text", "of", "utterance"))
    if text is None:
        return None
    # What was actually said, kept for the one thing that needs the English
    # rather than the meaning: asking the chair to pick when nothing else could.
    ctx.realizer._utterance = text
    wrapped = ctx.realize(
        Call("Hear", (Arg("text", Lit(text)), Arg("as", Call("Utterance", ()))))
    )
    expr = _unwrap_meaning(wrapped)
    heard = ctx.result.heard
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is not None and discourse.pending_ask is not None:
        handled = _try_ask(ctx, expr, heard, text)
        if handled is not None:
            return handled
    if getattr(ctx.realizer, "lesson", None) is not None:
        handled = _try_lesson(ctx, expr, heard, text)
        if handled is not None:
            return handled
    return _reply_to(ctx, expr, heard, text)


def _reply_to(ctx: Context, expr: Expr, heard, text: str) -> Expr:
    expr = _pre_resolve(ctx, expr)
    if isinstance(expr, Call) and expr.concept in _PURE_SPEECH:
        # Failing to read the sentence is not a reason to answer nothing. If
        # the sentence came with options, one of them is still the answer.
        settled = _settle_utterance(ctx, text)
        if settled is not None:
            said = _speak_of(ctx, settled)
            return _finish_turn(ctx, settled, said, heard, text, remember=True, keep_last=True)
        reason = getattr(heard, "reason", "") if heard is not None else ""
        if reason and not reason.lower().startswith("unparseable"):
            said = reason
        else:
            said = _speak_of(ctx, expr)
        return _finish_turn(ctx, expr, said, heard, text, remember=False, keep_last=False)
    previous = ctx.realizer._asking
    ctx.realizer._asking = isinstance(expr, Call) and expr.concept in ("Question", "Query")
    try:
        value = ctx.realize(expr)
    finally:
        ctx.realizer._asking = previous
    if ctx.result.gaps or _unknown_about(value) is not None:
        thought = _think_it_through(ctx, expr, value)
        if thought is not None:
            value = thought
            del ctx.result.gaps[:]
    taught: List[str] = []
    if ctx.result.gaps:
        taught = _fill_gaps_from_seat(ctx)
        if taught:
            del ctx.result.gaps[:]
            value = ctx.realize(expr)
    if ctx.result.gaps and _is_choice(expr):
        del ctx.result.gaps[:]
        value = ctx.realize(expr)
    elif ctx.result.gaps:
        seat = ctx.realizer.seat
        if seat is not None and getattr(seat, "available", True):
            # A word the model could not define is not a question for you.
            # You get asked where you live. You do not get asked what Do means.
            del ctx.result.gaps[:]
            about = expr.first("about", "action", "of") if isinstance(expr, Call) else expr
            value = unknown(about if about is not None else expr)
            said = _speak_of(ctx, value)
            return _finish_turn(ctx, value, said, heard, text, remember=True, keep_last=True)
        return _ask_teacher(ctx, value, heard, text)
    value = _collapse_unknown(value)
    value = _pick_from_utterance(value, text)
    if is_unknown(value) or _unknown_about(value) is not None:
        # Last net of all. Everything Soup knows has been tried, the Teacher
        # has had its go, and the world has been asked. A shrug helps nobody,
        # so if the sentence offered options the chair picks one.
        settled = _settle_utterance(ctx, text)
        if settled is not None:
            value = settled
    if ctx.result.learned and (is_unknown(value) or _unknown_about(value) is not None):
        names = [line.split("(", 1)[0].strip() for line in ctx.result.learned]
        _unfile_failed_lessons(ctx, names)
    about = _unknown_about(value)
    discourse = getattr(ctx.realizer, "discourse", None)
    yours = only_you_know(about)
    if yours is not None:
        # The one thing no lookup can settle, so it goes back to the person
        # as a question rather than out as a shrug.
        if discourse is not None:
            discourse.pending_ask = about
            discourse.pending_ask_from = expr
        said = _speak_of(ctx, yours)
        return _finish_turn(ctx, yours, said, heard, text, remember=False, keep_last=True)
    said = _speak_of(ctx, value)
    if ctx.result.learned:
        said += " (worked out %s for myself)" % "; ".join(ctx.result.learned)
    return _finish_turn(ctx, value, said, heard, text, remember=True, keep_last=True)


def _collapse_unknown(value: Expr) -> Expr:
    """An answer with a hole in it is not an answer.

    `Doing(subject=Unknown(about=Nth(collection=Sisters(count=5), index=5)))`
    is a tree we got halfway through, and reading it out loud gets you
    "it's what nth of collection sisters of count 5, index 5 is's doing".
    Nobody is served by that. Say the honest thing, about the question that
    was actually asked rather than about the wreckage of answering it.
    """
    if not isinstance(value, Call) or value.concept != "Answer":
        return value
    answer = value.get("value")
    asked = value.get("to")
    if answer is None or asked is None or is_unknown(answer):
        return value
    if not any(is_unknown(n) for n in walk(answer)):
        return value
    return Call("Answer", (Arg("value", unknown(asked)), Arg("to", asked)))


def _think_it_through(ctx: Context, expr: Expr, value: Expr) -> Optional[Expr]:
    """A hole is a reason to think before it is a reason to ask.

    Only for holes thinking could possibly close. Where somebody lives is
    not one of those, and grinding on it would be both rude and useless.
    """
    seat = ctx.realizer.seat
    if seat is None or not getattr(seat, "available", True):
        return None
    if only_you_know(_unknown_about(value)) is not None:
        return None
    thought = ctx.realize(Call("Solve", (Arg("about", expr),)))
    if _is_choice(expr):
        if not is_unknown(thought) and _unknown_about(thought) is None and thought != value:
            ctx.result.trace.append("thought it through rather than asking")
            return thought
        retry = ctx.realize(expr)
        if not is_unknown(retry) and retry != value:
            ctx.result.trace.append("thought it through rather than asking")
            return retry
        return None
    if is_unknown(thought) or _unknown_about(thought) is not None:
        return None
    if thought == value:
        return None
    ctx.result.trace.append("thought it through rather than asking")
    return thought


def _speak_of(ctx: Context, value: Expr) -> str:
    spoken = ctx.realize(Call("Speak", (Arg("of", value),)))
    if isinstance(spoken, Lit):
        return str(spoken.value)
    return _say(ctx, value)


def _finish_turn(
    ctx: Context,
    value: Expr,
    said: str,
    heard,
    text: str,
    remember: bool,
    keep_last: bool,
) -> Expr:
    ctx.result.said = said
    ctx.result.heard = heard
    _record_turn(ctx, text, heard, value, said, remember)
    if keep_last:
        ctx.realizer.last_result = ctx.result
    return value


def _record_turn(ctx: Context, utterance: str, heard, answer: Expr, said: str, remember: bool) -> None:
    from .discourse import Turn as DiscourseTurn

    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None:
        return
    discourse.record(
        DiscourseTurn(
            heard=utterance,
            meaning=getattr(heard, "expr", None) if heard is not None else None,
            answer=answer,
            said=said,
        )
    )
    if remember:
        discourse.note_answer(answer)


def _pre_resolve(ctx: Context, expr: Expr) -> Expr:
    if not (isinstance(expr, Call) and expr.concept == "Explain"):
        return expr
    target = expr.first("concept", "about")
    if not (isinstance(target, Call) and target.concept == "Ref"):
        return expr
    prior = getattr(ctx.realizer, "last_result", None)
    if prior is None:
        return call("Unknown", about=call("Ref", Lit("that")))
    if not prior.trace:
        return call(
            "Explanation",
            concept=call("Ref", Lit("that")),
            gloss=Lit("nothing to unpack, that one resolved in one step"),
        )
    return call(
        "Explanation",
        concept=call("Ref", Lit("that")),
        gloss=Lit("here is how i got there:"),
        rules=Seq(tuple(Lit(line) for line in prior.trace)),
    )


def _fill_gaps_from_seat(ctx: Context) -> List[str]:
    """Ask the model what the missing words mean, never the person.

    The learn-budget is for one pass of realization. Leftover gaps after
    that still belong to the Teacher in the chair, not to the human. A
    definition that survives is an ordinary rule, same as if they had
    typed it.
    """
    seat = ctx.realizer.seat
    if seat is None or not getattr(seat, "available", True):
        return []
    from .teacher import Teacher

    teacher = Teacher(ctx.knowledge)
    declined = getattr(ctx.realizer, "_declined", None)
    if declined is None:
        declined = set()
        ctx.realizer._declined = declined
    taught: List[str] = []
    seen = set()
    for gap in list(ctx.result.gaps):
        if gap.concept in seen or gap.concept in declined:
            continue
        seen.add(gap.concept)
        lesson = teacher.ask(gap)
        meaning = seat.define(
            lesson.signature(),
            list(ctx.knowledge.concepts),
            context=render(gap.expr, False),
        )
        if meaning is None or teacher.learn(lesson, meaning) is None:
            declined.add(gap.concept)
            continue
        ctx.result.learned.append(
            "%s := %s" % (lesson.signature(), render(meaning, False))
        )
        taught.append(gap.concept)
    return taught


def _unfile_failed_lessons(ctx: Context, taught: List[str]) -> None:
    for name in taught:
        ctx.knowledge.drop_learned_rule(name)
    ctx.result.learned = [
        line for line in ctx.result.learned if not any(line.startswith(n) for n in taught)
    ]


def _pick_from_utterance(value: Expr, text: str) -> Expr:
    """Last chance: a realized value that matches A./B. lines in the English."""
    if is_unknown(value) or _unknown_about(value) is not None:
        return value
    if isinstance(value, Call) and value.concept == "Answer" and value.get("letter") is not None:
        return value
    from .ears import split_mcq

    parsed = split_mcq(text)
    if parsed is None:
        return value
    _stem, opts = parsed
    options = [(letter, option, Lit(option)) for letter, option in opts]
    inner = value.get("value") if isinstance(value, Call) and value.concept == "Answer" else value
    picked = _match_among(inner if inner is not None else value, options)
    if picked is None:
        return value
    letter, option, _expr = picked
    stem = value.get("to") if isinstance(value, Call) else value
    return Call(
        "Answer",
        (
            Arg("value", Lit(option)),
            Arg("letter", Lit(letter)),
            Arg("to", stem if stem is not None else value),
        ),
    )


def _ask_teacher(ctx: Context, value: Expr, heard, text: str) -> Expr:
    from .teacher import Teacher

    gap = sorted(ctx.result.gaps, key=lambda g: (-g.arity, g.concept))[0]
    teacher = Teacher(ctx.knowledge)
    lesson = teacher.ask(gap)
    ctx.realizer.lesson = lesson
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is not None:
        discourse.pending_teach = gap.concept
        discourse.pending_param = lesson.params[0] if lesson.params else None
    said = teacher.question(lesson)
    return _finish_turn(ctx, value, said, heard, text, remember=False, keep_last=True)


def _try_lesson(ctx: Context, expr: Expr, heard, text: str) -> Optional[Expr]:
    from .teacher import Teacher

    lesson = ctx.realizer.lesson
    if lesson is None:
        return None
    if isinstance(expr, Call) and expr.concept in ("Unintelligible", "Deny", "Farewell"):
        _clear_lesson(ctx)
        if expr.concept != "Unintelligible":
            return None
        said = "alright, never mind %s then" % lesson.concept
        ctx.result.said = said
        ctx.result.heard = heard
        return expr
    teacher = Teacher(ctx.knowledge)
    if not teacher.is_answer(lesson, expr):
        return None
    taught = teacher.learn(lesson, expr)
    if taught is None:
        _clear_lesson(ctx)
        return None
    _clear_lesson(ctx)
    said = _speak_of(ctx, taught)
    retry = _replay_lesson(ctx, lesson)
    if retry is not None:
        said = "%s. %s" % (said, retry)
    ctx.result.said = said
    ctx.result.heard = heard
    _record_turn(ctx, "", None, taught, said, remember=False)
    return taught


def _clear_lesson(ctx: Context) -> None:
    ctx.realizer.lesson = None
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None:
        return
    discourse.pending_teach = None
    discourse.pending_param = None


_HOLES = frozenset({"Where", "Who", "What", "When", "Which", "How"})


def _unknown_about(value: Optional[Expr]) -> Optional[Expr]:
    while isinstance(value, Call) and value.concept in ("Answer", "Assertion"):
        value = value.get("value") or value.get("truth") or value.get("about")
    if isinstance(value, Call) and value.concept == "Unknown":
        return value.get("about")
    return None


def _ask_filler(expr: Expr) -> Optional[Expr]:
    if isinstance(expr, Lit):
        return expr
    if isinstance(expr, Call) and is_name(expr):
        return expr
    return None


def _plug_holes(expr: Expr, filler: Expr) -> Expr:
    if isinstance(expr, Call) and expr.concept in _HOLES:
        return filler
    if not isinstance(expr, Call):
        return expr
    args = tuple(Arg(a.name, _plug_holes(a.value, filler)) for a in expr.args)
    if args != expr.args:
        return Call(expr.concept, args)
    if expr.concept in ("Age", "Name") and not expr.has("value"):
        return Call(expr.concept, expr.args + (Arg("value", filler),))
    return expr


def _clear_ask(ctx: Context) -> None:
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None:
        return
    discourse.pending_ask = None
    discourse.pending_ask_from = None


def _replay_ask(ctx: Context, original: Optional[Expr]) -> Optional[str]:
    if original is None:
        return None
    retry = ctx.realizer.realize(original)
    if retry.gaps or is_unknown(retry.value) or _unknown_about(retry.value) is not None:
        return None
    ctx.realizer.last_result = retry
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is not None:
        discourse.note_answer(retry.value)
    return "so: %s" % _say(ctx, retry.value)


def _try_ask(ctx: Context, expr: Expr, heard, text: str) -> Optional[Expr]:
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None or discourse.pending_ask is None:
        return None
    about = discourse.pending_ask
    original = discourse.pending_ask_from
    if isinstance(expr, Call) and expr.concept in ("Unintelligible", "Deny", "Farewell"):
        _clear_ask(ctx)
        if expr.concept != "Unintelligible":
            return None
        said = "alright, never mind"
        ctx.result.said = said
        ctx.result.heard = heard
        return expr
    if isinstance(expr, Call) and expr.concept in (
        "Question",
        "Query",
        "Request",
        "Advice",
        "Teach",
        "Learn",
    ):
        return None
    if isinstance(expr, Call) and expr.concept in ("Remember", "Believe"):
        value = ctx.realize(expr)
        _clear_ask(ctx)
        said = _speak_of(ctx, value)
        retry = _replay_ask(ctx, original)
        if retry:
            said = "%s. %s" % (said, retry)
        return _finish_turn(ctx, value, said, heard, text, remember=False, keep_last=True)
    filler = _ask_filler(expr)
    if filler is None:
        return None
    fact = _plug_holes(about, filler)
    if fact is about:
        return None
    remembered = ctx.realize(Call("Remember", (Arg("proposition", fact),)))
    _clear_ask(ctx)
    said = _speak_of(ctx, remembered)
    retry = _replay_ask(ctx, original)
    if retry:
        said = "%s. %s" % (said, retry)
    return _finish_turn(
        ctx, remembered, said, heard, text, remember=False, keep_last=True
    )


def _replay_lesson(ctx: Context, lesson) -> Optional[str]:
    from .expr import concept_names

    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None:
        return None
    for turn in reversed(discourse.turns):
        if turn.meaning is None:
            continue
        if lesson.concept not in concept_names(turn.meaning):
            continue
        retry = ctx.realizer.realize(turn.meaning)
        if retry.gaps:
            return None
        ctx.realizer.last_result = retry
        discourse.note_answer(retry.value)
        return "so: %s" % _say(ctx, retry.value)
    return None


# ---------------------------------------------------------------------------
# speech acts
# ---------------------------------------------------------------------------


@native("Choose", "Select", "Pick", special=True)
def _choose(ctx: Context, c: Call) -> Optional[Expr]:
    """Which of these is the stem we just worked out.

    The options are candidates. The stem is the actual question. A missing
    verb here used to be taught as junk (`Select := Concat`) and then the
    turn gave up. Now we realize the stem, keep a definition only when it
    produces a value that matches, and speak the matching option.
    """
    options = _choice_options(c)
    if not options:
        return None
    stem = c.first("by", "about", "of")
    given = c.get("given")
    if stem is None:
        return unknown(c)
    if given is not None:
        value = ctx.realize(Call("Solve", (Arg("about", stem), Arg("given", given))))
    else:
        value = ctx.realize(stem)
    if isinstance(value, Call) and value.concept == "Answer":
        inner = value.get("value")
        if inner is not None:
            value = inner
    picked = _match_among(value, options)
    if picked is None:
        picked = _true_option(ctx, options)
    if picked is None:
        picked = _settle_from_seat(ctx, stem, options)
    if picked is None:
        return unknown(stem)
    letter, text, expr = picked
    return Call(
        "Answer",
        (
            Arg("value", Lit(text) if text else expr),
            Arg("letter", Lit(letter)),
            Arg("to", stem),
        ),
    )


def _choice_options(c: Call) -> list:
    items: List[Expr] = []
    among = c.first("among", "options")
    if isinstance(among, Seq):
        items.extend(among.items)
    elif among is not None:
        items.append(among)
    for name in ("between", "or"):
        extra = c.get(name)
        if extra is not None:
            items.append(extra)
    out = []
    for i, item in enumerate(items):
        letter = chr(ord("A") + i) if i < 10 else str(i)
        text = ""
        if isinstance(item, Lit):
            text = str(item.value)
        elif isinstance(item, Call) and item.concept == "Option":
            lit = item.first("letter", "as")
            if isinstance(lit, Lit) and lit.value:
                letter = str(lit.value).strip().upper()[:1]
            body = item.first("text", "value", "of")
            text = _as_text(body) or ""
            if not text and isinstance(body, Call):
                text = body.concept
        elif isinstance(item, Call):
            text = item.concept
        if letter and (text or item is not None):
            out.append((letter, text, item))
    return out


def _match_among(value: Optional[Expr], options: list) -> Optional[tuple]:
    if value is None or is_unknown(value):
        return None
    if isinstance(value, Seq) and len(value.items) != 1:
        return None
    if isinstance(value, Seq) and value.items:
        value = value.items[0]
    if isinstance(value, Call) and value.concept == "Answer":
        inner = value.get("value")
        if inner is not None:
            return _match_among(inner, options)
    num = _num(value)
    text = _as_text(value)
    if text is None and isinstance(value, Call) and not value.args:
        text = value.concept
    hits = []
    for letter, opt, expr in options:
        if text and _text_hits(text, opt):
            hits.append((letter, opt, expr))
        elif num is not None and _number_hits(num, opt):
            hits.append((letter, opt, expr))
    letters = {h[0] for h in hits}
    if len(letters) == 1:
        return hits[0]
    if text:
        exact = [h for h in hits if h[1].strip().lower() == text.strip().lower()]
        if len(exact) == 1:
            return exact[0]
    if num is not None:
        numbered = [h for h in options if _number_hits(num, h[1])]
        if len(numbered) == 1:
            return numbered[0]
        # No exact number match. Pick the option whose number is closest to
        # what we computed - strictly better than letting the 4B guess A.
        closest = _closest_number_option(num, options)
        if closest is not None:
            return closest
    return None


def _closest_number_option(computed: float, options: list) -> Optional[tuple]:
    """When a computed number is close but not exact, pick the nearest option.

    Only fires when at least two options contain numbers (a numeric question).
    Prefers the option whose leading number is closest by relative error.
    """
    scored = []
    for letter, opt, expr in options:
        n = _extract_leading_number(opt)
        if n is not None:
            scored.append((abs(n - computed), letter, opt, expr, n))
    if len(scored) < 2:
        return None
    scored.sort()
    best_dist, best_letter, best_opt, best_expr, best_n = scored[0]
    second_dist = scored[1][0]
    # Only pick if the best is meaningfully closer than the runner-up.
    if best_dist == 0:
        return (best_letter, best_opt, best_expr)
    if second_dist > 0 and best_dist / second_dist < 0.5:
        return (best_letter, best_opt, best_expr)
    return None


def _extract_leading_number(text: str) -> Optional[float]:
    """The first number in an option string, ignoring currency symbols."""
    m = re.search(r'[$]?\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m is None:
        return None
    raw = m.group(1).replace(',', '')
    try:
        return float(raw)
    except ValueError:
        return None


def _text_hits(text: str, option: str) -> bool:
    needle = text.strip().lower()
    hay = option.strip().lower()
    if not needle or not hay:
        return False
    if needle == hay:
        return True
    if len(needle) >= 4 and needle in hay:
        return True
    if len(hay) >= 4 and hay in needle:
        return True
    folded_n = re.sub(r"[^a-z0-9]", "", needle)
    folded_h = re.sub(r"[^a-z0-9]", "", hay)
    if folded_n and folded_n == folded_h:
        return True
    return False


def _number_hits(num: float, option: str) -> bool:
    token = str(int(num)) if float(num).is_integer() else str(num)
    return re.search(r"(?<![0-9.])%s(?![0-9])" % re.escape(token), option) is not None


def _settle_utterance(ctx: Context, text: str) -> Optional[Expr]:
    """The options as the person wrote them, settled by the chair.

    Reads the letters straight off the English rather than the concept tree,
    so it works even when the sentence could not be read at all. A question
    with options has an answer available by recognition; refusing to give one
    is the only way to be useless as well as unsure.
    """
    from .ears import split_mcq

    if not text:
        return None
    parsed = split_mcq(text)
    if parsed is None:
        return None
    stem_text, opts = parsed
    if not opts:
        return None
    options = [(letter, option, Lit(option)) for letter, option in opts]
    picked = _settle_from_seat(ctx, Lit(stem_text), options)
    if picked is None:
        return None
    letter, option, _expr = picked
    return Call(
        "Answer",
        (Arg("value", Lit(option)), Arg("letter", Lit(letter)), Arg("to", Lit(stem_text))),
    )


def _settle_from_seat(ctx: Context, stem: Expr, options: list) -> Optional[tuple]:
    """Out of ideas, with the options still in hand. Ask the chair to pick.

    Everything better has already been tried: the stem was realized, the
    Teacher was asked what the missing words mean, and the world was asked
    for the facts. What is left is recognition, which is the one thing a
    model is reliably good at and the only thing still on the table.
    """
    seat = ctx.realizer.seat
    settle = getattr(seat, "settle", None)
    if settle is None or not getattr(seat, "available", True):
        return None
    question = getattr(ctx.realizer, "_utterance", "") or render(stem, False)
    letter = settle(question, [(l, t) for l, t, _e in options])
    if letter is None:
        return None
    for candidate in options:
        if candidate[0] == letter:
            ctx.result.trace.append(
                "nothing of mine settled it, so the chair picked %s" % letter
            )
            ctx.result.guessed = True
            return candidate
    return None


def _true_option(ctx: Context, options: list) -> Optional[tuple]:
    """An option that is itself a proposition we can prove."""
    true_hits = []
    for letter, text, expr in options:
        if not isinstance(expr, Call) or expr.concept == "Option":
            continue
        value = ctx.realize(expr)
        if isinstance(value, Lit) and value.value is True:
            true_hits.append((letter, text, expr))
    if len(true_hits) == 1:
        return true_hits[0]
    return None


def _is_choice(expr: Optional[Expr]) -> bool:
    if not isinstance(expr, Call):
        return False
    if expr.concept in ("Choose", "Select", "Pick"):
        return True
    if expr.get("among") is not None or expr.get("options") is not None:
        return True
    if expr.concept in ("Question", "Query"):
        inner = expr.first("about", "of")
        return _is_choice(inner)
    return False


def _choice_answer(value: Expr, letter: str, text: str, stem: Expr) -> Call:
    return Call(
        "Answer",
        (
            Arg("value", Lit(text) if text else value),
            Arg("letter", Lit(letter)),
            Arg("to", stem),
        ),
    )


@native("Question", special=True)
def _question(ctx: Context, c: Call) -> Optional[Expr]:
    about = c.first("about", "of")
    if about is None:
        return None
    given = c.get("given")
    if (
        given is not None
        and isinstance(about, Call)
        and about.concept in ("Choose", "Select", "Pick")
        and about.get("given") is None
    ):
        about = Call(about.concept, about.args + (Arg("given", given),))
    if given is not None:
        # A question that arrives with a scene attached is a question to be
        # worked out, not looked up. The givens hold for the duration.
        value = ctx.realize(
            Call("Solve", (Arg("about", about), Arg("given", given)))
        )
        if isinstance(value, Call) and value.concept == "Answer":
            return value
        return Call("Answer", (Arg("value", value), Arg("to", about)))
    value = ctx.realize(about)
    if isinstance(value, Call) and value.concept in (
        "Acknowledged",
        "Taught",
        "Explanation",
        "Answer",
    ):
        return value
    if isinstance(value, Call) and value == about and is_name(value):
        # "who is Greg" and "what is military time" did not compute anything,
        # so say what we know of them rather than handing the name back as
        # though it were its own answer.
        recalled = ctx.realize(Call("Recall", (Arg("about", about),)))
        if isinstance(recalled, Call) and recalled.concept == "Explanation":
            return recalled
        # We hold nothing about it ourselves, so treat the bare name as the
        # question it plainly is: what is this thing? That is answerable
        # from facts, from a lookup or by a model, where a bare name is not.
        asking = Call("Identity", (Arg("subject", about),))
        described = ctx.realize(asking)
        if described != asking:
            return Call("Answer", (Arg("value", described), Arg("to", about)))
        return unknown(about)
    return Call("Answer", (Arg("value", value), Arg("to", about)))


@native("Request", special=True)
def _request(ctx: Context, c: Call) -> Optional[Expr]:
    action = c.first("action", "of")
    if action is None:
        return None
    value = ctx.realize(action)
    if isinstance(value, Call) and value.concept in ("Acknowledged", "Taught", "Explanation"):
        return value
    return Call("Answer", (Arg("value", value), Arg("to", action)))


@native("What", "Who", "Which")
def _wh(ctx: Context, c: Call) -> Optional[Expr]:
    """"what is the boiling point of water" is answered by the boiling point.

    The wh-word marks the sentence as a question; it is not itself a further
    question. So once the thing it wraps has an answer, the wh-word has
    nothing left to add and gets out of the way.
    """
    inner = c.get("proposition")
    if inner is not None and not isinstance(inner, Call):
        return inner
    return None


@native("Identity")
def _identity(ctx: Context, c: Call) -> Optional[Expr]:
    """Who someone is.

    Asked about Soup it is a finished answer and the mouth has the words for
    it. Asked about a named person it is an ordinary lookup. Asked about a
    clause (Ensure(that=Not(Carry(...)))) Identity has nothing to add: the
    clause is what we realize, piece by piece, so the Teacher can learn the
    verbs instead of a model answering the blob.
    """
    from .seat import _packed_name

    subject = c.first("subject", "of")
    if isinstance(subject, Call) and subject.concept in ("Assistant", "User"):
        return c
    if isinstance(subject, Call) and subject.concept == "Concept":
        inner = subject.first("name", "of", "value")
        if inner is not None:
            return ctx.realize(Call("Identity", (Arg("subject", inner),)))
    if isinstance(subject, Lit):
        return subject
    if isinstance(subject, Call) and (subject.args or _packed_name(subject.concept)):
        return subject
    return None


@native(
    "Capability",
    "State",
    "Explanation",
    "Taught",
    "Answer",
    "Assertion",
    "Acknowledged",
    "Meaning",
    "Advice",
    "Opinion",
    special=True,
)
def _self_describing(ctx: Context, c: Call) -> Optional[Expr]:
    """Finished answers. Mouth knows what to do with these, and realizing
    their insides again would only chew on things we are merely reporting."""
    return c


@native("Teach", "Learn", special=True)
def _teach(ctx: Context, c: Call) -> Optional[Expr]:
    head = c.first("concept", "of")
    body = c.get("meaning") or c.get("as") or c.get("rule")
    if head is None:
        rule = c.get("rule")
        if rule is not None:
            ctx.knowledge.assert_fact(rule, evidence=Evidence(source="user"))
            return Call("Acknowledged", (Arg("about", rule),))
        return None
    if not isinstance(head, Call):
        return None
    if body is None:
        return _define_from_seat(ctx, head, c.get("context"))
    ctx.knowledge.add_rule(head, body, Evidence(source="teacher"))
    ctx.effect("learned %s" % head.concept)
    return Call("Taught", (Arg("concept", head), Arg("as", body)))


def _define_from_seat(
    ctx: Context, head: Call, context: Optional[Expr] = None
) -> Optional[Expr]:
    """Teach(concept=Quintuple(x)) with no body: ask what it means."""
    from .teacher import Teacher

    seat = ctx.realizer.seat
    if seat is None or not getattr(seat, "available", True):
        return None
    teacher = Teacher(ctx.knowledge)
    lesson = teacher.ask(Gap(head.concept, head))
    meaning = seat.define(
        lesson.signature(),
        list(ctx.knowledge.concepts),
        context=render(context, False) if context is not None else None,
    )
    if meaning is None:
        return None
    taught = teacher.learn(lesson, meaning)
    if taught is None:
        return None
    ctx.effect("learned %s" % head.concept)
    return taught


@native("Forget", special=True)
def _forget(ctx: Context, c: Call) -> Optional[Expr]:
    about = c.first("about", "of")
    if not isinstance(about, Call):
        return None
    name = about.concept
    before = len(ctx.knowledge.facts)
    ctx.knowledge.facts = [
        f for f in ctx.knowledge.facts if name not in _names_in(f.proposition)
    ]
    ctx.knowledge.rules.pop(name, None)
    ctx.effect("forgot %d things about %s" % (before - len(ctx.knowledge.facts), name))
    return Call("Acknowledged", (Arg("about", Call("Forget", (Arg("about", about),))),))


@native("Explain", special=True)
def _explain(ctx: Context, c: Call) -> Optional[Expr]:
    target = c.first("concept", "about", "of")
    if not isinstance(target, Call):
        return None
    name = target.concept
    cd = ctx.knowledge.concept(name)
    if cd is None:
        ctx.gap(name, target, "never heard of it")
        return Call("Unknown", (Arg("about", target),))
    rules = [Lit(r.source_text()) for r in ctx.knowledge.rules_for(name)]
    relations = [
        Call(e.relation, (Arg(None, Call(e.target if e.source == name else e.source, ())),))
        for e in ctx.knowledge.edges_for(name)
    ]
    facts = [f.proposition for f in ctx.knowledge.facts_mentioning(name) if f.truth]
    return Call(
        "Explanation",
        (
            Arg("concept", target),
            Arg("gloss", Lit(cd.gloss)),
            Arg("rules", Seq(tuple(rules))),
            Arg("relations", Seq(tuple(relations))),
            Arg("facts", Seq(tuple(facts))),
        ),
    )


@native("AllOf", special=True)
def _all_of(ctx: Context, c: Call) -> Optional[Expr]:
    """The members of a kind, gathered from what we have been told."""
    kind = c.first("kind", "of")
    if not isinstance(kind, Call):
        return None
    members: List[Expr] = []
    probe = call("IsA", subject=Var("_m"), kind=Call(kind.concept, ()))
    for fact, bindings in ctx.knowledge.query(probe):
        if fact.truth and "_m" in bindings and bindings["_m"] not in members:
            members.append(bindings["_m"])
    plural = call("Is", subject=Var("_m"), value=Call(kind.concept, ()))
    for fact, bindings in ctx.knowledge.query(plural):
        if fact.truth and "_m" in bindings and bindings["_m"] not in members:
            members.append(bindings["_m"])
    if not members:
        return unknown(c)
    return Seq(tuple(members))


@native("Seq", "Sequence", special=True)
def _seq(ctx: Context, c: Call) -> Optional[Expr]:
    """Do these in order. Each step's result is kept; `it` is the latest one."""
    results: List[Expr] = []
    discourse = getattr(ctx.realizer, "discourse", None)
    for step in (a.value for a in c.args):
        value = ctx.realize(step)
        results.append(value)
        if discourse is not None:
            discourse.note_answer(value)
        if is_unknown(value):
            return value
    if not results:
        return Seq(())
    if len(results) == 1:
        return results[0]
    out = Seq(tuple(results))
    if discourse is not None:
        discourse.last_collection = out
        discourse.last_value = results[-1]
    return out


@native("Ref", special=True)
def _ref(ctx: Context, c: Call) -> Optional[Expr]:
    word = _as_text(c.args[0].value if c.args else None) or "it"
    discourse = getattr(ctx.realizer, "discourse", None)
    if discourse is None:
        return unknown(c)
    w = word.lower()
    if w in ("it", "that", "this", "one") and discourse.last_value is not None:
        return ctx.realize(discourse.last_value)
    resolved = discourse.resolve_pronoun(word)
    if resolved is None:
        return unknown(c)
    return ctx.realize(resolved)


def _names_in(expr: Expr) -> List[str]:
    from .expr import concept_names

    return concept_names(expr)


# ---------------------------------------------------------------------------
# the starting vocabulary
# ---------------------------------------------------------------------------

_KINDS = {
    "entity": [
        ("User", "the person talking to me"),
        ("Assistant", "me, Soup"),
        ("Nothing", ""),
        ("File", "a file on disk"),
        ("Directory", "a folder on disk"),
        ("Folder", "a folder on disk"),
        ("Object", "a bundle of named fields, as json becomes"),
        ("Program", "source, to be heard and not done"),
        ("Document", "text, to be heard and not done"),
        ("Thought", "a conversation you are not in"),
        ("Utterance", "a turn addressed to you, heard and not yet done"),
        ("Self", "this soup, thinking"),
        ("Seat", "the model in the chair"),
        ("Agent", "someone else to consult"),
    ],
    "value": [("Number", ""), ("Text", ""), ("Truth", ""), ("List", "")],
    "operation": [
        "Add",
        "Plus",
        "Sum",
        "Multiply",
        "Times",
        "Product",
        "Subtract",
        "Minus",
        "Difference",
        "Divide",
        "DividedBy",
        "Power",
        "Modulo",
        "Negate",
        "Round",
        "Average",
        "Map",
        "Filter",
        "First",
        "Last",
        ("Nth", "the one at this position, counting from 1"),
        "Count",
        "Maximum",
        "Minimum",
        "Sort",
        "Reverse",
        "Range",
        "Double",
        "Triple",
        "Half",
        "Square",
        "Increment",
        "Decrement",
        "Let",
        "Query",
        "Find",
        "Ask",
        "Remember",
        "Recall",
        ("Python", "run a bit of python on this machine"),
        ("Shell", "run a shell command on this machine"),
        ("Bash", "Shell"),
        ("Seq", "do these in order; it is the last result"),
        ("Sequence", "Seq"),
        "Read",
        "Write",
        "Files",
        "Delete",
        ("WebSearch", "search the web and return results with links as Markdown"),
        ("VisitWebpage", "fetch a page's static text as Markdown; JavaScript is not run"),
        ("WikipediaSearch", "find a Wikipedia article and return its introduction and link"),
        ("TranscribeAudio", "turn an audio file or URL into text with a local speech model"),
        "Fetch",
        "Url",
        "Json",
        "GetProperty",
        "HasKey",
        "At",
        "Field",
        "Join",
        "Concat",
        "Replace",
        "Lowercase",
        "PathOf",
        "Create",
        "Change",
        "Edit",
        "Put",
        "Hear",
        "Speak",
        "Teach",
        "Learn",
        ("Kind", "what sort of thing a name is"),
        ("Solve", "think it out in rounds, learning what is missing"),
        ("Reason", "Solve"),
        ("Choose", "pick which of these matches what we worked out"),
        ("Select", "Choose"),
        ("Pick", "Choose"),
    ],
    "relation": [
        "WikidataSearchResult",
        "WikidataStatement",
        "WikidataEntityLabel",
        "GreaterThan",
        "LessThan",
        "EqualTo",
        "NotEqualTo",
        "And",
        "Or",
        "Not",
        "If",
        "Conditional",
        "Has",
        "Likes",
        "Knows",
        "OlderThan",
        "YoungerThan",
        "BiggerThan",
        "PartOf",
        "IsA",
        "Is",
        "Was",
        "InstanceOf",
        "OppositeOf",
        "SimilarTo",
        "RealizedBy",
        "HasProperty",
        "InverseOf",
    ],
    "property": [
        ("Symmetric", "holds just as well the other way round"),
        ("Transitive", "carries on down the chain"),
        ("Taxonomic", "points at a more general kind"),
    ],
    "attribute": [
        "Age",
        "Name",
        "Size",
        "Color",
        "Height",
        "Weight",
        "Price",
        "FileSize",
        "Location",
        "Job",
        ("Path", "where a file or folder lives"),
        ("Contents", "what a file holds"),
        ("Source", "the text a thing should be written as"),
        ("Time", "what time it is now, on an ordinary 12-hour clock"),
        ("TwentyFourHourTime", "what time it is now, on a 24-hour clock"),
        ("Timezone", "the zone this clock is set to"),
        "Even",
        "Odd",
    ],
    "quality": ["Big", "Small", "Old", "Young", "Good", "Bad", "Fast", "Slow"],
    "modality": ["Probably", "Certainly", "Maybe", "Must", "Can", "Should"],
    "speech": [
        "Request",
        "Assertion",
        "Answer",
        "Greeting",
        "Farewell",
        "Laugh",
        "Thanks",
        "Affirm",
        "Deny",
        "Unintelligible",
        "Acknowledged",
        "Unknown",
        "Who",
        "What",
        "Which",
        "Where",
        "When",
        "Why",
        "Explain",
        "Teach",
        "Say",
        "Speak",
        "Hear",
        "Meaning",
        "Turn",
        "Think",
        "Consult",
        "Conversation",
        "Subagent",
        "Identity",
        "Capability",
        "Opinion",
        "Advice",
    ],
}

# Short natural-language meanings for the built-in vocabulary. These are
# shown to the ears and included in Explain; implementation names alone are
# not enough for a model to choose the right concept.
_BUILTIN_GLOSSES = {
    # Values and entities.
    "Nothing": "the absence of a thing or value",
    "List": "an ordered group of values",
    "Number": "a numeric value",
    "Text": "a piece of written or spoken text",
    "Truth": "a true or false value",
    # Arithmetic and collection operations.
    "Add": "combine values by addition",
    "Plus": "add values together",
    "Sum": "the total produced by addition",
    "Multiply": "multiply numeric values",
    "Times": "multiply one value by another",
    "Product": "the result of multiplication",
    "Subtract": "take one value away from another",
    "Minus": "subtract one value from another",
    "Difference": "the result of subtraction",
    "Divide": "divide one value by another",
    "DividedBy": "divide one value by another",
    "Power": "raise a base value to an exponent",
    "ToThePowerOf": "raise a base value to an exponent",
    "Modulo": "the remainder after division",
    "Remainder": "the remainder after division",
    "Negate": "change a number to its additive opposite",
    "Round": "round a number to the nearest integer",
    "Average": "the mean of a group of numbers",
    "Mean": "the average of a group of numbers",
    "Bigger": "test whether the left value is greater than the right",
    "GreaterThan": "test whether the left value is greater than the right",
    "MoreThan": "test whether the left value is greater than the right",
    "LessThan": "test whether the left value is less than the right",
    "Smaller": "test whether the left value is less than the right",
    "EqualTo": "test whether two values are equal",
    "Equals": "test whether two values are equal",
    "SameAs": "test whether two values are the same",
    "NotEqualTo": "test whether two values differ",
    "Different": "test whether two values differ",
    "Not": "reverse a true-or-false value",
    "And": "true when every condition is true",
    "Both": "require both conditions to be true",
    "Or": "true when at least one condition is true",
    "Either": "true when at least one condition is true",
    "Map": "apply an operation to every item in a collection",
    "Filter": "keep only collection items that pass a test",
    "Sort": "put collection items in order",
    "First": "the first item in a collection",
    "Head": "the first item in a collection",
    "Last": "the final item in a collection",
    "Nth": "the item at a numbered position, counting from one",
    "ItemAt": "the item at a numbered position, counting from one",
    "Count": "the number of items in a collection",
    "HowMany": "count the items in a collection",
    "Maximum": "the largest item, optionally by a measured property",
    "Minimum": "the smallest item, optionally by a measured property",
    "Reverse": "put collection items in the opposite order",
    "Range": "the inclusive sequence of integers between two endpoints",
    "Size": "the numeric size or item count of a value",
    "Length": "the number of items in a collection",
    "Even": "true when a whole number is divisible by two",
    "Odd": "true when a whole number is not divisible by two",
    "Let": "bind an intermediate value for use in a later expression",
    "If": "choose a result according to a condition",
    "Conditional": "choose a result according to a condition",
    "AtLeast": "a value greater than or equal to a threshold",
    "AtMost": "a value less than or equal to a threshold",
    "Double": "multiply a value by two",
    "Triple": "multiply a value by three",
    "Half": "divide a value by two",
    "Square": "multiply a value by itself",
    "Increment": "add one to a value",
    "Decrement": "subtract one from a value",
    # Knowledge, files, and structured data.
    "Query": "find facts that match a pattern",
    "Find": "find a value or record that matches a query",
    "Get": "retrieve a value from an object or collection",
    "Give": "return the value requested by a query",
    "Show": "return matching values so they can be seen",
    "Ask": "ask the user a question or check a proposition",
    "Remember": "store a proposition as something the user told Soup",
    "Recall": "retrieve remembered facts about a concept",
    "Learn": "teach Soup a concept or realization",
    "Teach": "give Soup a concept or realization to learn",
    "Create": "create files from a list of file descriptions",
    "Change": "replace text in a file and save the result",
    "Edit": "replace text in a file and save the result",
    "Put": "write a described file into a folder",
    "Read": "read the contents of a local file",
    "Write": "write text to a local file",
    "Files": "list the names of files in a folder",
    "Delete": "remove a local file or empty folder",
    "PathOf": "get the path belonging to a file or folder",
    "Join": "join path parts into one path",
    "Concat": "join pieces of text end to end",
    "Replace": "replace occurrences of one piece of text with another",
    "Lowercase": "convert text to lowercase",
    "At": "read a named field or numbered item from a value",
    "Field": "read a named field from an object",
    "GetProperty": "read a named field from structured data",
    "HasKey": "test whether an object contains a named field",
    "Json": "parse JSON text into structured values",
    "Url": "build a URL from its scheme, host, path, and query fields",
    "Fetch": "request a URL and return its response text",
    "WebSearch": "search the web and return linked results as Markdown",
    "SearchWeb": "search the web and return linked results as Markdown",
    "DuckDuckGoSearch": "search the web and return linked results as Markdown",
    "VisitWebpage": "fetch a page's static text as Markdown; JavaScript is not run",
    "BrowseWebpage": "fetch a page's static text as Markdown; JavaScript is not run",
    "ReadWebpage": "fetch a page's static text as Markdown; JavaScript is not run",
    "WikipediaSearch": "find a Wikipedia article and return its introduction",
    "SearchWikipedia": "find a Wikipedia article and return its introduction",
    "TranscribeAudio": "turn an audio file or URL into text with a local speech model",
    "SpeechToText": "turn audio into written words with a local speech model",
    "Transcribe": "turn audio into written words with a local speech model",
    "Hear": "translate an utterance into Soup's concept language",
    "Speak": "turn a concept result into words for the user",
    "Now": "the current time",
    "Date": "today's calendar date",
    "Today": "today's calendar date",
    "Day": "the current day of the week",
    "Weekday": "the current day of the week",
    "Year": "the current calendar year",
    # Relations and properties.
    "WikidataSearchResult": "a candidate returned by a Wikidata search",
    "WikidataStatement": "a Wikidata claim with its rank and qualifiers",
    "WikidataEntityLabel": "a saved English label for a Wikidata entity",
    "GreaterThan": "the left value is greater than the right",
    "LessThan": "the left value is less than the right",
    "EqualTo": "the two values are equal",
    "NotEqualTo": "the two values are different",
    "And": "both related conditions hold",
    "Or": "at least one related condition holds",
    "Not": "the condition does not hold",
    "If": "a conditional relation between a condition and its result",
    "Conditional": "a conditional relation between a condition and its result",
    "Has": "a subject possesses or contains an object",
    "HasProperty": "a concept has a named property",
    "InstanceOf": "an individual belongs to a kind",
    "InverseOf": "one relation reverses the direction of another",
    "Is": "a subject has a stated value or identity",
    "IsA": "an individual or kind belongs to a more general kind",
    "Was": "a relation or state that held in the past",
    "Knows": "one person or system has knowledge of something",
    "Likes": "one person or thing has a positive preference for another",
    "OlderThan": "the left person is older than the right",
    "YoungerThan": "the left person is younger than the right",
    "BiggerThan": "the left thing is larger than the right",
    "PartOf": "one thing is a component or member of another",
    "OppositeOf": "the two concepts have contrasting meanings",
    "SimilarTo": "the two concepts have related meanings",
    "RealizedBy": "one concept is carried out by another operation",
    # Attributes, qualities, and modalities.
    "Age": "how old a person or thing is",
    "Color": "the color of a person or thing",
    "Even": "whether a whole number is divisible by two",
    "Odd": "whether a whole number is not divisible by two",
    "FileSize": "how much storage a file occupies",
    "Height": "how tall a person or thing is",
    "Job": "the work or occupation of a person",
    "Location": "where a person or thing is",
    "Name": "what a person or thing is called",
    "Price": "how much something costs",
    "Size": "how large a thing is",
    "Weight": "how heavy a person or thing is",
    "Bad": "having poor or undesirable qualities",
    "Big": "large in size or degree",
    "Fast": "moving or happening at high speed",
    "Good": "having desirable qualities",
    "Old": "having existed for a long time",
    "Slow": "moving or happening at low speed",
    "Small": "limited in size or degree",
    "Young": "having existed for a short time",
    "Can": "possible or able to happen",
    "Certainly": "regarded as definitely true",
    "Maybe": "possibly true, but uncertain",
    "Must": "required or necessarily true",
    "Probably": "more likely true than not",
    "Should": "advisable or expected",
    # Speech acts and explanations.
    "Acknowledged": "confirmation that a message or fact was received",
    "Advice": "a suggested action or course of thought",
    "Affirm": "a positive answer or agreement",
    "Answer": "a value or response that addresses a question",
    "Assertion": "a proposition stated as true or false",
    "Capability": "something Soup or another agent can do",
    "Consult": "ask another agent to think about a question",
    "Conversation": "a nested exchange of thought with an agent",
    "Deny": "a negative answer or disagreement",
    "Explain": "show the meaning, rules, relations, and facts for a concept",
    "Farewell": "a goodbye",
    "Greeting": "a hello or other opening acknowledgement",
    "Identity": "who or what a named person or thing is",
    "Laugh": "an expression of amusement",
    "Meaning": "an expression described as a particular kind of thought",
    "Opinion": "a belief or judgment that may be subjective",
    "Request": "an instruction for Soup to carry out",
    "Say": "express something in words",
    "Subagent": "a helper agent working on a nested question",
    "Thanks": "an expression of gratitude",
    "Think": "work through a question without speaking the result",
    "Turn": "one complete exchange: hear, realize, and speak",
    "Unintelligible": "utterance the ears could not turn into meaning",
    "Unknown": "a value or fact Soup could not determine",
    "What": "asks for the value or identity of something",
    "When": "asks for a time or date",
    "Where": "asks for a place or location",
    "Which": "asks which item or choice fits",
    "Who": "asks for a person or identity",
    "Why": "asks for a reason or cause",
    # Named internal rules in the Wikidata source.
    "Understand": "read a file as a program and translate it into meaning",
    "WikidataAliasMatches": "filter search results to exact alias matches",
    "WikidataBestStatement": "choose the preferred current Wikidata statement",
    "WikidataCandidates": "keep non-deprecated current statements when available",
    "WikidataClaims": "fetch claims for one Wikidata entity and property",
    "WikidataCurrent": "keep statements without an end-date qualifier",
    "WikidataEntities": "fetch labels for a batch of Wikidata entities",
    "WikidataExactAlias": "test whether a search match is an exact alias",
    "WikidataExactLabel": "test whether a search match is an exact label",
    "WikidataHasNoEndDate": "test whether a statement lacks an end date",
    "WikidataLabelMatches": "filter search results to exact label matches",
    "WikidataNonDeprecated": "remove deprecated Wikidata statements",
    "WikidataNotDeprecated": "test whether a statement is not deprecated",
    "WikidataPreferred": "test whether a statement has preferred rank",
    "WikidataPreferredStatements": "keep preferred-rank candidate statements",
    "WikidataSearch": "search Wikidata for entities or properties",
    "WikidataSearchHit": "choose an exact label match, then an exact alias",
    "WikidataSearchResults": "get the candidate list from a Wikidata search",
}

# Realizations that are themselves just meaning, not code.
_CORE_RULES = [
    'WikidataSearch(text=t, kind=k) := Json(Fetch(Url(scheme="https", host="www.wikidata.org", path="/w/api.php", query=Object(action="wbsearchentities", search=t, type=k, language="en", uselang="en", limit=10, format="json", formatversion=1))))',
    'WikidataSearchResults(text=t, kind=k) := GetProperty(WikidataSearch(text=t, kind=k), "search")',
    'WikidataClaims(entity=e, property=p) := GetProperty(GetProperty(Json(Fetch(Url(scheme="https", host="www.wikidata.org", path="/w/api.php", query=Object(action="wbgetclaims", entity=e, property=p)))), "claims"), p)',
    'WikidataEntities(ids=i) := GetProperty(Json(Fetch(Url(scheme="https", host="www.wikidata.org", path="/w/api.php", query=Object(action="wbgetentities", ids=i, props="labels", languages="en")))), "entities")',
    'WikidataExactLabel(t, h) := And(EqualTo(GetProperty(GetProperty(h, "match"), "type"), "label"), EqualTo(Lowercase(GetProperty(GetProperty(h, "match"), "text")), Lowercase(t)))',
    'WikidataExactAlias(t, h) := And(EqualTo(GetProperty(GetProperty(h, "match"), "type"), "alias"), EqualTo(Lowercase(GetProperty(GetProperty(h, "match"), "text")), Lowercase(t)))',
    'WikidataLabelMatches(results=r, text=t) := Filter(collection=r, where=WikidataExactLabel(t))',
    'WikidataAliasMatches(results=r, text=t) := Filter(collection=r, where=WikidataExactAlias(t))',
    'WikidataSearchHit(results=r, text=t) := If(condition=EqualTo(Count(WikidataLabelMatches(results=r, text=t)), 0), then=First(WikidataAliasMatches(results=r, text=t)), else=First(WikidataLabelMatches(results=r, text=t)))',
    'WikidataNotDeprecated(s) := NotEqualTo(GetProperty(s, "rank"), "deprecated")',
    'WikidataHasNoEndDate(s) := Not(HasKey(object=GetProperty(s, "qualifiers"), key="P582"))',
    'WikidataPreferred(s) := EqualTo(GetProperty(s, "rank"), "preferred")',
    'WikidataNonDeprecated(statements=s) := Filter(collection=s, where=WikidataNotDeprecated())',
    'WikidataCurrent(statements=s) := Filter(collection=WikidataNonDeprecated(statements=s), where=WikidataHasNoEndDate())',
    'WikidataCandidates(statements=s) := If(condition=EqualTo(Count(WikidataCurrent(statements=s)), 0), then=WikidataNonDeprecated(statements=s), else=WikidataCurrent(statements=s))',
    'WikidataPreferredStatements(statements=s) := Filter(collection=WikidataCandidates(statements=s), where=WikidataPreferred())',
    'WikidataBestStatement(statements=s) := If(condition=EqualTo(Count(WikidataPreferredStatements(statements=s)), 0), then=First(WikidataCandidates(statements=s)), else=First(WikidataPreferredStatements(statements=s)))',
    "Double(x) := Multiply(x, 2)",
    "Triple(x) := Multiply(x, 3)",
    "Half(x) := Divide(x, 2)",
    "Square(x) := Multiply(x, x)",
    "Increment(x) := Add(x, 1)",
    "Decrement(x) := Subtract(x, 1)",
    "OlderThan(left=a, right=b) := GreaterThan(left=Age(subject=a), right=Age(subject=b))",
    "YoungerThan(left=a, right=b) := LessThan(left=Age(subject=a), right=Age(subject=b))",
    "BiggerThan(left=a, right=b) := GreaterThan(left=Size(subject=a), right=Size(subject=b))",
    "AtLeast(left=a, right=b) := Not(LessThan(left=a, right=b))",
    "AtMost(left=a, right=b) := Not(GreaterThan(left=a, right=b))",
    "Location(subject=s, attribute=Timezone()) := Timezone()",
    "Location(subject=s) := Query(pattern=LivesIn(subject=s, object=Where()))",
    "Put(folder=f, x) := Write(path=Join(f, PathOf(x)), contents=GetProperty(x, \"contents\"))",
    "Create(target=t, files=xs) := Map(collection=xs, transformation=Put(folder=PathOf(t)))",
    "Change(target=t, from=a, to=b) := Write(path=PathOf(t), contents=Replace(Read(path=PathOf(t)), a, b))",
    "Edit(target=t, from=a, to=b) := Write(path=PathOf(t), contents=Replace(Read(path=PathOf(t)), a, b))",
    "Contents(of=x) := GetProperty(x, \"contents\")",
    "Understand(path=p) := Hear(Read(path=p), as=Program())",
]

# How the relations themselves behave. Ordinary edges, which is the whole
# point: symmetry and taxonomy are not wired into the traversal, so a relation
# invented this afternoon can be given the same standing just by saying so.
_RELATION_PROPERTIES = [
    ("IsA", HAS_PROPERTY, TAXONOMIC),
    ("IsA", HAS_PROPERTY, TRANSITIVE),
    ("InstanceOf", HAS_PROPERTY, TAXONOMIC),
    ("PartOf", HAS_PROPERTY, TRANSITIVE),
    ("OppositeOf", HAS_PROPERTY, SYMMETRIC),
    ("SimilarTo", HAS_PROPERTY, SYMMETRIC),
]

_CORE_EDGES = [
    ("Greeting", "OppositeOf", "Farewell"),
    ("Big", "OppositeOf", "Small"),
    ("Old", "OppositeOf", "Young"),
    ("Good", "OppositeOf", "Bad"),
    ("Fast", "OppositeOf", "Slow"),
    ("GreaterThan", "OppositeOf", "LessThan"),
    ("Double", "RealizedBy", "Multiply"),
    ("Age", "IsA", "Number"),
    ("Sum", "SimilarTo", "Add"),
]


def seed(knowledge: Knowledge) -> Knowledge:
    """Give a fresh Knowledge store its starting vocabulary."""
    builtin = Evidence(source="builtin")
    for kind, entries in _KINDS.items():
        for entry in entries:
            if isinstance(entry, tuple):
                name, gloss = entry
            else:
                name, gloss = entry, ""
            knowledge.define(name, kind=kind, gloss=gloss)
    for name in NATIVES:
        if not knowledge.knows_concept(name):
            knowledge.define(name, kind="operation")
    for src in _CORE_RULES:
        head, body = parse_definition(src)
        assert isinstance(head, Call)
        knowledge.add_rule(head, body, Evidence(source="builtin"))
        knowledge.concepts[head.concept].learned = False
    for s, r, t in _RELATION_PROPERTIES + _CORE_EDGES:
        knowledge.relate(s, r, t, builtin)
    for name, gloss in _BUILTIN_GLOSSES.items():
        if knowledge.knows_concept(name):
            knowledge.define(name, gloss=gloss)
    knowledge.define("Soup", kind="entity", gloss="me")
    knowledge.assert_fact(
        call("Name", subject=call("Assistant"), value=Lit("Soup")),
        evidence=builtin,
    )
    return knowledge


def fresh_knowledge() -> Knowledge:
    return seed(Knowledge())
