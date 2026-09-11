"""The concepts Soup is born knowing, and the Python realizations for the ones
that bottom out in arithmetic, collections, logic or memory.

Plenty of concepts in here are realized by *rewrite rules* rather than code
(`Double(x) := Multiply(x, 2)`), because a realization is allowed to resolve
to more concepts. That is the whole point.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, call, render, substitute
from .knowledge import (
    HAS_PROPERTY,
    SYMMETRIC,
    TAXONOMIC,
    TRANSITIVE,
    Evidence,
    Knowledge,
)
from .parse import parse, parse_definition
from .realize import Context, NATIVES, native

UNKNOWN_TOLERANT = {
    "Unknown",
    "Ask",
    "If",
    "Answer",
    "Assertion",
    "Explain",
    "Teach",
    "Remember",
    "Query",
    "Not",
    "Say",
    "Request",
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


def _numbers(c: Call) -> Optional[List[float]]:
    out: List[float] = []
    for a in c.args:
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


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


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
    left = _num(c.first("left", "value", "numerator"))
    right = _num(c.get("right") or c.get("by") or _second(c))
    if left is None or right is None:
        return None
    if right == 0:
        return unknown(c)
    return _pack(left / right)


@native("Power", "ToThePowerOf")
def _power(ctx: Context, c: Call) -> Optional[Expr]:
    ns = _numbers(c)
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


@native("Count", "HowMany")
def _count(ctx: Context, c: Call) -> Optional[Expr]:
    items = _items(c.first("collection", "of", "items"))
    if items is None:
        return None
    return Lit(len(items))


@native("Maximum", "Biggest", "Largest", "Max", special=True)
def _maximum(ctx: Context, c: Call) -> Optional[Expr]:
    return _extreme(ctx, c, biggest=True)


@native("Minimum", "Smallest", "Min", special=True)
def _minimum(ctx: Context, c: Call) -> Optional[Expr]:
    return _extreme(ctx, c, biggest=False)


def _extreme(ctx: Context, c: Call, biggest: bool) -> Optional[Expr]:
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
# The only realizations that read anything outside the store. They answer the
# questions people actually open with, and "i don't know what the time is" is
# a silly thing for a program to say.
# ---------------------------------------------------------------------------


@native("Time", "Now")
def _time(ctx: Context, c: Call) -> Optional[Expr]:
    if c.args:
        return None
    ctx.effect("read the clock")
    return Lit(datetime.now().strftime("%I:%M %p").lstrip("0").lower())


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


# ---------------------------------------------------------------------------
# speech acts
# ---------------------------------------------------------------------------


@native("Question", special=True)
def _question(ctx: Context, c: Call) -> Optional[Expr]:
    about = c.first("about", "of")
    if about is None:
        return None
    value = ctx.realize(about)
    if isinstance(value, Call) and value.concept in ("Acknowledged", "Taught", "Explanation"):
        return value
    if isinstance(value, Call) and not value.args and value == about:
        # "who is Greg" did not compute anything, so say what we know of him.
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
    it. Asked about anybody else it is an ordinary question, so return
    nothing and let facts, then the model, have a go.
    """
    subject = c.first("subject", "of")
    if isinstance(subject, Call) and subject.concept in ("Assistant", "User"):
        return c
    return None


@native(
    "Capability",
    "State",
    "Explanation",
    "Taught",
    "Answer",
    "Assertion",
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
    if head is None or body is None:
        rule = c.get("rule")
        if rule is not None:
            ctx.knowledge.assert_fact(rule, evidence=Evidence(source="user"))
            return Call("Acknowledged", (Arg("about", rule),))
        return None
    if not isinstance(head, Call):
        return None
    ctx.knowledge.add_rule(head, body, Evidence(source="teacher"))
    ctx.effect("learned %s" % head.concept)
    return Call("Taught", (Arg("concept", head), Arg("as", body)))


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


@native("Ref", special=True)
def _ref(ctx: Context, c: Call) -> Optional[Expr]:
    return unknown(c)


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
    ],
    "relation": [
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
        "Thanks",
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
        "Identity",
        "Capability",
        "Opinion",
    ],
}

# Realizations that are themselves just meaning, not code.
_CORE_RULES = [
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
    "Location(subject=s) := Query(pattern=LivesIn(subject=s, object=Where()))",
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
    knowledge.define("Soup", kind="entity", gloss="me")
    knowledge.assert_fact(
        call("Name", subject=call("Assistant"), value=Lit("Soup")),
        evidence=builtin,
    )
    return knowledge


def fresh_knowledge() -> Knowledge:
    return seed(Knowledge())
