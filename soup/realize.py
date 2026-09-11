"""Realization: concepts in, concepts out.

A realization is anything that turns one concept expression into another. It
might be Python code, a taught rewrite rule, a lookup in what we believe, or a
composition of all three. Execution is just the case where a realization
happens to touch the outside world.

When nothing can realize a concept we do not throw. We record a `Gap` and keep
going, because the half-resolved expression is exactly the diagnostic surface
the Teacher needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, render, substitute
from .knowledge import Evidence, Knowledge

__all__ = ["Gap", "Result", "Realizer", "Context", "native", "NATIVES", "SPECIAL_FORMS"]

MAX_DEPTH = 48

# Concepts that would rather see an `Unknown` argument than be short-circuited
# by it. Everything else passes the ignorance straight up the tree, so the
# answer names the thing we actually failed to resolve.
UNKNOWN_TOLERANT = {
    "Unknown",
    "Ask",
    "If",
    "Conditional",
    "Answer",
    "Assertion",
    "Explain",
    "Teach",
    "Remember",
    "Recall",
    "Query",
    "Say",
    "Request",
    "Acknowledged",
}


@dataclass
class Gap:
    """A concept we could not resolve, and the shape it was used in."""

    concept: str
    expr: Expr
    reason: str = "no realization"

    @property
    def arity(self) -> int:
        return len(self.expr.args) if isinstance(self.expr, Call) else 0

    def signature(self) -> str:
        if not isinstance(self.expr, Call) or not self.expr.args:
            return "%s()" % self.concept
        parts = []
        for a in self.expr.args:
            parts.append(a.name if a.name else "_")
        return "%s(%s)" % (self.concept, ", ".join(parts))


@dataclass
class Result:
    value: Expr
    gaps: List[Gap] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return not self.gaps

    def __str__(self) -> str:
        return render(self.value, multiline=False)


NativeFn = Callable[["Context", Call], Optional[Expr]]
NATIVES: Dict[str, NativeFn] = {}
SPECIAL_FORMS: Dict[str, NativeFn] = {}


def native(*names: str, special: bool = False) -> Callable[[NativeFn], NativeFn]:
    """Register a Python realization for one or more concepts.

    `special=True` means "do not realize my arguments first" - needed for
    binding forms, conditionals and anything that treats an argument as a
    pattern rather than a value.
    """

    def deco(fn: NativeFn) -> NativeFn:
        table = SPECIAL_FORMS if special else NATIVES
        for n in names:
            table[n] = fn
        return fn

    return deco


class Context:
    """What a native realization gets to play with."""

    def __init__(self, realizer: "Realizer", env: Dict[str, Expr], depth: int, result: Result):
        self.realizer = realizer
        self.env = env
        self.depth = depth
        self.result = result

    @property
    def knowledge(self) -> Knowledge:
        return self.realizer.knowledge

    def realize(self, expr: Expr) -> Expr:
        return self.realizer._realize(expr, self.env, self.depth + 1, self.result)

    def note(self, message: str) -> None:
        self.result.trace.append(message)

    def effect(self, message: str) -> None:
        self.result.effects.append(message)

    def gap(self, concept: str, expr: Expr, reason: str = "no realization") -> None:
        self.realizer._record_gap(self.result, Gap(concept, expr, reason))


class Realizer:
    """Walks a concept expression and resolves as much meaning as it can."""

    def __init__(self, knowledge: Knowledge) -> None:
        self.knowledge = knowledge

    def realize(self, expr: Expr, env: Optional[Dict[str, Expr]] = None) -> Result:
        result = Result(value=expr)
        environment: Dict[str, Expr] = dict(env or {})
        result.value = self._realize(expr, environment, 0, result)
        return result

    # -- core --------------------------------------------------------------
    def _realize(self, expr: Expr, env: Dict[str, Expr], depth: int, result: Result) -> Expr:
        if depth > MAX_DEPTH:
            result.trace.append("stopped: expression kept unfolding past depth %d" % MAX_DEPTH)
            return expr

        if isinstance(expr, Lit):
            return expr

        if isinstance(expr, Var):
            bound = env.get(expr.name)
            if bound is None:
                return expr
            return self._realize(bound, env, depth + 1, result) if bound != expr else expr

        if isinstance(expr, Seq):
            return Seq(tuple(self._realize(i, env, depth + 1, result) for i in expr.items))

        assert isinstance(expr, Call)
        ctx = Context(self, env, depth, result)

        special = SPECIAL_FORMS.get(expr.concept)
        if special is not None:
            out = special(ctx, expr)
            return out if out is not None else expr

        evaluated = Call(
            expr.concept,
            tuple(Arg(a.name, self._realize(a.value, env, depth + 1, result)) for a in expr.args),
        )

        cd = self.knowledge.concept(evaluated.concept)
        known_concept = cd is not None and cd.kind != "concept"
        if known_concept and evaluated.concept not in UNKNOWN_TOLERANT:
            # Pass ignorance upward so the reply names the thing we actually
            # lack. Concepts we do not know keep going: their own gap is the
            # more useful thing to report.
            for a in evaluated.args:
                if isinstance(a.value, Call) and a.value.concept == "Unknown":
                    return a.value

        fn = NATIVES.get(evaluated.concept)
        if fn is not None:
            out = fn(ctx, evaluated)
            if out is not None:
                if out != evaluated:
                    result.trace.append(
                        "%s -> %s" % (render(evaluated, False), render(out, False))
                    )
                    return self._realize(out, env, depth + 1, result)
                return out

        rewritten = self._apply_rules(evaluated, result)
        if rewritten is not None:
            result.trace.append(
                "%s -> %s  (taught)" % (render(evaluated, False), render(rewritten, False))
            )
            return self._realize(rewritten, env, depth + 1, result)

        from_facts = self._from_facts(evaluated)
        if from_facts is not None:
            result.trace.append(
                "%s -> %s  (known)" % (render(evaluated, False), render(from_facts, False))
            )
            return self._realize(from_facts, env, depth + 1, result)

        inherited = self._inherited_rule(evaluated, result)
        if inherited is not None:
            return self._realize(inherited, env, depth + 1, result)

        spread = self._distribute(evaluated, env, depth, result)
        if spread is not None:
            return spread

        # Nothing resolved it, so decide what kind of not-knowing this is.
        if not evaluated.args:
            # A bare name. An entity, a quality, an operation referred to as a
            # value. Names stand for themselves; you cannot fail to evaluate
            # "Greg".
            return evaluated

        if known_concept:
            # We know this concept, we just do not know this particular thing.
            # Ignorance about the world, not about language.
            return Call("Unknown", (Arg("about", evaluated),))

        # Something is being done to arguments and we have no idea what.
        # That is a hole in the language, and the Teacher can fill exactly it.
        self._record_gap(result, Gap(evaluated.concept, evaluated))
        return evaluated

    # -- resolution strategies --------------------------------------------
    def _apply_rules(self, expr: Call, result: Result) -> Optional[Expr]:
        for rule in self.knowledge.rules_for(expr.concept):
            out = rule.apply(expr)
            if out is not None:
                return out
        return None

    def _inherited_rule(self, expr: Call, result: Result) -> Optional[Expr]:
        """If `Sprint IsA Run`, try Run's realizations for a Sprint."""
        for parent in self.knowledge.ancestors(expr.concept):
            for rule in self.knowledge.rules_for(parent):
                out = rule.apply(Call(parent, expr.args))
                if out is not None:
                    result.trace.append(
                        "%s is a %s, so: %s" % (expr.concept, parent, render(out, False))
                    )
                    return out
            fn = NATIVES.get(parent)
            if fn is not None:
                return Call(parent, expr.args)
        return None

    def _distribute(
        self, expr: Call, env: Dict[str, Expr], depth: int, result: Result
    ) -> Optional[Expr]:
        """An operation handed a collection where it wanted one thing.

        "add 10 to [1, 2, 3]" means do it to each of them. Only for operations
        we actually know, and only when exactly one argument is a collection,
        so this never quietly papers over a real mismatch.
        """
        cd = self.knowledge.concept(expr.concept)
        if cd is None or cd.kind != "operation" or len(expr.args) < 2:
            return None
        positions = [i for i, a in enumerate(expr.args) if isinstance(a.value, Seq)]
        if len(positions) != 1:
            return None
        index = positions[0]
        collection = expr.args[index].value
        assert isinstance(collection, Seq)
        spread: List[Expr] = []
        for item in collection.items:
            args = list(expr.args)
            args[index] = Arg(args[index].name, item)
            spread.append(self._realize(Call(expr.concept, tuple(args)), env, depth + 1, result))
        result.trace.append(
            "%s applied to each of %d" % (expr.concept, len(collection.items))
        )
        return Seq(tuple(spread))

    def _from_facts(self, expr: Call) -> Optional[Expr]:
        """Answer straight out of what we believe.

        `Age(subject=Alice)` finds the fact `Age(subject=Alice, value=30)`.
        `Has(subject=Greg, object=Car)` finds itself and answers True.
        """
        if not expr.has("value"):
            probe = Call(expr.concept, expr.args + (Arg("value", Var("_v")),))
            hits = self.knowledge.query(probe)
            for fact, bindings in hits:
                if fact.truth and "_v" in bindings:
                    return bindings["_v"]

        hits = self.knowledge.query(expr)
        for fact, _ in hits:
            if _same_shape(fact.proposition, expr):
                return Lit(bool(fact.truth))
        return None

    def _is_value(self, expr: Call) -> bool:
        """Entities, qualities and other nouns realize to themselves."""
        cd = self.knowledge.concept(expr.concept)
        if cd is None:
            return not expr.args
        return cd.kind in ("entity", "thing", "quality", "value", "modality", "relation")

    def _record_gap(self, result: Result, gap: Gap) -> None:
        for existing in result.gaps:
            if existing.concept == gap.concept and existing.expr == gap.expr:
                return
        result.gaps.append(gap)


def _same_shape(a: Expr, b: Expr) -> bool:
    return isinstance(a, Call) and isinstance(b, Call) and len(a.args) == len(b.args)
