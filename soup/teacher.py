"""The Teacher: what happens when Soup hits a concept it does not have.

The gap is surgical by construction. Ears already said exactly which concept
failed and the shape it was used in, so the question is "teach me `Spooky` as
a visual quality", never "what does this whole sentence mean".
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, call, concept_names, render
from .knowledge import GRAPH_RELATIONS, Evidence, Knowledge
from .realize import Gap

__all__ = ["Teacher", "Lesson"]


class Lesson:
    """A gap we have asked about and are waiting to have filled."""

    def __init__(self, gap: Gap) -> None:
        self.gap = gap
        self.head = _head_for(gap)
        # The Var's name, not the Arg's. A positional argument has no name of
        # its own but its placeholder still does, and `Quadruple(5)` has to be
        # teachable as `Quadruple(x)`.
        self.params = [a.value.name for a in self.head.args if isinstance(a.value, Var)]

    @property
    def concept(self) -> str:
        return self.gap.concept

    def signature(self) -> str:
        return render(self.head, multiline=False)


class Teacher:
    def __init__(self, knowledge: Knowledge) -> None:
        self.knowledge = knowledge

    # -- asking ------------------------------------------------------------
    def ask(self, gap: Gap) -> Lesson:
        return Lesson(gap)

    def question(self, lesson: Lesson) -> str:
        name = lesson.concept
        if lesson.params:
            return (
                "i don't know %s. teach me: %s := ... "
                "(or just say \"%s means <something>\")"
                % (name, lesson.signature(), name.lower())
            )
        return (
            "i don't know what %s means. tell me, like \"%s means <something i know>\", "
            "or give me %s := <expression>" % (name, name.lower(), name)
        )

    # -- receiving ---------------------------------------------------------
    def is_answer(self, lesson: Lesson, meaning: Expr) -> bool:
        """Is this an attempt to define the gap, or some other utterance?

        Grounded arithmetic is still not a definition of Choose. A name that
        merely showed up in a tree (Seq with no gloss, no params, never
        learned) is not a reason to file the turn as a realization.
        """
        if isinstance(meaning, Var):
            return meaning.name in lesson.params
        if isinstance(meaning, Lit):
            return False
        if isinstance(meaning, Seq):
            return any(self._mentions(i, lesson) for i in meaning.items)
        assert isinstance(meaning, Call)
        if meaning.concept in ("Python", "Shell", "Bash"):
            # A method the machine can run. Prefer composing known concepts;
            # this is the bottom when nothing else will do.
            return True
        if meaning.concept in ("Seq", "Sequence"):
            return any(self._mentions(a.value, lesson) for a in meaning.args)
        if meaning.concept in ("Teach", "Learn"):
            head = meaning.first("concept", "of")
            return True if head is None else self._mentions(head, lesson)
        if meaning.concept in ("Question", "Query", "Request"):
            inner = meaning.first("about", "action", "of")
            return False if inner is None else self.is_answer(lesson, inner)
        if meaning.concept in ("Remember", "Assertion"):
            inner = meaning.get("proposition") or meaning.first("about", "of")
            return False if inner is None else self.is_answer(lesson, inner)
        if meaning.concept in (
            "Unintelligible",
            "Unknown",
            "Affirm",
            "Deny",
            "Greeting",
            "Farewell",
            "Laugh",
            "Thanks",
        ):
            return False
        if self._mentions(meaning, lesson):
            return True
        if meaning.concept in _ARITHMETIC and self._only_literals(meaning):
            return False
        return self._uses_params_or_placeholders(meaning, lesson)

    def learn(self, lesson: Lesson, meaning: Expr) -> Optional[Expr]:
        """Turn the user's answer into a realization for the pending concept."""
        if not self.is_answer(lesson, meaning):
            return None
        body = self._extract_body(meaning)
        if body is None:
            return None
        if not self._grounded(body, lesson):
            # Defining a mystery in terms of other mysteries teaches nothing.
            return None
        body = self._parameterise(body, lesson)
        if self._mentions(body, lesson):
            # A definition that uses the word it is defining is not one.
            return None
        if self._junk_body(body, lesson):
            return None
        if isinstance(body, Call) and body.concept in GRAPH_RELATIONS:
            # `RealizedBy`, `HasProperty`, `InverseOf` say how the knowledge
            # graph is wired. They are edges, not realizations, and a rule
            # headed by one can never resolve to anything. Offered the word
            # in a vocabulary list, a model will happily answer "Doing is
            # realized by the subject", which reads like sense and is not.
            return None
        self.knowledge.add_rule(lesson.head, body, Evidence(source="teacher"))
        return call("Taught", concept=lesson.head, **{"as": body})

    def classify(self, name: Call, kind: Expr) -> Optional[Expr]:
        """File what a name IS, as knowledge about it rather than a rewrite.

        Deliberately not `learn`. A definition rewrites a concept away, and
        a name must not be rewritten: chess is *a* game with two players,
        not interchangeable with one, and a `Chess() := Game(players=2)`
        rule would have Kate playing the game of two players. So this lands
        as an IsA edge plus one fact per property, which is what
        inheritance, `Query` and `Recall` already read.

        A kind we have never heard of, carrying nothing, explains nothing:
        `Blorp IsA Flumph` is the classification equivalent of defining a
        mystery with mysteries. Either the kind is one we know or it has to
        tell us something about the thing.
        """
        if not isinstance(kind, Call) or not isinstance(name, Call):
            return None
        if name.concept in concept_names(kind):
            # "Chess is the game of chess." Filing that leaves us knowing
            # exactly as much as before, but believing we made progress,
            # which is worse than knowing nothing.
            return None
        properties = [a for a in kind.args if a.name]
        if not properties and not self.knowledge.knows_concept(kind.concept):
            return None
        evidence = Evidence(source="teacher", confidence=0.7)
        bare = Call(name.concept, ())
        self.knowledge.assert_fact(
            call("IsA", subject=bare, kind=Call(kind.concept, ())), True, evidence
        )
        # The fact is what gets described back; the edge is what inheritance
        # walks. Both, because they answer different questions.
        self.knowledge.relate(name.concept, "IsA", kind.concept, evidence)
        for arg in properties:
            attribute = arg.name[:1].upper() + arg.name[1:]
            self.knowledge.define(attribute, kind="attribute")
            self.knowledge.assert_fact(
                Call(attribute, (Arg("subject", bare), Arg("value", arg.value))),
                True,
                evidence,
            )
        return call("Taught", concept=bare, **{"as": kind})

    def _extract_body(self, meaning: Expr) -> Optional[Expr]:
        """Dig the actual definition out of whatever speech act wrapped it."""
        if isinstance(meaning, Call):
            if meaning.concept in ("Question", "Request"):
                inner = meaning.first("about", "action", "of")
                return self._extract_body(inner) if inner is not None else None
            if meaning.concept == "Teach":
                return meaning.get("meaning")
            if meaning.concept == "Remember":
                proposition = meaning.get("proposition")
                if isinstance(proposition, Call):
                    value = proposition.get("value") or proposition.get("object")
                    if value is not None and proposition.concept in ("Is", "EqualTo", "Value"):
                        return value
                return proposition
            if meaning.concept in ("Unintelligible", "Unknown", "Affirm", "Deny"):
                return None
        return meaning

    def _mentions(self, expr: Expr, lesson: Lesson) -> bool:
        from .expr import concept_names

        return lesson.concept in concept_names(expr)

    def _only_literals(self, expr: Call) -> bool:
        for a in expr.args:
            v = a.value
            if isinstance(v, Lit):
                continue
            if isinstance(v, Call) and v.concept in _ARITHMETIC and self._only_literals(v):
                continue
            return False
        return True

    def _uses_params_or_placeholders(self, expr: Expr, lesson: Lesson) -> bool:
        params = set(lesson.params)
        placeholders = {"Ref", "It", "Something", "Thing", "X"}

        def walk(e: Expr) -> bool:
            if isinstance(e, Var) and e.name in params:
                return True
            if isinstance(e, Call):
                if e.concept in placeholders:
                    return True
                return any(walk(a.value) for a in e.args)
            if isinstance(e, Seq):
                return any(walk(i) for i in e.items)
            return False

        return walk(expr)

    def _junk_body(self, body: Expr, lesson: Lesson) -> bool:
        """A definition that cannot possibly compute the thing it names."""
        if not isinstance(body, Call):
            return False
        if body.concept in ("Concat", "Join") and lesson.concept not in ("Concat", "Join"):
            return True
        if body.concept in ("Equals", "EqualTo"):
            for arg in body.args:
                if isinstance(arg.value, Lit) and isinstance(arg.value.value, bool):
                    return True
        # Unit-conversion tautologies: "Months(x) := Multiply(x, 30)" and
        # friends teach nothing and pollute later questions. The head is a
        # time/unit word, the body is just Multiply(param, magic-number).
        if body.concept in ("Multiply", "Divide", "Times") and len(lesson.params) == 1:
            lits = [a for a in body.args if isinstance(a.value, Lit)]
            vars_ = [a for a in body.args if isinstance(a.value, Var)]
            if len(lits) == 1 and len(vars_) == 1 and len(body.args) == 2:
                val = lits[0].value.value if isinstance(lits[0].value, Lit) else None
                if isinstance(val, (int, float)) and val in _CONVERSION_CONSTANTS:
                    return True
        # A:=B and B:=A create infinite loops. If a rule already exists going
        # the other direction, this is the circular half.
        if not body.args and isinstance(body, Call):
            existing = self.knowledge.rules_for(body.concept)
            for rule in existing:
                if hasattr(rule, 'head') and isinstance(rule.head, Call):
                    if rule.head.concept == body.concept:
                        rhs = getattr(rule, 'body', None)
                        if isinstance(rhs, Call) and rhs.concept == lesson.concept:
                            return True
        return False

    def _grounded(self, body: Expr, lesson: Lesson) -> bool:
        """Is the offered meaning made of things Soup already has?"""
        from .expr import concept_names

        names = [n for n in concept_names(body) if n != lesson.concept]
        if not names:
            return isinstance(body, (Lit, Var, Seq)) or bool(names) is False
        known = [n for n in names if self.knowledge.knows_concept(n)]
        return len(known) * 2 >= len(names)

    def _parameterise(self, body: Expr, lesson: Lesson) -> Expr:
        """Rewrite the taught body so it talks about the head's parameters."""
        replacements: Dict[str, Expr] = {}
        for arg in lesson.gap.expr.args if isinstance(lesson.gap.expr, Call) else ():
            name = arg.name or "x"
            replacements[render(arg.value, multiline=False)] = Var(name)
        placeholders = {"Ref", "It", "Something", "Thing", "X"}
        default = Var(lesson.params[0]) if lesson.params else None

        def swap(e: Expr) -> Expr:
            key = render(e, multiline=False)
            if key in replacements:
                return replacements[key]
            if isinstance(e, Var) and e.name not in lesson.params and default is not None:
                # The ears named the parameter whatever they liked; the head
                # gets to decide what it is actually called.
                return default
            if isinstance(e, Call):
                if e.concept in placeholders and default is not None:
                    return default
                return Call(e.concept, tuple(Arg(a.name, swap(a.value)) for a in e.args))
            if isinstance(e, Seq):
                return Seq(tuple(swap(i) for i in e.items))
            return e

        return swap(body)


def _head_for(gap: Gap) -> Call:
    """`MostAdorable(collection=Pokemon())` becomes `MostAdorable(collection)`."""
    if not isinstance(gap.expr, Call) or not gap.expr.args:
        return Call(gap.concept, ())
    args: List[Arg] = []
    used: set = set()
    positional = 0
    for a in gap.expr.args:
        if a.name:
            name = a.name
        else:
            name = _positional_name(positional)
            positional += 1
        while name in used:
            name += "_"
        used.add(name)
        args.append(Arg(a.name, Var(name)))
    return Call(gap.concept, tuple(args))


def _positional_name(index: int) -> str:
    return ["x", "y", "z", "w"][index] if index < 4 else "a%d" % index


_ARITHMETIC = frozenset(
    {"Add", "Subtract", "Multiply", "Divide", "Power", "Modulo", "Plus", "Minus", "Times"}
)

# Magic numbers that signal a unit-conversion tautology when they show up as
# the sole constant in Multiply(param, N). 30 days, 60 minutes, 365 days,
# 24 hours, 12 months, 86400 seconds. Deliberately excludes small numbers
# (2-10) which are legitimate multiplier definitions (Double, Triple, etc).
_CONVERSION_CONSTANTS = frozenset(
    {12, 24, 30, 60, 365, 1000, 86400, 500, 2.54, 52, 360}
)
