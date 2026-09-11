"""The Teacher: what happens when Soup hits a concept it does not have.

The gap is surgical by construction. Ears already said exactly which concept
failed and the shape it was used in, so the question is "teach me `Spooky` as
a visual quality", never "what does this whole sentence mean".
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, call, render
from .knowledge import Evidence, Knowledge
from .realize import Gap

__all__ = ["Teacher", "Lesson"]


class Lesson:
    """A gap we have asked about and are waiting to have filled."""

    def __init__(self, gap: Gap) -> None:
        self.gap = gap
        self.head = _head_for(gap)
        self.params = [a.name for a in self.head.args if isinstance(a.value, Var)]

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
    def learn(self, lesson: Lesson, meaning: Expr) -> Optional[Expr]:
        """Turn the user's answer into a realization for the pending concept."""
        body = self._extract_body(meaning)
        if body is None:
            return None
        if not self._grounded(body, lesson):
            # Defining a mystery in terms of other mysteries teaches nothing.
            return None
        body = self._parameterise(body, lesson)
        self.knowledge.add_rule(lesson.head, body, Evidence(source="teacher"))
        return call("Taught", concept=lesson.head, **{"as": body})

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
    for index, a in enumerate(gap.expr.args):
        name = a.name or _positional_name(index)
        while name in used:
            name += "_"
        used.add(name)
        args.append(Arg(a.name, Var(name)))
    return Call(gap.concept, tuple(args))


def _positional_name(index: int) -> str:
    return ["x", "y", "z", "w"][index] if index < 4 else "a%d" % index
