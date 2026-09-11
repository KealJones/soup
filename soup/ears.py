"""Ears: messy human language in, concept expressions out.

The ears never decide *how* Soup accomplishes anything. They only say, as
precisely as they can, what the human meant, using Soup's conceptual language.
Whether any of those concepts can be resolved is somebody else's problem, and
that is deliberate: the unresolved bits are what the Teacher gets to work on.

There used to be two thousand lines of construction grammar here. It is gone,
and it should be. What it actually did was recognise the phrasings somebody had
thought of in advance: `(hi|hello|hey|yo|sup|howdy)` is not understanding that
a greeting has occurred, it is a list, and the list does not contain "heya".
Every sentence it got wrong got fixed by extending a list, which is not
learning either, and the result was a system that looked clever on the examples
it was built from and fell over on the next thing anybody said.

Reading a sentence is the one part of this whole design that is genuinely
solved elsewhere, and better. So a model does it. That is not a shortcut taken
for want of a parser; it is the parser being the wrong tool. Soup's own ideas
live everywhere else: in what the concepts mean, in how they resolve, in what
gets learned from a gap and kept. Those are the parts worth building by hand.

The contract on the model is narrow and strictly enforced in `seat.py`. It
returns one concept expression. It does not answer anything, it does not decide
how anything is done, and a concept it invents is not a failure but the point:
an unknown concept is exactly what the Teacher and the lookup exist for.

No model means no ears. Soup says so plainly rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .discourse import Discourse
from .expr import Call, Expr, Lit, call
from .knowledge import Knowledge
from .parse import ParseError, looks_like_expression, parse, parse_definition

__all__ = ["Ears", "Heard"]


@dataclass
class Heard:
    """What the ears made of an utterance."""

    expr: Expr
    # The words as spoken, and who turned them into an expression.
    raw: str
    source: str = "llm"
    confidence: float = 1.0
    # Why we failed, when we did. Worth saying out loud: "i could not reach
    # the model" and "i read that and it meant nothing" are different
    # problems and the person talking to us can only fix one of them.
    reason: str = ""

    @property
    def understood(self) -> bool:
        return not (isinstance(self.expr, Call) and self.expr.concept == "Unintelligible")


class Ears:
    def __init__(self, knowledge: Knowledge, discourse: Discourse, seat=None) -> None:
        self.knowledge = knowledge
        self.discourse = discourse
        self.seat = seat

    def listen(self, utterance: str) -> Heard:
        raw = utterance.strip()
        if not raw:
            return Heard(call("Unintelligible"), raw)

        typed = self._typed_directly(raw)
        if typed is not None:
            return self._settle(typed)

        if self.seat is None:
            return self._deaf(raw, "i have no model to hear you with")
        guess = self.seat.hear(raw, list(self.knowledge.concepts))
        if guess is None:
            return self._deaf(raw, self.seat.last_error or "i could not make that out")
        return self._settle(Heard(guess, raw, "llm", confidence=0.8))

    def _typed_directly(self, raw: str) -> Optional[Heard]:
        """Somebody wrote Soup at us rather than English.

        Not a grammar. `Double(x) := Multiply(x, 2)` is this system's own
        notation, and reading your own notation is not natural language
        understanding, it is just parsing.
        """
        if ":=" in raw:
            try:
                head, body = parse_definition(raw)
                return Heard(call("Teach", concept=head, meaning=body), raw, "definition")
            except (ParseError, ValueError, IndexError):
                pass
        if looks_like_expression(raw):
            try:
                return Heard(parse(raw), raw, "concept-syntax")
            except (ParseError, ValueError, IndexError):
                pass
        return None

    def _deaf(self, raw: str, reason: str) -> Heard:
        return Heard(
            call("Unintelligible", text=Lit(raw)),
            raw,
            "none",
            confidence=0.0,
            reason=reason,
        )

    def _settle(self, heard: Heard) -> Heard:
        self.discourse.note_meaning(heard.expr)
        return heard
