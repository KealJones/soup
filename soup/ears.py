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

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .discourse import Discourse
from .expr import Arg, Call, Expr, Lit, Seq, call, walk
from .knowledge import Knowledge
from .parse import ParseError, looks_like_expression, parse, parse_definition

__all__ = ["Ears", "Heard", "split_mcq"]


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
    guessed: Optional[Expr] = None
    note: str = ""

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

        mcq = split_mcq(raw)
        if mcq is not None:
            return self._settle(self._hear_mcq(raw, mcq))

        if self.seat is None:
            return self._deaf(raw, "i have no model to hear you with")
        context = self._discourse_context()
        try:
            guess = self.seat.hear(raw, list(self.knowledge.concepts), context=context)
        except TypeError:
            guess = self.seat.hear(raw, list(self.knowledge.concepts))
        if guess is None:
            heard = self._deaf(raw, self.seat.last_error or "i could not make that out")
            heard.guessed = getattr(self.seat, "last_guess", None)
            return heard
        note = self.seat.last_error or ""
        return self._settle(Heard(guess, raw, "llm", confidence=0.8, note=note))

    def _hear_mcq(self, raw: str, mcq: Tuple[str, List[Tuple[str, str]]]) -> Heard:
        """Hear the stem. Options stay literals. Do not let the dump invent Select."""
        stem_text, opts = mcq
        among = Seq(
            tuple(
                Call(
                    "Option",
                    (Arg("letter", Lit(letter)), Arg("text", Lit(text))),
                )
                for letter, text in opts
            )
        )
        guess = None
        if self.seat is not None:
            guess = self.seat.hear(stem_text, list(self.knowledge.concepts))
            if guess is None:
                guess = self.seat.hear(raw, list(self.knowledge.concepts))
        by, given = _stem_of(guess)
        choose_args = [Arg("among", among), Arg("by", by)]
        if given is not None:
            choose_args.append(Arg("given", given))
        choose = Call("Choose", tuple(choose_args))
        question_args = [Arg("about", choose)]
        if given is not None:
            question_args.append(Arg("given", given))
        source = "llm" if guess is not None else "none"
        return Heard(
            Call("Question", tuple(question_args)),
            raw,
            source,
            confidence=0.8 if guess is not None else 0.4,
            reason="",
        )

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

    def _discourse_context(self) -> str:
        """Recent conversation, so the model can resolve 'the game' -> Chess."""
        if not self.discourse.turns:
            return ""
        lines = []
        for turn in self.discourse.turns[-3:]:
            if turn.heard:
                lines.append("user: %s" % turn.heard[:120])
            if turn.said:
                lines.append("soup: %s" % turn.said[:120])
        if not lines:
            return ""
        return "RECENT CONVERSATION (resolve references like 'the game', 'it', etc):\n" + "\n".join(lines)

    def _settle(self, heard: Heard) -> Heard:
        self.discourse.note_meaning(heard.expr)
        self._remember_names(heard.expr)
        return heard

    def _remember_names(self, expr: Expr) -> None:
        """A name that got through the ears is a name this session has."""
        added = False
        for node in walk(expr):
            if not isinstance(node, Call) or node.args:
                continue
            if self.knowledge.knows_concept(node.concept):
                continue
            self.knowledge.define(node.concept, kind="entity")
            added = True
        if added and self.seat is not None and hasattr(self.seat, "learn_vocabulary"):
            self.seat.learn_vocabulary(self.knowledge)


_MCQ_LINE = re.compile(r"^\s*[\(\[]?([A-Ja-j])[\)\]\.\:]\s+(\S.*)$")


def split_mcq(text: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """Stem plus A./B. options, or nothing.

    The options are the exam. The stem is the question Soup has to work out.
    """
    lines = (text or "").splitlines()
    opts: List[Tuple[str, str]] = []
    stem_lines: List[str] = []
    started = False
    for line in lines:
        found = _MCQ_LINE.match(line)
        if found:
            started = True
            opts.append((found.group(1).upper(), found.group(2).strip()))
            continue
        if not started:
            stem_lines.append(line)
        elif line.strip() and opts:
            letter, prev = opts[-1]
            opts[-1] = (letter, (prev + " " + line.strip()).strip())
    if len(opts) < 2:
        return None
    stem = "\n".join(stem_lines).strip()
    if not stem:
        return None
    return stem, opts


def _stem_of(expr: Optional[Expr]) -> Tuple[Expr, Optional[Expr]]:
    if expr is None:
        return call("Unintelligible"), None
    given = None
    by = expr
    if isinstance(expr, Call) and expr.concept in ("Question", "Query"):
        given = expr.get("given")
        inner = expr.first("about", "of")
        if inner is not None:
            by = inner
    if isinstance(by, Call) and by.concept in ("Choose", "Select", "Pick"):
        inner = by.first("by", "about", "of")
        if inner is not None:
            by = inner
        if given is None:
            given = by.get("given") if isinstance(by, Call) else None
    return by, given
