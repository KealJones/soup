"""A conversation.

    utterance -> EARS -> concept expression -> realization -> concept structure
              -> MOUTH -> utterance

with a detour to the Teacher whenever a concept fails to resolve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .builtins import fresh_knowledge
from .discourse import Discourse, Turn
from .ears import Ears, Heard
from .expr import Arg, Call, Expr, Lit, Seq, call, render
from .knowledge import Evidence, Knowledge
from .mouth import Mouth
from .parse import ParseError
from .realize import Gap, Realizer, Result
from .teacher import Lesson, Teacher

__all__ = ["Session", "Reply", "DEFAULT_MEMORY"]

DEFAULT_MEMORY = os.path.join("data", "memory.json")

# Utterances that are moves in the conversation rather than things to work out.
_PURE_SPEECH = ("Affirm", "Deny", "Unintelligible")

# Wrappers that say what kind of move an utterance is. They never count as
# evidence that we understood the actual content.
_SPEECH_ACTS = frozenset(
    [
        "Question",
        "Request",
        "Ask",
        "Remember",
        "Recall",
        "Explain",
        "Teach",
        "Learn",
        "Query",
        "Forget",
        "Greeting",
        "Farewell",
        "Thanks",
        "Affirm",
        "Deny",
        "Unintelligible",
        "Ref",
    ]
)


@dataclass
class Reply:
    said: str
    heard: Optional[Heard] = None
    result: Optional[Result] = None
    lesson: Optional[Lesson] = None

    @property
    def meaning(self) -> Optional[Expr]:
        return self.heard.expr if self.heard else None

    @property
    def answer(self) -> Optional[Expr]:
        return self.result.value if self.result else None

    @property
    def trace(self) -> List[str]:
        return list(self.result.trace) if self.result else []


class Session:
    def __init__(
        self,
        knowledge: Optional[Knowledge] = None,
        memory_path: Optional[str] = DEFAULT_MEMORY,
        seed: Optional[int] = None,
        llm: object = None,
        lookup: object = None,
    ) -> None:
        self.knowledge = knowledge or fresh_knowledge()
        self.memory_path = memory_path
        if memory_path:
            self.knowledge.load(memory_path)
        self.discourse = Discourse()
        if llm is not None and hasattr(llm, "learn_vocabulary"):
            # Show the model the vocabulary once. It is what stops it
            # inventing Times alongside the Multiply we already have.
            llm.learn_vocabulary(self.knowledge)
        self.ears = Ears(self.knowledge, self.discourse, seat=llm)
        self.realizer = Realizer(self.knowledge, seat=llm, lookup=lookup)
        self.mouth = Mouth(self.knowledge, self.discourse, seed=seed)
        self.teacher = Teacher(self.knowledge)
        self.lesson: Optional[Lesson] = None
        self.last_result: Optional[Result] = None

    # -- the loop ----------------------------------------------------------
    def respond(self, utterance: str) -> Reply:
        heard = self.ears.listen(utterance)
        expr = heard.expr

        if self.lesson is not None:
            taught = self._try_lesson(expr)
            if taught is not None:
                return taught

        expr = self._pre_resolve(expr)

        if isinstance(expr, Call) and expr.concept in _PURE_SPEECH:
            reply = Reply(self.mouth.say(expr), heard)
            self._record(utterance, heard, None, reply.said)
            return reply

        result = self.realizer.realize(expr)
        self.last_result = result

        if result.gaps:
            gap = self._most_telling(result.gaps)
            self.lesson = self.teacher.ask(gap)
            said = self.teacher.question(self.lesson)
            self.discourse.pending_teach = gap.concept
            self.discourse.pending_param = (
                self.lesson.params[0] if self.lesson.params else None
            )
            reply = Reply(said, heard, result, self.lesson)
            self._record(utterance, heard, result.value, said)
            return reply

        self.discourse.note_answer(result.value)
        said = self.mouth.say(result.value)
        self._record(utterance, heard, result.value, said)
        return Reply(said, heard, result)

    # -- teaching detour ---------------------------------------------------
    def _try_lesson(self, expr: Expr) -> Optional[Reply]:
        lesson = self.lesson
        assert lesson is not None
        if isinstance(expr, Call) and expr.concept in ("Unintelligible", "Deny", "Farewell"):
            self._clear_lesson()
            if expr.concept != "Unintelligible":
                return None
            return Reply("alright, never mind %s then" % lesson.concept)

        taught = self.teacher.learn(lesson, expr)
        if taught is None:
            # Not a usable definition. Drop the lesson rather than nag, and
            # treat what they said as a normal thing to say.
            self._clear_lesson()
            return None
        self._clear_lesson()
        said = self.mouth.say(taught)

        retry = self._replay(lesson)
        if retry is not None:
            said = "%s. %s" % (said, retry)
        self._record("", None, taught, said)
        return Reply(said)

    def _clear_lesson(self) -> None:
        self.lesson = None
        self.discourse.pending_teach = None
        self.discourse.pending_param = None

    def _replay(self, lesson: Lesson) -> Optional[str]:
        """Now that we know the concept, finish the thing we got stuck on."""
        for turn in reversed(self.discourse.turns):
            if turn.meaning is None:
                continue
            if lesson.concept not in _mentions(turn.meaning):
                continue
            result = self.realizer.realize(turn.meaning)
            if result.gaps:
                return None
            self.last_result = result
            self.discourse.note_answer(result.value)
            return "so: %s" % self.mouth.say(result.value)
        return None

    # -- small rewrites before realization --------------------------------
    def _pre_resolve(self, expr: Expr) -> Expr:
        """Handle the couple of speech acts that need the conversation itself."""
        if isinstance(expr, Call) and expr.concept == "Explain":
            target = expr.first("concept", "about")
            if isinstance(target, Call) and target.concept == "Ref":
                return self._explain_last()
        return expr

    def _explain_last(self) -> Expr:
        if self.last_result is None:
            return call("Unknown", about=call("Ref", Lit("that")))
        if not self.last_result.trace:
            return call(
                "Explanation",
                concept=call("Ref", Lit("that")),
                gloss=Lit("nothing to unpack, that one resolved in one step"),
            )
        steps = [Lit(line) for line in self.last_result.trace]
        return call(
            "Explanation",
            concept=call("Ref", Lit("that")),
            gloss=Lit("here is how i got there:"),
            rules=Seq(tuple(steps)),
        )

    def _most_telling(self, gaps: List[Gap]) -> Gap:
        """Prefer the gap with the most structure; it makes the best question."""
        return sorted(gaps, key=lambda g: (-g.arity, g.concept))[0]

    # -- bookkeeping -------------------------------------------------------
    def _record(
        self,
        utterance: str,
        heard: Optional[Heard],
        answer: Optional[Expr],
        said: str,
    ) -> None:
        self.discourse.record(
            Turn(
                heard=utterance,
                meaning=heard.expr if heard else None,
                answer=answer,
                said=said,
            )
        )

    def save(self) -> Optional[str]:
        if not self.memory_path:
            return None
        self.knowledge.save(self.memory_path)
        return self.memory_path


def _mentions(expr: Expr) -> List[str]:
    from .expr import concept_names

    return concept_names(expr)
