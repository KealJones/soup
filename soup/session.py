"""A conversation.

The host holds knowledge, discourse and a seat. One turn is realizing
`Turn(text)`. Hear, Teacher and Speak all live inside that concept, because
listening, learning and talking are concepts here like everything else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .builtins import fresh_knowledge
from .discourse import Discourse
from .ears import Ears, Heard
from .expr import Expr, Lit, call, render
from .knowledge import Knowledge
from .realize import Realizer, Result
from .teacher import Lesson, Teacher

__all__ = ["Session", "Reply", "DEFAULT_MEMORY"]

DEFAULT_MEMORY = os.path.join("data", "memory.json")


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
        sources=None,
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
        self.teacher = Teacher(self.knowledge)
        if sources is None:
            # A normal session can look up missing facts without each caller
            # wiring a source by hand. Pass [] to opt out (the CLI exposes
            # that as --no-wikidata).
            from .lookup import Wikidata

            sources = [Wikidata(self.knowledge, seat=llm)]
        self.realizer = Realizer(
            self.knowledge,
            seat=llm,
            sources=sources,
            discourse=self.discourse,
        )
        self.last_result: Optional[Result] = None

    @property
    def lesson(self) -> Optional[Lesson]:
        return self.realizer.lesson

    def respond(self, utterance: str) -> Reply:
        result = self.realizer.realize(call("Turn", text=Lit(utterance)))
        if self.realizer.last_result is not None:
            self.last_result = self.realizer.last_result
        else:
            self.last_result = result
        heard = result.heard
        if heard is None:
            heard = Heard(result.value, utterance)
        said = result.said
        if said is None:
            # Saying it is a concept too, so this is the same Speak the turn
            # itself uses rather than a second way of talking.
            spoken = self.realizer.realize(call("Speak", of=result.value)).value
            said = spoken.value if isinstance(spoken, Lit) else render(result.value, False)
        return Reply(said, heard, result, self.realizer.lesson)

    def save(self) -> Optional[str]:
        if not self.memory_path:
            return None
        self.knowledge.save(self.memory_path)
        return self.memory_path
