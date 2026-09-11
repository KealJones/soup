"""Short-term conversational memory.

Ears need this to turn "double it" into something with an actual argument, and
mouth needs it so Soup does not repeat itself like a broken toy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .expr import Call, Expr, Lit, Seq, Var

__all__ = ["Turn", "Discourse"]


@dataclass
class Turn:
    heard: str
    meaning: Optional[Expr] = None
    answer: Optional[Expr] = None
    said: str = ""


@dataclass
class Discourse:
    turns: List[Turn] = field(default_factory=list)
    last_value: Optional[Expr] = None
    last_collection: Optional[Expr] = None
    last_entity: Optional[Expr] = None
    last_question: Optional[Expr] = None
    entities: List[str] = field(default_factory=list)
    bindings: Dict[str, Expr] = field(default_factory=dict)
    pending_teach: Optional[str] = None
    pending_param: Optional[str] = None

    # -- updates -----------------------------------------------------------
    def note_entity(self, name: str) -> None:
        if name in self.entities:
            self.entities.remove(name)
        self.entities.insert(0, name)
        self.last_entity = Call(name, ())

    def note_meaning(self, expr: Expr) -> None:
        for node in _entities_in(expr):
            self.note_entity(node)

    def note_answer(self, expr: Optional[Expr]) -> None:
        """Remember what "it" refers to next turn: the payload, not the wrapper."""
        if expr is None:
            return
        if isinstance(expr, Call):
            if expr.concept == "Answer":
                return self.note_answer(expr.get("value"))
            if expr.concept == "Assertion":
                return self.note_answer(expr.get("truth"))
            if expr.concept in ("Unknown", "Acknowledged", "Explanation", "Taught"):
                return
        if isinstance(expr, Seq):
            self.last_collection = expr
        self.last_value = expr

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > 200:
            del self.turns[:-200]

    # -- lookups -----------------------------------------------------------
    def resolve_pronoun(self, word: str) -> Optional[Expr]:
        w = word.lower()
        if self.pending_param and w in ("it", "that", "this", "them", "those", "one"):
            # Mid-lesson, "it" means the thing being defined, not whatever we
            # were talking about a minute ago.
            return Var(self.pending_param)
        if w in ("it", "that", "this", "one"):
            return self.last_value or self.last_entity
        if w in ("them", "those", "these", "they"):
            return self.last_collection or self.last_value
        if w in ("he", "him", "she", "her", "his", "hers"):
            return self.last_entity
        if w in ("i", "me", "my", "mine", "myself"):
            return Call("User", ())
        if w in ("you", "your", "yours", "yourself"):
            return Call("Assistant", ())
        if w in ("we", "us", "our"):
            return Call("We", ())
        return None

    def recent_said(self, n: int = 3) -> List[str]:
        return [t.said for t in self.turns[-n:] if t.said]


def _entities_in(expr: Expr) -> List[str]:
    from .expr import walk

    out: List[str] = []
    for node in walk(expr):
        if isinstance(node, Call) and not node.args and node.concept[:1].isupper():
            out.append(node.concept)
    return out
