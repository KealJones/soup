"""Soup: a small conversational system that thinks in concepts, not tokens.

    messy human language
        -> Hear      (concept expression, python-ish, with real binding)
        -> realize   (concepts in, concepts out: code, taught rules, memory)
        -> Speak     (casual english)

Everything is a concept, including hearing, learning and talking: Hear, Teach
and Speak are ordinary entries in the vocabulary, realized like any other.
Nothing in the middle is natural language. When a concept cannot be resolved,
the half-resolved expression names exactly what is missing, and the Teacher
fills that one hole.
"""

from .builtins import fresh_knowledge, seed
from .discourse import Discourse
from .ears import Ears, Heard
from .expr import Call, Expr, Lit, Seq, Var, call, render
from .knowledge import Evidence, Knowledge, Rule
from .parse import parse, parse_definition
from .seat import Seat, seat_from_env
from .realize import Gap, Realizer, Result
from .session import Reply, Session
from .teacher import Teacher

__version__ = "0.1.0"

__all__ = [
    "Session",
    "Reply",
    "Ears",
    "Heard",
    "Realizer",
    "Result",
    "Gap",
    "Teacher",
    "Knowledge",
    "Rule",
    "Evidence",
    "Discourse",
    "Expr",
    "Call",
    "Lit",
    "Seq",
    "Var",
    "call",
    "render",
    "parse",
    "parse_definition",
    "fresh_knowledge",
    "seed",
    "__version__",
]
