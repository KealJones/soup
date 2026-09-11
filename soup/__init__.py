"""Soup: a small conversational system that thinks in concepts, not tokens.

    messy human language
        -> EARS      (concept expression, python-ish, with real binding)
        -> REALIZE   (concepts in, concepts out: code, taught rules, memory)
        -> MOUTH     (casual english)

Nothing in the middle is natural language, and nothing in the middle needs a
model. When a concept cannot be resolved, the half-resolved expression names
exactly what is missing, and the Teacher fills that one hole.
"""

from .builtins import fresh_knowledge, seed
from .discourse import Discourse
from .ears import Ears, Heard
from .expr import Call, Expr, Lit, Seq, Var, call, render
from .knowledge import Evidence, Knowledge, Relation, Rule
from .mouth import Mouth
from .parse import parse, parse_definition
from .realize import Gap, Realizer, Result
from .session import Reply, Session
from .teacher import Teacher

__version__ = "0.1.0"

__all__ = [
    "Session",
    "Reply",
    "Ears",
    "Heard",
    "Mouth",
    "Realizer",
    "Result",
    "Gap",
    "Teacher",
    "Knowledge",
    "Relation",
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
