"""What Soup knows: concepts, relations between them, asserted facts, and the
conceptual rewrite rules it has been taught.

Everything in here is plain serialisable data. Native (Python) realizations
live in `builtins.py` and are registered at runtime, because you cannot write a
closure into a JSON file and mean it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .expr import Call, Expr, Var, match, render
from .parse import parse, parse_definition

__all__ = [
    "Relation",
    "RELATIONS",
    "Evidence",
    "Edge",
    "Fact",
    "ConceptDef",
    "Rule",
    "Knowledge",
]


class Relation:
    """Edge kinds. A small closed set on purpose: fewer choices, fewer mistakes."""

    IsA = "IsA"
    InstanceOf = "InstanceOf"
    HasProperty = "HasProperty"
    PartOf = "PartOf"
    RealizedBy = "RealizedBy"
    OppositeOf = "OppositeOf"
    SimilarTo = "SimilarTo"
    Causes = "Causes"
    Requires = "Requires"


RELATIONS: Tuple[str, ...] = (
    Relation.IsA,
    Relation.InstanceOf,
    Relation.HasProperty,
    Relation.PartOf,
    Relation.RealizedBy,
    Relation.OppositeOf,
    Relation.SimilarTo,
    Relation.Causes,
    Relation.Requires,
)

_INVERSE = {
    Relation.OppositeOf: Relation.OppositeOf,
    Relation.SimilarTo: Relation.SimilarTo,
}


@dataclass
class Evidence:
    """Where a belief came from and how much we trust it."""

    source: str = "builtin"  # builtin | user | teacher | inferred
    confidence: float = 1.0
    at: float = field(default_factory=time.time)
    note: str = ""

    def to_json(self) -> dict:
        d = {"source": self.source, "confidence": self.confidence, "at": self.at}
        if self.note:
            d["note"] = self.note
        return d

    @staticmethod
    def from_json(d: dict) -> "Evidence":
        return Evidence(
            source=d.get("source", "user"),
            confidence=float(d.get("confidence", 1.0)),
            at=float(d.get("at", time.time())),
            note=d.get("note", ""),
        )


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    evidence: Evidence = field(default_factory=Evidence)

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "evidence": self.evidence.to_json(),
        }

    @staticmethod
    def from_json(d: dict) -> "Edge":
        return Edge(
            d["source"],
            d["relation"],
            d["target"],
            Evidence.from_json(d.get("evidence", {})),
        )


@dataclass
class Fact:
    """An asserted proposition, held as a concept expression."""

    proposition: Expr
    truth: bool = True
    evidence: Evidence = field(default_factory=Evidence)

    def to_json(self) -> dict:
        return {
            "proposition": render(self.proposition, multiline=False),
            "truth": self.truth,
            "evidence": self.evidence.to_json(),
        }

    @staticmethod
    def from_json(d: dict) -> "Fact":
        return Fact(
            parse(d["proposition"]),
            bool(d.get("truth", True)),
            Evidence.from_json(d.get("evidence", {})),
        )


@dataclass
class ConceptDef:
    """A concept Soup has heard of. Knowing *of* it is not knowing what it means."""

    name: str
    kind: str = "concept"  # entity | quality | operation | relation | modality
    params: Tuple[str, ...] = ()
    gloss: str = ""
    learned: bool = False

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": list(self.params),
            "gloss": self.gloss,
            "learned": self.learned,
        }

    @staticmethod
    def from_json(d: dict) -> "ConceptDef":
        return ConceptDef(
            d["name"],
            d.get("kind", "concept"),
            tuple(d.get("params", ())),
            d.get("gloss", ""),
            bool(d.get("learned", False)),
        )


@dataclass
class Rule:
    """A conceptual realization: concepts in, concepts out.

        MostAdorable(items) := Maximum(collection=items, by=AdorablenessScore())

    The head's `Var` arguments bind against the incoming expression and get
    substituted into the body. No code runs; meaning is just rewritten.
    """

    head: Call
    body: Expr
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def concept(self) -> str:
        return self.head.concept

    def source_text(self) -> str:
        return "%s := %s" % (
            render(self.head, multiline=False),
            render(self.body, multiline=False),
        )

    def apply(self, target: Expr) -> Optional[Expr]:
        from .expr import substitute

        bindings = match(self.head, target)
        if bindings is None:
            bindings = self._loose_match(target)
        if bindings is None:
            return None
        return substitute(self.body, bindings)

    def _loose_match(self, target: Expr) -> Optional[Dict[str, Expr]]:
        """Bind by position when the argument names disagree.

        `Double(x)` should still fire on `Double(value=6)` and on the
        `Half(subject=42)` the ears produced from "half of that". The shape of
        the meaning is what matters, not the label the ears happened to pick.
        """
        if not isinstance(target, Call) or target.concept != self.head.concept:
            return None
        params = [a.value for a in self.head.args]
        if len(params) != len(target.args) or not all(isinstance(p, Var) for p in params):
            return None
        bindings: Dict[str, Expr] = {}
        for param, arg in zip(params, target.args):
            assert isinstance(param, Var)
            bindings[param.name] = arg.value
        return bindings

    def to_json(self) -> dict:
        return {"rule": self.source_text(), "evidence": self.evidence.to_json()}

    @staticmethod
    def from_json(d: dict) -> "Rule":
        head, body = parse_definition(d["rule"])
        assert isinstance(head, Call)
        return Rule(head, body, Evidence.from_json(d.get("evidence", {})))


class Knowledge:
    """The store. Concepts, edges, facts, rules, plus a little bit of memory."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptDef] = {}
        self.edges: List[Edge] = []
        self.facts: List[Fact] = []
        self.rules: Dict[str, List[Rule]] = {}
        self.notes: List[str] = []
        # Surface trivia, not belief: which words get "the" in front of them.
        self.common_nouns: set = set()

    def note_noun(self, name: str, common: bool) -> None:
        if common:
            self.common_nouns.add(name)

    # -- concepts ----------------------------------------------------------
    def define(
        self,
        name: str,
        kind: str = "concept",
        params: Iterable[str] = (),
        gloss: str = "",
        learned: bool = False,
    ) -> ConceptDef:
        existing = self.concepts.get(name)
        if existing:
            if gloss and not existing.gloss:
                existing.gloss = gloss
            if params and not existing.params:
                existing.params = tuple(params)
            if kind != "concept" and existing.kind == "concept":
                existing.kind = kind
            return existing
        cd = ConceptDef(name, kind, tuple(params), gloss, learned)
        self.concepts[name] = cd
        return cd

    def knows_concept(self, name: str) -> bool:
        return name in self.concepts

    def concept(self, name: str) -> Optional[ConceptDef]:
        return self.concepts.get(name)

    # -- edges -------------------------------------------------------------
    def relate(
        self,
        source: str,
        relation: str,
        target: str,
        evidence: Optional[Evidence] = None,
    ) -> Edge:
        self.define(source)
        self.define(target)
        for e in self.edges:
            if e.source == source and e.relation == relation and e.target == target:
                return e
        edge = Edge(source, relation, target, evidence or Evidence())
        self.edges.append(edge)
        return edge

    def related(self, source: str, relation: str) -> List[str]:
        out = [e.target for e in self.edges if e.source == source and e.relation == relation]
        inverse = _INVERSE.get(relation)
        if inverse:
            out.extend(
                e.source for e in self.edges if e.target == source and e.relation == inverse
            )
        return out

    def edges_for(self, name: str) -> List[Edge]:
        return [e for e in self.edges if e.source == name or e.target == name]

    def ancestors(self, name: str, _seen: Optional[set] = None) -> List[str]:
        seen = _seen if _seen is not None else set()
        out: List[str] = []
        for parent in self.related(name, Relation.IsA) + self.related(name, Relation.InstanceOf):
            if parent in seen:
                continue
            seen.add(parent)
            out.append(parent)
            out.extend(self.ancestors(parent, seen))
        return out

    def is_a(self, name: str, kind: str) -> bool:
        return name == kind or kind in self.ancestors(name)

    # -- facts -------------------------------------------------------------
    def assert_fact(
        self,
        proposition: Expr,
        truth: bool = True,
        evidence: Optional[Evidence] = None,
    ) -> Fact:
        for existing in self.facts:
            if existing.proposition == proposition:
                existing.truth = truth
                if evidence:
                    existing.evidence = evidence
                return existing
        fact = Fact(proposition, truth, evidence or Evidence(source="user"))
        self.facts.append(fact)
        for name in _mentioned(proposition):
            self.define(name)
        return fact

    def query(self, pattern: Expr) -> List[Tuple[Fact, Dict[str, Expr]]]:
        """Every fact matching `pattern`, with the bindings that made it match."""
        out: List[Tuple[Fact, Dict[str, Expr]]] = []
        for fact in self.facts:
            bindings = match(pattern, fact.proposition)
            if bindings is not None:
                out.append((fact, bindings))
        return out

    def facts_mentioning(self, name: str) -> List[Fact]:
        return [f for f in self.facts if name in _mentioned(f.proposition)]

    # -- rules -------------------------------------------------------------
    def add_rule(
        self,
        head: Call,
        body: Expr,
        evidence: Optional[Evidence] = None,
    ) -> Rule:
        rule = Rule(head, body, evidence or Evidence(source="teacher"))
        bucket = self.rules.setdefault(head.concept, [])
        bucket[:] = [r for r in bucket if r.head != head]
        bucket.append(rule)
        self.define(
            head.concept,
            params=tuple(a.value.name for a in head.args if isinstance(a.value, Var)),
            learned=True,
        )
        self.relate(head.concept, Relation.RealizedBy, _body_head(body))
        return rule

    def rules_for(self, concept: str) -> List[Rule]:
        return self.rules.get(concept, [])

    def learned_rules(self) -> List[Rule]:
        out: List[Rule] = []
        for bucket in self.rules.values():
            out.extend(r for r in bucket if r.evidence.source != "builtin")
        return out

    # -- persistence -------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": 1,
            "concepts": [c.to_json() for c in self.concepts.values()],
            "edges": [e.to_json() for e in self.edges],
            "facts": [f.to_json() for f in self.facts],
            "rules": [r.to_json() for bucket in self.rules.values() for r in bucket],
            "notes": self.notes,
            "common_nouns": sorted(self.common_nouns),
        }

    def load_json(self, data: dict) -> None:
        for c in data.get("concepts", []):
            cd = ConceptDef.from_json(c)
            self.concepts[cd.name] = cd
        for e in data.get("edges", []):
            self.edges.append(Edge.from_json(e))
        for f in data.get("facts", []):
            try:
                self.facts.append(Fact.from_json(f))
            except Exception:
                continue
        for r in data.get("rules", []):
            try:
                rule = Rule.from_json(r)
            except Exception:
                continue
            self.rules.setdefault(rule.concept, []).append(rule)
        self.notes.extend(data.get("notes", []))
        self.common_nouns.update(data.get("common_nouns", []))

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.load_json(json.load(fh))
            return True
        except (ValueError, OSError):
            return False

    # -- stats -------------------------------------------------------------
    def summary(self) -> str:
        learned = sum(1 for c in self.concepts.values() if c.learned)
        return "%d concepts (%d learned), %d relations, %d facts, %d rules" % (
            len(self.concepts),
            learned,
            len(self.edges),
            len(self.facts),
            sum(len(b) for b in self.rules.values()),
        )


def _mentioned(expr: Expr) -> List[str]:
    from .expr import concept_names

    return concept_names(expr)


def _body_head(body: Expr) -> str:
    if isinstance(body, Call):
        return body.concept
    return "Value"
