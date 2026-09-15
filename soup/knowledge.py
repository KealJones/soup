"""What Soup knows: concepts, relations between them, asserted facts, and the
conceptual rewrite rules it has been taught.

Everything in here is plain serialisable data. Native function realizations
live in `builtins.py` and are registered at runtime, because you cannot write
a closure into a JSON file and mean it. A snippet of Python or shell, as a
rule body (`HomeDir() := Python("os.path.expanduser('~')")`), is ordinary
data and does persist: it is a realization Soup can run today, and a note
that a native might belong there later.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from .expr import Call, Expr, Var, match, render
from .parse import parse, parse_definition

__all__ = [
    "HAS_PROPERTY",
    "INVERSE_OF",
    "SYMMETRIC",
    "TRANSITIVE",
    "TAXONOMIC",
    "PROPERTIES",
    "GRAPH_RELATIONS",
    "Evidence",
    "Edge",
    "Fact",
    "ConceptDef",
    "Rule",
    "Knowledge",
]


# The meta-vocabulary: the few names the traversal itself has to understand.
#
# Relation kinds are deliberately absent. There is no closed set of them,
# because a relation is only another concept; what a relation *does* is
# decided by the properties asserted about it, and anyone can assert one.
# "IsA" is not special to this module, it is merely the first thing anybody
# happened to call taxonomic.
HAS_PROPERTY = "HasProperty"
INVERSE_OF = "InverseOf"

SYMMETRIC = "Symmetric"  # holds just as well the other way round
TRANSITIVE = "Transitive"  # a R b and b R c means a R c
TAXONOMIC = "Taxonomic"  # the target is a more general kind than the source

PROPERTIES: Tuple[str, ...] = (SYMMETRIC, TRANSITIVE, TAXONOMIC)

# Relations describing how the graph itself is wired. Fine as edges, never
# as the body of a rule: "Doing is realized by the subject" is a sentence
# about Soup's bookkeeping wearing the clothes of a definition.
GRAPH_RELATIONS: Tuple[str, ...] = (HAS_PROPERTY, INVERSE_OF, "RealizedBy")


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
    substituted into the body. Most bodies are more concepts. A body of
    `Python(...)` or `Shell(...)` is still a rule; the snippet is the
    realization until a native exists.
    """

    head: Call
    body: Expr
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def concept(self) -> str:
        return self.head.concept

    @property
    def method(self) -> Optional[str]:
        """`python` / `shell` when this realization bottoms out in code."""
        return code_kind(self.body)

    def source_text(self) -> str:
        return "%s := %s" % (
            render(self.head, multiline=False),
            render(self.body, multiline=False),
        )

    def apply(self, target: Expr, loose: bool = True) -> Optional[Expr]:
        """Rewrite `target` if this rule's head matches it.

        `loose` allows binding by position when the argument names disagree,
        which is how `Double(x)` fires on `Double(value=6)`. Turn it off to
        ask the narrower question: is this rule about *this* shape, named the
        way this call names it? That is what tells the electrical Power from
        the exponential one.
        """
        from .expr import substitute

        bindings = match(self.head, target)
        if bindings is None and loose:
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
        if any(a.name for a in self.head.args):
            # A head that named its parameters is a claim about this shape:
            # `Power(voltage, current)` is the electrical reading and has no
            # business firing on `Power(work, time)` just because both happen
            # to have two numbers in them. Positional heads stay forgiving.
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
            if learned:
                existing.learned = True
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
        """Everything reachable from `source` by `relation`.

        How far this walks is not wired in. It reads the relation's own
        properties, and those are ordinary edges, so telling Soup that some
        relation is symmetric changes what this returns from then on.
        """
        inverses = self.inverses_of(relation)
        transitive = TRANSITIVE in self.properties_of(relation)
        out: List[str] = []
        seen = {source}
        frontier = [source]
        while frontier:
            for nxt in self._step(frontier.pop(0), relation, inverses):
                if nxt in seen:
                    continue
                seen.add(nxt)
                out.append(nxt)
                if transitive:
                    frontier.append(nxt)
        return out

    def properties_of(self, relation: str) -> Set[str]:
        """What sort of relation this is: symmetric, transitive, taxonomic.

        Scanned raw rather than through `related`, because deciding whether
        HasProperty is symmetric would otherwise require deciding whether
        HasProperty is symmetric.
        """
        return {
            e.target for e in self.edges if e.source == relation and e.relation == HAS_PROPERTY
        }

    def inverses_of(self, relation: str) -> Set[str]:
        """Relations saying the same thing the other way round.

        A symmetric relation is its own inverse, which is all symmetry means.
        """
        out = {e.target for e in self.edges if e.source == relation and e.relation == INVERSE_OF}
        out.update(
            e.source for e in self.edges if e.target == relation and e.relation == INVERSE_OF
        )
        if SYMMETRIC in self.properties_of(relation):
            out.add(relation)
        return out

    def relations_with(self, prop: str) -> List[str]:
        """Every relation carrying `prop`."""
        return sorted(
            {e.source for e in self.edges if e.relation == HAS_PROPERTY and e.target == prop}
        )

    def _step(self, source: str, relation: str, inverses: Set[str]) -> List[str]:
        out = [e.target for e in self.edges if e.source == source and e.relation == relation]
        for e in self.edges:
            if e.target == source and e.relation in inverses and e.source not in out:
                out.append(e.source)
        return out

    def edges_for(self, name: str) -> List[Edge]:
        return [e for e in self.edges if e.source == name or e.target == name]

    def ancestors(self, name: str, _seen: Optional[set] = None) -> List[str]:
        """The kinds `name` falls under, by whichever relations claim to say so.

        Which relations those are is itself asserted knowledge, so a relation
        taught this afternoon joins the taxonomy as soon as someone says it
        is taxonomic.
        """
        seen = _seen if _seen is not None else set()
        out: List[str] = []
        for relation in self.relations_with(TAXONOMIC):
            for parent in self.related(name, relation):
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
        self._absorb_property(fact)
        return fact

    def _absorb_property(self, fact: Fact) -> None:
        """"PartOf is transitive" is a claim about how to walk the graph.

        The ears have no reason to treat it as anything but another IsA, so
        catch the ones naming a traversal property and mirror them onto an
        edge, which is where `related` will look.
        """
        p = fact.proposition
        if not fact.truth or not isinstance(p, Call) or p.concept != "IsA":
            return
        subject, kind = p.get("subject"), p.get("kind")
        if isinstance(subject, Call) and isinstance(kind, Call) and kind.concept in PROPERTIES:
            self.relate(subject.concept, HAS_PROPERTY, kind.concept, fact.evidence)

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
        bucket = self.rules.setdefault(head.concept, [])
        for existing in bucket:
            if existing.head == head and existing.body == body:
                return existing
        rule = Rule(head, body, evidence or Evidence(source="teacher"))
        bucket[:] = [r for r in bucket if r.head != head]
        bucket.append(rule)
        cd = self.define(
            head.concept,
            params=tuple(a.value.name for a in head.args if isinstance(a.value, Var)),
            learned=True,
        )
        if not cd.gloss and rule.method:
            cd.gloss = render(body, False)
        self.relate(head.concept, "RealizedBy", _body_head(body), rule.evidence)
        return rule

    def rules_for(self, concept: str) -> List[Rule]:
        return self.rules.get(concept, [])

    def drop_learned_rule(self, concept: str, head: Optional[Call] = None) -> None:
        """Unfile a teacher rule that did not help the question that prompted it.

        With a `head`, only that shape goes. A word is allowed several
        meanings, so a bad reading of `Power(subject)` is no reason to forget
        that `Power(voltage, current)` is watts.
        """
        bucket = self.rules.get(concept) or []
        kept = [
            r
            for r in bucket
            if r.evidence.source == "builtin" or (head is not None and r.head != head)
        ]
        if kept:
            self.rules[concept] = kept
        else:
            self.rules.pop(concept, None)

    def learned_rules(self) -> List[Rule]:
        out: List[Rule] = []
        for bucket in self.rules.values():
            out.extend(r for r in bucket if r.evidence.source != "builtin")
        return out

    # -- persistence -------------------------------------------------------
    def to_json(self) -> dict:
        """Only what was learned. The seed lives in code; overheard names die with the process."""
        rules = [
            r for bucket in self.rules.values() for r in bucket
            if r.evidence.source != "builtin"
        ]
        facts = [f for f in self.facts if f.evidence.source != "builtin"]
        edges = [e for e in self.edges if e.evidence.source != "builtin"]
        return {
            "version": 1,
            "concepts": [c.to_json() for c in self.concepts.values() if c.learned],
            "edges": [e.to_json() for e in edges],
            "facts": [f.to_json() for f in facts],
            "rules": [r.to_json() for r in rules],
            "notes": self.notes,
            "common_nouns": sorted(self.common_nouns),
        }

    def load_json(self, data: dict) -> None:
        for c in data.get("concepts", []):
            cd = ConceptDef.from_json(c)
            if not cd.learned:
                continue
            existing = self.concepts.get(cd.name)
            if existing is None:
                self.concepts[cd.name] = cd
                continue
            existing.learned = True
            if cd.params and not existing.params:
                existing.params = cd.params
            if cd.gloss and not existing.gloss:
                existing.gloss = cd.gloss
            if cd.kind != "concept" and existing.kind == "concept":
                existing.kind = cd.kind
        for e in data.get("edges", []):
            edge = Edge.from_json(e)
            if edge.evidence.source == "builtin":
                continue
            self.relate(edge.source, edge.relation, edge.target, edge.evidence)
        for f in data.get("facts", []):
            try:
                fact = Fact.from_json(f)
            except Exception:
                continue
            if fact.evidence.source == "builtin":
                continue
            self.assert_fact(fact.proposition, fact.truth, fact.evidence)
        for r in data.get("rules", []):
            try:
                rule = Rule.from_json(r)
            except Exception:
                continue
            if rule.evidence.source == "builtin":
                continue
            self.add_rule(rule.head, rule.body, rule.evidence)
        for note in data.get("notes", []):
            if note not in self.notes:
                self.notes.append(note)
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


def code_kind(body: Expr) -> Optional[str]:
    if isinstance(body, Call) and body.concept in ("Python", "Shell", "Bash"):
        return "python" if body.concept == "Python" else "shell"
    return None
