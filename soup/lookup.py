"""Use Wikidata as a fact source and keep what comes back.

A model asked for a fact will produce something fact-shaped. Wikidata asked
for a fact will produce the fact, or nothing, which is the more useful pair of
outcomes. Its requests, exact-match filtering, and claim selection are
realized from the Wikidata concepts seeded by :mod:`soup.builtins`. This
module maps a missing fact to those concepts, converts Wikidata's typed
value, and remembers useful identifiers and glosses.

Using the source also extends the vocabulary. Asking for the height of the
Eiffel Tower means finding out that `Height` is Wikidata's P2048 and that the
Eiffel Tower is Q243, and both of those are worth writing down:

    Height(subject=EiffelTower())
      -> WikidataProperty(name=Height(), value="P2048")     learned
      -> WikidataId(name=EiffelTower(), value="Q243")       learned
      -> Height(subject=EiffelTower(), value="324 metres")  answered

Every search hit and every statement from the claims response is also kept as
a source-tagged `WikidataSearchResult` or `WikidataStatement` fact. The
identifiers come back as ordinary facts, so the second question about that
entity is cheaper, and the concepts get Wikidata's own descriptions as glosses.

The command line can include this as one of the realizer's generic fact
sources. With no network, `Fetch` returns `Unknown`, this source disables
itself, and realization carries on with the next source.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Var, call
from .knowledge import Evidence, Knowledge

ENDPOINT = "https://www.wikidata.org/w/api.php"

# Where a learned identifier is kept. Facts, like everything else, so they
# survive a restart and show up in `:facts` with wikidata as their source.
ID_OF = "WikidataId"
PROPERTY_OF = "WikidataProperty"

# Argument names that hold the thing a question is *about*.
_SUBJECT = ("subject", "of", "for", "about", "object")

# Some questions are about the entity itself rather than a property of it.
_DESCRIBING = frozenset(["Identity", "Description", "Definition", "Meaning"])

_UNITLESS = ("1", "http://www.wikidata.org/entity/Q199")


def words_of(name: str) -> str:
    """`BoilingPoint` is Wikidata's "boiling point"."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)
    return " ".join(p.lower() for p in parts)


class Wikidata:
    """A place to look things up, and a habit of remembering what was found."""

    tag = "wikidata"
    note = "looked it up on Wikidata"

    def __init__(
        self,
        knowledge: Optional[Knowledge] = None,
        seat=None,
    ) -> None:
        self.knowledge = knowledge
        from .realize import Realizer

        # The HTTP requests and their selection logic are ordinary concepts.
        # This small evaluator has no outside sources of its own, so a failed
        # Fetch simply returns Unknown instead of recursing back into us.
        self.realizer = Realizer(knowledge) if knowledge is not None else None
        # Only ever asked what a property is *called*. See `_paraphrased`.
        self.seat = seat
        # Set for the length of one lookup when a list-op was really asking
        # for the minimum / maximum of an attribute. See `_realize_attribute`.
        self.specifically: Optional[str] = None
        self.available = True
        self.last_error: Optional[str] = None
        self.calls = 0

    # -- the one thing the realizer wants ---------------------------------
    def answer(self, expr: Call) -> Optional[Expr]:
        """The value of this expression according to Wikidata, or nothing.

        Nothing is a perfectly good answer here. It means the question was
        well formed and the world does not happen to record it, which is the
        point at which guessing becomes someone else's job.
        """
        if not self.available or self.realizer is None:
            return None
        subject = expr.first(*_SUBJECT)
        text = _subject_text(subject)
        if text is None:
            # We can only look up a plain named thing. "the height of the
            # tallest building in france" is a question, not a name.
            return None

        # The property is settled first because it is the cheaper refusal.
        # Looking up an entity we will have no property for leaves a Q-id
        # lying around for a question we never answered.
        prop = None
        if expr.concept not in _DESCRIBING:
            prop = self._property(expr.concept, text)
            if prop is None:
                return None

        entity = self._entity(subject, text)
        if entity is None:
            return None
        if prop is None:
            gloss = entity.get("description")
            return Lit(gloss) if gloss else None
        return self._claim(entity["id"], prop, subject, expr.concept)

    # -- naming things ----------------------------------------------------
    def _entity(self, subject: Call, text: str) -> Optional[Dict[str, str]]:
        """Find the Q-id for a thing, remembering it and what it means."""
        known = self._recall(ID_OF, subject)
        if known is not None:
            return {"id": known, "description": self._gloss(subject.concept)}

        hit = self._search(text, "item")
        if hit is None:
            return None
        self._keep(ID_OF, subject, hit["id"])
        self._teach(subject.concept, "entity", hit.get("description", ""))
        return {"id": hit["id"], "description": hit.get("description", "")}

    def _property(self, concept: str, subject: Optional[str] = None) -> Optional[str]:
        """Find the P-id for a concept, remembering it and what it means.

        A concept we can already act on is never looked up, because this is
        here to extend the vocabulary rather than quietly reinterpret it.
        Knowing the *word* is not the same as being able to do anything with
        it though: `Height` is defined as an attribute and has no
        realization at all, which is precisely the hole P2048 fills.
        """
        specifically = getattr(self, "specifically", None)
        stored = _qualified(concept, specifically)
        known = self._recall(PROPERTY_OF, stored)
        if known is not None:
            return known
        if self._realizable(concept):
            return None
        hit = None
        if specifically:
            # "minimum number of players" is what Wikidata actually calls
            # it. Searching that first means this question does not need a
            # model at all, once Minimum has been peeled off Players.
            for phrase in (
                "%s number of %s" % (specifically, words_of(concept)),
                "%s %s" % (specifically, words_of(concept)),
            ):
                hit = self._search(phrase, "property")
                if hit is not None:
                    break
        if hit is None:
            hit = self._search(words_of(concept), "property")
        if hit is None:
            hit = self._paraphrased(concept, subject, specifically)
        if hit is None:
            return None
        self._keep(PROPERTY_OF, stored, hit["id"])
        self._teach(concept, "relation", hit.get("description", ""))
        return hit["id"]

    def _paraphrased(
        self, concept: str, subject: Optional[str], specifically: Optional[str] = None
    ) -> Optional[Dict]:
        """Ask a model what this property is *called*, then look that up.

        Wikidata files how many people play chess under "minimum number of
        players". Nobody asks a question in those words, and `Players`
        matches no label at all, so a fact that is sitting right there gets
        missed and the question falls through to somebody guessing.

        This is the smallest job a model can usefully be given. It supplies
        a word, never a value: the suggestion still has to match a real
        property name exactly, and the answer still comes out of the
        record. A wrong guess finds nothing, which is the failure mode you
        want. The P-id it lands on is kept as an ordinary fact, so you can
        see which property answered you and say it was the wrong one.
        """
        if self.seat is None or not getattr(self.seat, "available", True):
            return None
        for name in self.seat.property_names(concept, subject, specifically):
            hit = self._search(name, "property")
            if hit is not None:
                return hit
        return None

    def _realizable(self, concept: str) -> bool:
        """Can soup already do something with this concept on its own?"""
        from .builtins import NATIVES

        if concept in NATIVES:
            return True
        if self.knowledge is None:
            return False
        if self.knowledge.rules_for(concept):
            return True
        defined = self.knowledge.concept(concept)
        return defined is not None and defined.kind == "speech"

    def _search(self, text: str, kind: str) -> Optional[Dict]:
        """Search, and insist on being answered the question that was asked.

        Search is a ranking, not a lookup, so the top hit is routinely
        something else entirely: the best property match for "tower" is
        "Tower Records Online artist ID" and for "is" it is "ISNI". Taking
        either would not be a near miss, it would be a fabricated fact with
        a citation attached. So only an exact name counts, label before
        alias, because "height" is the label of P2048 and merely an alias of
        "elevation above sea level".
        """
        hits = self._realize(
            call("WikidataSearchResults", text=Lit(text), kind=Lit(kind))
        )
        self._remember_search_results(text, kind, hits)
        hit = self._realize(
            call("WikidataSearchHit", results=hits, text=Lit(text))
        )
        if not _is_unknown(hit):
            value = _to_python(hit)
            if isinstance(value, dict) and value.get("id"):
                return value
        self.last_error = "wikidata has no %s called %r" % (kind, text)
        return None

    # -- reading a value --------------------------------------------------
    def _claim(
        self,
        qid: str,
        pid: str,
        subject: Optional[Call] = None,
        attribute: Optional[str] = None,
    ) -> Optional[Expr]:
        statements = self._realize(
            call("WikidataClaims", entity=Lit(qid), property=Lit(pid))
        )
        self._remember_statements(qid, pid, statements, subject, attribute)
        chosen = self._realize(call("WikidataBestStatement", statements=statements))
        if _is_unknown(chosen):
            return None
        snak = _field(chosen, "mainsnak")
        if _text(_field(snak, "snaktype")) != "value":
            # Wikidata can record "known to have no value", which is an
            # answer, but not one we have anywhere to put yet.
            return None
        return self._value(_to_python(_field(snak, "datavalue")) or {})

    def _value(self, datavalue: Dict) -> Optional[Expr]:
        kind = datavalue.get("type")
        value = datavalue.get("value")
        if kind == "string":
            return Lit(value)
        if kind == "monolingualtext":
            return Lit(value.get("text"))
        if kind == "wikibase-entityid":
            label = self._labels([value.get("id")]).get(value.get("id"))
            return Lit(label) if label else None
        if kind == "quantity":
            return self._quantity(value)
        if kind == "time":
            return _time(value)
        if kind == "globecoordinate":
            return Lit("%s, %s" % (value.get("latitude"), value.get("longitude")))
        return None

    def _quantity(self, value: Dict) -> Optional[Expr]:
        amount = _number(value.get("amount"))
        if amount is None:
            return None
        unit = value.get("unit") or "1"
        if unit in _UNITLESS:
            # A count is worth keeping as a number; you can do sums with it.
            return Lit(amount)
        label = self._labels([unit.rsplit("/", 1)[-1]]).get(unit.rsplit("/", 1)[-1])
        if not label:
            return Lit(amount)
        if amount != 1 and not label.endswith("s"):
            label += "s"
        return Lit("%s %s" % (_plain(amount), label))

    def _labels(self, ids: List[str]) -> Dict[str, str]:
        wanted = [i for i in ids if i]
        if not wanted:
            return {}
        data = self._realize(call("WikidataEntities", ids=Lit("|".join(wanted))))
        out = {}
        for qid in wanted:
            body = _field(data, qid)
            label = _field(_field(_field(body, "labels"), "en"), "value")
            if isinstance(label, Lit) and isinstance(label.value, str):
                out[qid] = label.value
                self._remember(
                    Call(
                        "WikidataEntityLabel",
                        (
                            Arg("id", Lit(qid)),
                            Arg("language", Lit("en")),
                            Arg("value", label),
                        ),
                    )
                )
        return out

    def _remember_search_results(self, text: str, kind: str, hits: Expr) -> None:
        from .expr import Seq

        if not isinstance(hits, Seq):
            return
        for hit in hits.items:
            self._remember(
                Call(
                    "WikidataSearchResult",
                    (
                        Arg("query", Lit(text)),
                        Arg("kind", Lit(kind)),
                        Arg("id", _field(hit, "id") or Lit(None)),
                        Arg("label", _field(hit, "label") or Lit(None)),
                        Arg("description", _field(hit, "description") or Lit(None)),
                        Arg("match", _field(hit, "match") or Call("Object")),
                        Arg("data", hit),
                    ),
                )
            )

    def _remember_statements(
        self,
        qid: str,
        pid: str,
        statements: Expr,
        subject: Optional[Call],
        attribute: Optional[str],
    ) -> None:
        from .expr import Seq

        if not isinstance(statements, Seq):
            return
        for statement in statements.items:
            snak = _field(statement, "mainsnak") or Call("Object")
            datavalue = _field(snak, "datavalue") or Call("Object")
            qualifiers = _field(statement, "qualifiers") or Call("Object")
            args = [
                Arg("entityId", Lit(qid)),
                Arg("propertyId", Lit(pid)),
                Arg("rank", _field(statement, "rank") or Lit("normal")),
                Arg("snaktype", _field(snak, "snaktype") or Lit(None)),
                Arg("valueType", _field(datavalue, "type") or Lit(None)),
                Arg("value", _field(datavalue, "value") or Lit(None)),
                Arg("qualifiers", qualifiers),
                Arg("data", statement),
            ]
            if subject is not None:
                args.insert(0, Arg("subject", subject))
            if attribute:
                args.insert(1 if subject is not None else 0, Arg("attribute", call(attribute)))
            self._remember(Call("WikidataStatement", tuple(args)))

    def _remember(self, proposition: Expr) -> None:
        if self.knowledge is not None:
            self.knowledge.assert_fact(
                proposition,
                True,
                Evidence(source="wikidata", note="complete API result"),
            )

    # -- remembering ------------------------------------------------------
    def _recall(self, relation: str, named) -> Optional[str]:
        if self.knowledge is None:
            return None
        key = call(named) if isinstance(named, str) else named
        pattern = Call(relation, (Arg("name", key), Arg("value", Var("id"))))
        for _fact, binding in self.knowledge.query(pattern):
            found = binding.get("id")
            if isinstance(found, Lit) and isinstance(found.value, str):
                return found.value
        return None

    def _keep(self, relation: str, named, identifier: str) -> None:
        if self.knowledge is None:
            return
        key = call(named) if isinstance(named, str) else named
        self.knowledge.assert_fact(
            Call(relation, (Arg("name", key), Arg("value", Lit(identifier)))),
            True,
            Evidence(source="wikidata"),
        )

    def _teach(self, concept: str, kind: str, gloss: str) -> None:
        """Define a concept we only just met, using Wikidata's own words.

        This is the part that stops the same question being a gap twice.
        `BoilingPoint` arrives as an unknown word and leaves as a relation
        that soup can tell you the meaning of.

        `define` merges rather than overwrites, so this fills in a gloss and
        a kind without ever talking over something we already settled. That
        matters because writing the identifier down mentions the concept,
        which is enough to bring it into being with nothing in it.
        """
        if self.knowledge is None:
            return
        self.knowledge.define(concept, kind=kind, gloss=gloss)

    def _gloss(self, concept: str) -> str:
        if self.knowledge is None:
            return ""
        known = self.knowledge.concept(concept)
        return known.gloss if known is not None else ""

    # -- running the Wikidata concepts -----------------------------------
    def _realize(self, expr: Expr) -> Expr:
        if self.realizer is None:
            return call("Unknown", about=expr)
        result = self.realizer.realize(expr)
        self.calls += sum(effect.startswith("fetched ") for effect in result.effects)
        for note in result.trace:
            if "fetch failed:" in note:
                self.available = False
                self.last_error = note
            elif "fetch timed out:" in note:
                self.last_error = note
        return result.value


def _is_unknown(expr: Expr) -> bool:
    from .expr import walk

    return any(
        isinstance(node, Call) and node.concept in ("Unknown", "Nothing")
        for node in walk(expr)
    )


def _field(expr: Optional[Expr], name: str) -> Optional[Expr]:
    return expr.get(name) if isinstance(expr, Call) else None


def _text(expr: Optional[Expr]) -> Optional[str]:
    return expr.value if isinstance(expr, Lit) and isinstance(expr.value, str) else None


def _to_python(expr: Optional[Expr]):
    if isinstance(expr, Lit):
        return expr.value
    if isinstance(expr, Call):
        return {a.name: _to_python(a.value) for a in expr.args if a.name}
    if isinstance(expr, (list, tuple)):
        return [_to_python(item) for item in expr]
    from .expr import Seq

    if isinstance(expr, Seq):
        return [_to_python(item) for item in expr.items]
    return expr


def _qualified(concept: str, specifically: Optional[str]) -> str:
    """The cache key for a property, when min/max was part of the question.

    Looking up Players of Chess for "how many" and for "the maximum" must
    not share a P-id. Wikidata has both P1872 and P1873.
    """
    if not specifically:
        return concept
    return specifically[:1].upper() + specifically[1:] + concept


def _subject_text(subject) -> Optional[str]:
    """The words behind a noun phrase, for something that indexes by words.

    The ears hand back "the eiffel tower" as `Tower(quality=Eiffel())`,
    which is the right shape for reasoning and the wrong shape for a search
    box, so put the adjectives back in front of the noun. Anything more
    structured than that is a question rather than a name, and gets refused.
    """
    if not isinstance(subject, Call):
        return None
    parts = []
    for arg in subject.args:
        inner = arg.value
        if arg.name != "quality" or not isinstance(inner, Call) or inner.args:
            return None
        parts.append(words_of(inner.concept))
    parts.append(words_of(subject.concept))
    return " ".join(parts)


def _number(amount) -> Optional[float]:
    try:
        value = float(str(amount).lstrip("+"))
    except (TypeError, ValueError):
        return None
    return int(value) if value == int(value) else value


def _plain(amount) -> str:
    return str(int(amount)) if float(amount) == int(amount) else str(amount)


_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()


def _time(value: Dict) -> Optional[Expr]:
    """Wikidata dates carry how precisely they are known. Say only that much."""
    stamp = (value.get("time") or "").lstrip("+")
    precision = value.get("precision", 11)
    match = re.match(r"(-?\d+)-(\d\d)-(\d\d)", stamp)
    if not match:
        return None
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    if precision >= 11 and 1 <= month <= 12 and day:
        return Lit("%d %s %s" % (day, _MONTHS[month - 1], year))
    if precision == 10 and 1 <= month <= 12:
        return Lit("%s %s" % (_MONTHS[month - 1], year))
    return Lit(year)
