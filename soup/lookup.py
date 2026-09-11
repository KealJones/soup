"""Answering out of Wikidata, and keeping what comes back.

A model asked for a fact will produce something fact-shaped. Wikidata asked
for a fact will produce the fact, or nothing, which is the more useful pair of
outcomes. So this sits ahead of the seat: anything with an actual answer
should be looked up, and only the rest is worth guessing at.

The lookup is also the one resolution strategy that extends the vocabulary as
a side effect of being used. Asking for the height of the Eiffel Tower means
finding out that `Height` is Wikidata's P2048 and that the Eiffel Tower is
Q243, and both of those are worth writing down:

    Height(subject=EiffelTower())
      -> WikidataProperty(name=Height(), value="P2048")     learned
      -> WikidataId(name=EiffelTower(), value="Q243")       learned
      -> Height(subject=EiffelTower(), value="324 metres")  answered

The identifiers come back as ordinary facts, so the second question about
that entity is cheaper than the first, and the concepts themselves get
defined with Wikidata's own description as their gloss. A concept that was a
gap five seconds ago is now a concept soup knows the meaning of.

Nothing here is imported by the rest of soup unless you ask for it, and
nothing here is required: with no network, `available` goes false and
realization carries on exactly as it did before.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Var, call
from .knowledge import Evidence, Knowledge

ENDPOINT = "https://www.wikidata.org/w/api.php"

# Wikimedia asks for a descriptive agent and is entitled to one.
AGENT = "soup/0.1 (https://github.com/kealjones/soup) python-urllib"

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

    def __init__(
        self,
        knowledge: Optional[Knowledge] = None,
        timeout: float = 6.0,
        endpoint: str = ENDPOINT,
    ) -> None:
        self.knowledge = knowledge
        self.timeout = timeout
        self.endpoint = endpoint
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
        if not self.available:
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
            prop = self._property(expr.concept)
            if prop is None:
                return None

        entity = self._entity(subject, text)
        if entity is None:
            return None
        if prop is None:
            gloss = entity.get("description")
            return Lit(gloss) if gloss else None
        return self._claim(entity["id"], prop)

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

    def _property(self, concept: str) -> Optional[str]:
        """Find the P-id for a concept, remembering it and what it means.

        A concept we can already act on is never looked up, because this is
        here to extend the vocabulary rather than quietly reinterpret it.
        Knowing the *word* is not the same as being able to do anything with
        it though: `Height` is defined as an attribute and has no
        realization at all, which is precisely the hole P2048 fills.
        """
        known = self._recall(PROPERTY_OF, concept)
        if known is not None:
            return known
        if self._realizable(concept):
            return None
        hit = self._search(words_of(concept), "property")
        if hit is None:
            return None
        self._keep(PROPERTY_OF, concept, hit["id"])
        self._teach(concept, "relation", hit.get("description", ""))
        return hit["id"]

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
        data = self._get(
            action="wbsearchentities",
            search=text,
            language="en",
            uselang="en",
            type=kind,
            limit=10,
        )
        hits = (data or {}).get("search") or []
        for wanted in ("label", "alias"):
            for hit in hits:
                match = hit.get("match") or {}
                if match.get("type") != wanted:
                    continue
                if (match.get("text") or "").lower() == text.lower():
                    return hit
        self.last_error = "wikidata has no %s called %r" % (kind, text)
        return None

    # -- reading a value --------------------------------------------------
    def _claim(self, qid: str, pid: str) -> Optional[Expr]:
        data = self._get(action="wbgetclaims", entity=qid, property=pid)
        statements = ((data or {}).get("claims") or {}).get(pid) or []
        chosen = _best(statements)
        if chosen is None:
            return None
        snak = chosen.get("mainsnak") or {}
        if snak.get("snaktype") != "value":
            # Wikidata can record "known to have no value", which is an
            # answer, but not one we have anywhere to put yet.
            return None
        return self._value(snak.get("datavalue") or {})

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
        data = self._get(
            action="wbgetentities",
            ids="|".join(wanted),
            props="labels",
            languages="en",
        )
        out = {}
        for qid, body in ((data or {}).get("entities") or {}).items():
            label = ((body.get("labels") or {}).get("en") or {}).get("value")
            if label:
                out[qid] = label
        return out

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

    # -- the wire ---------------------------------------------------------
    def _get(self, **params) -> Optional[Dict]:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "1")
        url = self.endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url, headers={"User-Agent": AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.calls += 1
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            # Slow once is not the same as gone. Stay available.
            self.last_error = "wikidata timed out"
            return None
        except (urllib.error.URLError, OSError) as problem:
            self.available = False
            self.last_error = "wikidata unreachable: %s" % problem
            return None
        except (ValueError, AttributeError) as problem:
            self.last_error = "wikidata sent something odd: %s" % problem
            return None


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


def _best(statements: List[Dict]) -> Optional[Dict]:
    """The statement that is true now.

    France has ten capitals on record and nine of them stopped being the
    capital some time ago, so rank and end date are not details.
    """
    live = [s for s in statements if s.get("rank") != "deprecated"]
    current = [s for s in live if "P582" not in (s.get("qualifiers") or {})]
    pool = current or live
    preferred = [s for s in pool if s.get("rank") == "preferred"]
    chosen = preferred or pool
    return chosen[0] if chosen else None


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
