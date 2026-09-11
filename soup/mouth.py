"""Mouth: concept structures in, casual English out.

This is the lossy end of the pipe and it is supposed to be. Soup's actual
answer is a semantic object; mouth is just a renderer with opinions about
sounding like a person.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

from .discourse import Discourse
from .expr import Arg, Call, Expr, Lit, Seq, Var, is_name, quality_items, render
from .knowledge import Knowledge

__all__ = ["Mouth"]


_CONCEPT_WORDS: Dict[str, str] = {
    "Add": "plus",
    "Subtract": "minus",
    "Multiply": "times",
    "Divide": "divided by",
    "Power": "to the power of",
    "Modulo": "mod",
    "GreaterThan": "is greater than",
    "LessThan": "is less than",
    "EqualTo": "is",
    "NotEqualTo": "is not",
    "OlderThan": "is older than",
    "YoungerThan": "is younger than",
    "BiggerThan": "is bigger than",
    "SmallerThan": "is smaller than",
    "TallerThan": "is taller than",
    "Has": "has",
    "Likes": "likes",
    "Loves": "loves",
    "Knows": "knows",
    "LivesIn": "lives in",
    "WorksAt": "works at",
    "IsA": "is a",
    "Is": "is",
    "Age": "age",
    "Name": "name",
    "User": "you",
    "Assistant": "me",
    "We": "we",
}

_SUBJECT_WORDS = {"User": "you", "Assistant": "I", "We": "we"}
_POSSESSIVE = {"User": "your", "Assistant": "my", "We": "our"}

# Modals take a bare verb after them, which is the only reason they need
# their own handling: "you can drive", never "you can drives".
_MODALS = {
    "Can": "can",
    "Could": "could",
    "Must": "must",
    "Should": "should",
    "May": "may",
    "Might": "might",
    "Will": "will",
}


def words_for(concept: str) -> str:
    if concept in _CONCEPT_WORDS:
        return _CONCEPT_WORDS[concept]
    parts = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", concept)
    return " ".join(p.lower() for p in parts) if parts else concept.lower()


class Mouth:
    def __init__(self, knowledge: Knowledge, discourse: Discourse, seed: Optional[int] = None):
        self.knowledge = knowledge
        self.discourse = discourse
        self.random = random.Random(seed)

    # -- top level ---------------------------------------------------------
    def say(self, expr: Expr) -> str:
        text = self._say(expr)
        return _polish(text)

    def _say(self, expr: Expr) -> str:
        if isinstance(expr, Call):
            handler = getattr(self, "_say_" + _snake(expr.concept), None)
            if handler is not None:
                return handler(expr)
        if isinstance(expr, (Lit, Seq)):
            return self.describe(expr)
        return self.describe(expr)

    def _pick(self, *options: str) -> str:
        recent = set(self.discourse.recent_said(4))
        fresh = [o for o in options if o not in recent]
        return self.random.choice(fresh if fresh else list(options))

    # -- speech acts -------------------------------------------------------
    def _say_greeting(self, expr: Call) -> str:
        return self._pick("hey", "hi there", "yo", "hey, what's up", "hello")

    def _say_farewell(self, expr: Call) -> str:
        return self._pick("later", "see ya", "bye", "catch you later")

    def _say_laugh(self, expr: Call) -> str:
        return self._pick("heh", "ha", "right?", "lol", "glad someone's enjoying this")

    def _say_thanks(self, expr: Call) -> str:
        return self._pick("anytime", "no problem", "sure thing", "you got it")

    def _say_affirm(self, expr: Call) -> str:
        return self._pick("cool", "got it", "alright")

    def _say_deny(self, expr: Call) -> str:
        return self._pick("ah, my bad", "okay, noted", "fair enough")

    def _say_answer(self, expr: Call) -> str:
        value = expr.first("value")
        asked = expr.get("to")
        if value is None:
            return "hm, nothing came back"
        if isinstance(value, Call) and getattr(self, "_say_" + _snake(value.concept), None):
            return self._say(value)
        described = self.describe(value)
        if isinstance(value, Lit) and isinstance(value.value, bool):
            return self._truth_reply(value.value, asked)
        if isinstance(value, Seq):
            if value.items and all(
                isinstance(i, Call) and i.concept == "Acknowledged" for i in value.items
            ):
                return self._pick("done", "got it", "okay, wrote that")
            return self._pick("%s" % described, "that gives %s" % described)
        if asked is not None and _is_computation(asked):
            return self._pick("%s" % described, "that's %s" % described, "it's %s" % described)
        return described

    def _say_assertion(self, expr: Call) -> str:
        proposition = expr.get("proposition")
        truth = expr.get("truth")
        if isinstance(truth, Call) and truth.concept == "Unknown":
            return self._say(truth)
        if isinstance(truth, Lit) and isinstance(truth.value, bool):
            return self._truth_reply(truth.value, proposition)
        if proposition is not None:
            return self.clause(proposition)
        return "hm"

    def _say_acknowledged(self, expr: Call) -> str:
        about = expr.get("about")
        if about is None:
            return "got it"
        if isinstance(about, Call) and about.concept in ("Write", "Delete"):
            return self._pick("done", "got it", "okay, wrote that")
        truth = expr.get("truth")
        said = self.clause(about)
        if isinstance(truth, Lit) and truth.value is False:
            said = _negated(said)
        return self._pick(
            "got it",
            "noted",
            "okay, %s" % said,
            "alright, i'll remember that",
        )

    def _say_unknown(self, expr: Call) -> str:
        about = expr.get("about")
        if about is None:
            return "i don't know"
        if isinstance(about, Call) and _is_proposition(about):
            clause = self.clause(about)
            return self._pick(
                "i don't know if %s" % clause,
                "no idea whether %s" % clause,
                "you never told me if %s" % clause,
            )
        if isinstance(about, Call) and about.args:
            thing = self.describe(about)
            return self._pick(
                "i don't know %s" % thing,
                "no idea about %s" % thing,
                "i don't have %s" % thing,
            )
        return "i don't know about %s" % self.describe(about)

    def _say_unintelligible(self, expr: Call) -> str:
        return self._pick(
            "i didn't catch the meaning of that",
            "that one went past me, try rephrasing",
            "hm, i couldn't turn that into anything",
        )

    def _say_taught(self, expr: Call) -> str:
        concept = expr.get("concept")
        name = concept.concept if isinstance(concept, Call) else self.describe(concept)
        label = re.sub(r"^is ", "", words_for(name))
        rule = expr.get("as")
        tail = " as %s" % render(rule, multiline=False) if rule is not None else ""
        return self._pick(
            "got it, %s%s" % (label, tail),
            "okay, i know %s now" % label,
            "learned %s" % label,
        )

    def _say_explanation(self, expr: Call) -> str:
        concept = expr.get("concept")
        name = concept.concept if isinstance(concept, Call) else "that"
        lines: List[str] = []
        gloss = expr.get("gloss")
        if isinstance(gloss, Lit) and gloss.value:
            lines.append(str(gloss.value))
        rules = expr.get("rules")
        if isinstance(rules, Seq) and rules.items:
            for r in rules.items:
                if isinstance(r, Lit):
                    lines.append(str(r.value))
        relations = expr.get("relations")
        if isinstance(relations, Seq) and relations.items:
            lines.append(
                "related: " + ", ".join(self._relation_words(i) for i in relations.items[:6])
            )
        facts = expr.get("facts")
        told = ""
        if isinstance(facts, Seq) and facts.items:
            told = _join([self.clause(i) for i in facts.items[:6]])
        if not lines:
            if told:
                return self._pick(
                    "i know that %s" % told,
                    "here's what i've got: %s" % told,
                    "%s" % told,
                )
            return "i know the word %s but not much else about it" % words_for(name)
        if told:
            lines.append("i know that %s" % told)
        body = " ".join(lines)
        return body if name == "Ref" else "%s: %s" % (words_for(name), body)

    def _relation_words(self, edge: Expr) -> str:
        if isinstance(edge, Call) and edge.args:
            return "%s %s" % (words_for(edge.concept), self.describe(edge.args[0].value))
        return self.describe(edge)

    def _say_identity(self, expr: Call) -> str:
        return (
            "i'm soup. you talk at me, my ears turn it into concepts, "
            "i resolve what i can, and my mouth turns the answer back into words"
        )

    def _say_capability(self, expr: Call) -> str:
        return (
            "arithmetic, lists, remembering facts about people and things, "
            "answering questions about them, and learning new concepts when you teach me"
        )

    def _say_conditional(self, expr: Call) -> str:
        condition = expr.get("condition")
        then = expr.get("then")
        return "if %s then %s" % (
            self.clause(condition) if condition is not None else "?",
            self.clause(then) if then is not None else "?",
        )

    def _say_state(self, expr: Call) -> str:
        return self._pick("running fine", "good, thanks", "can't complain")

    def _truth_reply(self, value: bool, proposition: Optional[Expr]) -> str:
        clause = self.clause(proposition) if proposition is not None else ""
        if value:
            return self._pick("yep", "yeah, %s" % clause, "yes", "correct")
        return self._pick("nope", "no, %s" % _negated(clause), "nah", "no")

    # -- describing values -------------------------------------------------
    def describe(self, expr: Optional[Expr]) -> str:
        if expr is None:
            return "nothing"
        if isinstance(expr, Lit):
            return _lit_words(expr.value)
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Seq):
            return _join([self.describe(i) for i in expr.items])
        assert isinstance(expr, Call)

        if not expr.args:
            if expr.concept in _SUBJECT_WORDS:
                return _SUBJECT_WORDS[expr.concept]
            return self._noun(expr.concept)

        if expr.concept == "Unknown":
            inner = expr.get("about")
            return "what %s is" % self.describe(inner) if inner is not None else "that"

        if expr.concept in ("Add", "Subtract", "Multiply", "Divide", "Power", "Modulo"):
            parts = [self.describe(a.value) for a in expr.args]
            return (" %s " % words_for(expr.concept)).join(parts)

        subject = expr.get("subject")
        if subject is not None and not expr.has("object") and not expr.has("value"):
            owner = self.possessive(subject)
            return "%s %s" % (owner, words_for(expr.concept))

        if expr.concept in ("Maximum", "Minimum", "First", "Last", "Count", "Average"):
            collection = expr.first("collection", "of")
            by = expr.get("by")
            label = {
                "Maximum": "the biggest",
                "Minimum": "the smallest",
                "First": "the first",
                "Last": "the last",
                "Count": "how many",
                "Average": "the average",
            }[expr.concept]
            tail = " by %s" % words_for(by.concept) if isinstance(by, Call) else ""
            return "%s of %s%s" % (label, self.describe(collection), tail)

        if expr.concept == "AllOf":
            kind = expr.get("kind")
            within = expr.get("within")
            base = "the %ss" % words_for(kind.concept) if isinstance(kind, Call) else "them"
            return base + (" in %s" % self.describe(within) if within is not None else "")

        if expr.concept == "Ref":
            inner = expr.first("value", "of")
            return str(inner.value) if isinstance(inner, Lit) else "that"

        if is_name(expr):
            # "the lazy dog", not "dog of quality the lazy".
            adjectives = " ".join(self._adjective(q) for q in quality_items(expr))
            # Something with an adjective in front of it is a common noun,
            # whether or not we happened to see it introduced with an
            # article. "military Time" is nobody's name.
            noun = self._noun(expr.concept).lower()
            if noun.startswith("the "):
                return "the %s %s" % (adjectives, noun[4:])
            return "%s %s" % (adjectives, noun)

        inner = ", ".join(
            ("%s %s" % (a.name, self.describe(a.value))) if a.name else self.describe(a.value)
            for a in expr.args
        )
        return "%s of %s" % (words_for(expr.concept), inner)

    def _adjective(self, expr: Expr) -> str:
        """An adjective as it sits in front of a noun, with no article."""
        if isinstance(expr, Call) and not expr.args:
            return _name_words(expr.concept).lower()
        return self.describe(expr)

    def _noun(self, concept: str) -> str:
        """Common nouns keep the article they were introduced with."""
        words = _name_words(concept)
        if concept in self.knowledge.common_nouns:
            return "the %s" % words.lower()
        return words

    def _bare_verb(self, action: Optional[Expr]) -> str:
        """The verb as it sits after a modal, uninflected."""
        if isinstance(action, Call) and not action.args:
            return words_for(action.concept)
        return self.describe(action)

    def _membership(self, kind: Optional[Expr]) -> str:
        """"is a dog", but "is symmetric": qualities take no article.

        The kind slot always names a category, so it is written bare, without
        whatever article the noun was introduced with.
        """
        if not isinstance(kind, Call) or kind.args:
            return "is a %s" % self.describe(kind)
        words = _name_words(kind.concept).lower()
        cd = self.knowledge.concept(kind.concept)
        if cd is not None and cd.kind in ("quality", "property"):
            return "is %s" % words
        return "is %s %s" % ("an" if words[:1] in tuple("aeiou") else "a", words)

    def possessive(self, subject: Expr) -> str:
        if isinstance(subject, Call) and not subject.args:
            if subject.concept in _POSSESSIVE:
                return _POSSESSIVE[subject.concept]
            return "%s's" % _name_words(subject.concept)
        return "%s's" % self.describe(subject)

    # -- describing propositions ------------------------------------------
    def clause(self, expr: Optional[Expr]) -> str:
        if expr is None:
            return ""
        if isinstance(expr, (Lit, Var, Seq)):
            return self.describe(expr)
        assert isinstance(expr, Call)

        if expr.concept == "Not":
            inner = expr.first("value", "of", "proposition")
            return "it is not the case that %s" % self.clause(inner)

        subject = expr.get("subject")
        obj = expr.get("object")
        value = expr.get("value")

        modal = _MODALS.get(expr.concept)
        if modal is not None and expr.has("action"):
            return "%s %s %s" % (
                self.subject_words(subject),
                modal,
                self._bare_verb(expr.get("action")),
            )

        if subject is not None and obj is not None:
            return "%s %s %s" % (
                self.subject_words(subject),
                _agree(words_for(expr.concept), subject),
                self.describe(obj),
            )
        if subject is not None and value is not None:
            if expr.concept == "Name":
                return "%s name is %s" % (
                    self.possessive(subject),
                    self.describe(value),
                )
            if expr.concept == "Age":
                return "%s %s %s" % (
                    self.subject_words(subject),
                    _agree("is", subject),
                    self.describe(value),
                )
            return "%s %s is %s" % (
                self.possessive(subject),
                words_for(expr.concept),
                self.describe(value),
            )
        if expr.concept == "IsA":
            return "%s %s" % (self.subject_words(subject), self._membership(expr.get("kind")))

        left = expr.get("left")
        right = expr.get("right")
        if left is not None and right is not None:
            return "%s %s %s" % (
                self.describe(left),
                words_for(expr.concept),
                self.describe(right),
            )
        if subject is not None:
            return "%s is %s" % (self.subject_words(subject), words_for(expr.concept))
        return self.describe(expr)

    def subject_words(self, subject: Optional[Expr]) -> str:
        if isinstance(subject, Call) and not subject.args:
            return _SUBJECT_WORDS.get(subject.concept) or self._noun(subject.concept)
        return self.describe(subject)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _name_words(concept: str) -> str:
    if concept in _CONCEPT_WORDS:
        return _CONCEPT_WORDS[concept]
    parts = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", concept)
    if len(parts) == 1:
        return parts[0]
    return " ".join(p.lower() for p in parts)


def _lit_words(value) -> str:
    if value is None:
        return "nothing"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _join(parts: Sequence[str]) -> str:
    parts = list(parts)
    if not parts:
        return "nothing"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return "%s and %s" % (parts[0], parts[1])
    return "%s and %s" % (", ".join(parts[:-1]), parts[-1])


def _agree(verb: str, subject: Expr) -> str:
    """"you has the car" is not a sentence anybody wants to read."""
    if isinstance(subject, Call) and subject.concept in ("User", "We", "Assistant"):
        swap = {"has": "have", "is": "are", "likes": "like", "knows": "know", "lives in": "live in"}
        if subject.concept == "Assistant":
            swap["is"] = "am"
        return swap.get(verb, verb)
    return verb


def _negated(clause: str) -> str:
    """"Bob is older than Alice" -> "Bob is not older than Alice"."""
    for verb in (
        " is ",
        " are ",
        " am ",
        " was ",
        " were ",
        " can ",
        " could ",
        " will ",
        " should ",
        " must ",
    ):
        if verb in clause:
            return clause.replace(verb, verb[:-1] + " not ", 1)
    return "that's not true" if not clause else "%s, no" % clause


def _is_proposition(expr: Call) -> bool:
    """Something that could be true or false, rather than something with a value."""
    if expr.concept in _MODALS and expr.has("action"):
        return True
    return (expr.has("subject") and expr.has("object")) or (
        expr.has("left") and expr.has("right")
    )


def _is_computation(expr: Expr) -> bool:
    return isinstance(expr, Call) and bool(expr.args)


def _polish(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",")
    if not text:
        return "hm"
    return text
