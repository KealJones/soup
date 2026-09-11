"""Ears: messy human language in, concept expressions out.

The ears never decide *how* Soup accomplishes anything. They only say, as
precisely as they can, what the human meant, using Soup's conceptual language.
Whether any of those concepts can be resolved is somebody else's problem, and
that is deliberate: the unresolved bits are what the Teacher gets to work on.

The grammar here is a small construction grammar. Each construction is a token
pattern with typed slots; slots are parsed by recursive generators so the
matcher can backtrack. Nothing statistical, nothing remote, no API key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .discourse import Discourse
from .expr import Arg, Call, Expr, Lit, Seq, Var, call, render
from .knowledge import Knowledge
from .parse import ParseError, looks_like_expression, parse, parse_definition

__all__ = ["Ears", "Heard", "tokenize"]


# ---------------------------------------------------------------------------
# tokenizing
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "whats": "what is",
    "what's": "what is",
    "who's": "who is",
    "whos": "who is",
    "where's": "where is",
    "when's": "when is",
    "hows": "how is",
    "how's": "how is",
    "that's": "that is",
    "thats": "that is",
    "it's": "it is",
    "its": "it is",
    "there's": "there is",
    "here's": "here is",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "i'd": "i would",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "we're": "we are",
    "they're": "they are",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "can't": "can not",
    "cant": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "wont": "will not",
    "let's": "let us",
    "lets": "let us",
    "gonna": "going to",
    "wanna": "want to",
    "gimme": "give me",
    "u": "you",
    "ur": "your",
    "r": "are",
    "n": "and",
    "&": "and",
    "yr": "your",
    "pls": "please",
    "plz": "please",
    "tho": "though",
    "ya": "you",
    "yea": "yes",
    "yep": "yes",
    "yup": "yes",
    "nope": "no",
    "nah": "no",
}

_FILLER = {
    "uh",
    "uhh",
    "um",
    "umm",
    "er",
    "erm",
    "lol",
    "lmao",
    "haha",
    "hehe",
    "hmm",
    "hm",
    "please",
    "kindly",
    "basically",
    "actually",
    "honestly",
    "literally",
    "seriously",
}

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
}

_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "some", "any", "each"}
_PRONOUNS = {
    "it",
    "that",
    "this",
    "them",
    "those",
    "these",
    "they",
    "he",
    "him",
    "she",
    "her",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "us",
    "our",
    "one",
}
_STOP_FOR_NOUN = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "and",
    "or",
    "than",
    "then",
    "if",
    "to",
    "of",
    "in",
    "on",
    "by",
    "with",
    "for",
    "from",
    "not",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "what",
    "who",
    "how",
    "why",
    "when",
    "where",
    "plus",
    "minus",
    "times",
    "over",
    "means",
    "mean",
    "as",
    "but",
    "so",
    "yes",
    "no",
    "has",
    "have",
    "had",
    "owns",
    "likes",
    "loves",
    "knows",
    "hates",
    "wants",
    "means",
    "every",
    "each",
    "all",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "some",
    "any",
}

# Words whose concept name is not just the capitalised word.
_WORD_CONCEPT = {
    "age": "Age",
    "old": "Age",
    "name": "Name",
    "called": "Name",
    "size": "Size",
    "big": "Size",
    "height": "Height",
    "tall": "Height",
    "weight": "Weight",
    "heavy": "Weight",
    "wide": "Width",
    "long": "Length",
    "price": "Price",
    "cost": "Price",
    "color": "Color",
    "colour": "Color",
    "job": "Job",
    "work": "Job",
    "sum": "Add",
    "total": "Add",
    "product": "Multiply",
    "difference": "Subtract",
    "average": "Average",
    "mean": "Average",
    "number": "Number",
    "numbers": "Number",
    "list": "List",
    "half": "Half",
    "double": "Double",
    "twice": "Double",
    "triple": "Triple",
    "square": "Square",
    "count": "Count",
    "length": "Count",
    "has": "Has",
    "have": "Has",
    "had": "Has",
    "owns": "Has",
    "own": "Has",
}

# Verbs handled by their own branch in `action`. Falling through to the
# generic "verb plus object" reading would let "remember" swallow a pronoun
# and file whatever we were last talking about as a fact.
_OWN_BRANCH = ("remember", "note", "forget", "add", "sum", "subtract", "divide", "multiply")

# Adjectives that name a measurable attribute: "alice is 170 tall".
_MEASURE_WORDS = ("tall", "old", "heavy", "big", "wide", "long")
_UNIT_WORDS = ("cm", "centimeters", "inches", "feet", "pounds", "kg", "kilos", "meters")

_COMPARATIVES = {
    "older": "OlderThan",
    "younger": "YoungerThan",
    "bigger": "BiggerThan",
    "larger": "BiggerThan",
    "greater": "GreaterThan",
    "more": "GreaterThan",
    "less": "LessThan",
    "fewer": "LessThan",
    "smaller": "SmallerThan",
    "taller": "TallerThan",
    "shorter": "ShorterThan",
    "heavier": "HeavierThan",
    "lighter": "LighterThan",
    "faster": "FasterThan",
    "slower": "SlowerThan",
    "better": "BetterThan",
    "worse": "WorseThan",
    "cheaper": "CheaperThan",
    "richer": "RicherThan",
}

_SUPERLATIVES = {
    "biggest": ("Maximum", "Size"),
    "largest": ("Maximum", "Size"),
    "smallest": ("Minimum", "Size"),
    "oldest": ("Maximum", "Age"),
    "youngest": ("Minimum", "Age"),
    "tallest": ("Maximum", "Height"),
    "shortest": ("Minimum", "Height"),
    "heaviest": ("Maximum", "Weight"),
    "fastest": ("Maximum", "Speed"),
    "cheapest": ("Minimum", "Price"),
    "priciest": ("Maximum", "Price"),
    "best": ("Maximum", "Goodness"),
    "worst": ("Minimum", "Goodness"),
    "longest": ("Maximum", "Length"),
    "most": ("Maximum", None),
    "least": ("Minimum", None),
    "first": ("First", None),
    "last": ("Last", None),
}

_TRANSITIVE = (
    "has",
    "have",
    "had",
    "owns",
    "own",
    "likes",
    "like",
    "loves",
    "love",
    "knows",
    "know",
    "hates",
    "hate",
    "wants",
    "want",
    "drives",
    "bought",
    "gave",
)

_IRREGULAR_STEM = {
    "has": "Has",
    "had": "Has",
    "have": "Has",
    "gave": "Gave",
    "gives": "Gave",
    "give": "Gave",
    "went": "Went",
    "goes": "Go",
    "made": "Made",
    "makes": "Make",
    "took": "Took",
    "knows": "Knows",
    "know": "Knows",
    "likes": "Likes",
    "like": "Likes",
    "loves": "Loves",
    "love": "Loves",
    "hates": "Hates",
    "wants": "Wants",
    "lives": "LivesIn",
    "works": "WorksAt",
    "owns": "Has",
    "bought": "Bought",
    "said": "Said",
    "is": "Is",
    "are": "Is",
    "was": "Is",
}


@dataclass
class Tok:
    raw: str
    word: str


def tokenize(source: str) -> Tuple[List[Tok], bool]:
    """Normalise an utterance into tokens. Returns (tokens, was_a_question)."""
    text = source.strip()
    question = "?" in text
    text = re.sub(r"([\[\]()*/+])", r" \1 ", text)
    # Split on punctuation, but leave the dot in 3.14 alone.
    rough = [t for t in re.split(r"[\s,;!?]+|(?<!\d)\.|\.(?!\d)", text) if t]
    toks: List[Tok] = []
    for piece in rough:
        low = piece.lower().strip('"')
        expanded = _CONTRACTIONS.get(low)
        if expanded:
            for w in expanded.split():
                toks.append(Tok(piece, w))
            continue
        if low.endswith("'s") or low.endswith("s'"):
            stem = low[:-2] if low.endswith("'s") else low[:-1]
            toks.append(Tok(piece[: len(stem)], stem))
            toks.append(Tok("'s", "'s"))
            continue
        cleaned = low.strip("'\"")
        if cleaned:
            toks.append(Tok(piece.strip("'\""), cleaned))
    kept = [t for t in toks if t.word not in _FILLER and _is_wordlike(t.word)]
    return (kept if kept else toks), question


def _is_wordlike(word: str) -> bool:
    """Emoji and stray punctuation are not tokens we can do anything with."""
    return bool(word) and (
        any(c.isalnum() for c in word) or word in ("[", "]", "(", ")", "+", "*", "/", "-")
    )


# ---------------------------------------------------------------------------
# construction patterns
# ---------------------------------------------------------------------------


@dataclass
class _Words:
    options: Tuple[Tuple[str, ...], ...]


@dataclass
class _Slot:
    name: str
    kind: str


@dataclass
class _Optional:
    inner: List[Any]


def _compile(pattern: str) -> List[Any]:
    elements: List[Any] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c.isspace():
            i += 1
            continue
        if c == "{":
            j = pattern.index("}", i)
            body = pattern[i + 1 : j]
            name, _, kind = body.partition(":")
            elements.append(_Slot(name, kind or "value"))
            i = j + 1
            continue
        if c == "[":
            j = _closing(pattern, i, "[", "]")
            elements.append(_Optional(_compile(pattern[i + 1 : j])))
            i = j + 1
            continue
        if c == "(":
            j = _closing(pattern, i, "(", ")")
            options = tuple(
                tuple(alt.split()) for alt in pattern[i + 1 : j].split("|") if alt.strip()
            )
            elements.append(_Words(options))
            i = j + 1
            continue
        j = i
        while j < n and not pattern[j].isspace() and pattern[j] not in "{[(":
            j += 1
        elements.append(_Words(((pattern[i:j],),)))
        i = j
    return elements


def _closing(s: str, start: int, open_c: str, close_c: str) -> int:
    depth = 0
    for k in range(start, len(s)):
        if s[k] == open_c:
            depth += 1
        elif s[k] == close_c:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced %s in %r" % (open_c, s))


@dataclass
class Construction:
    """A token pattern plus how to build meaning out of what it captured.

    `bonus` only breaks ties. What really decides between two constructions is
    how many literal words each one had to recognise: a pattern that pinned
    down five actual words understood more than one that shrugged everything
    into a slot.
    """

    pattern: str
    build: Callable[[Dict[str, Expr], "Ears"], Optional[Expr]]
    needs_question: Optional[bool] = None
    bonus: int = 0
    elements: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.elements = _compile(self.pattern)


@dataclass
class Heard:
    """What the ears made of an utterance."""

    expr: Expr
    source: str
    construction: str = ""
    confidence: float = 1.0

    @property
    def understood(self) -> bool:
        return not (isinstance(self.expr, Call) and self.expr.concept == "Unintelligible")


# ---------------------------------------------------------------------------
# the ears themselves
# ---------------------------------------------------------------------------


class Ears:
    def __init__(self, knowledge: Knowledge, discourse: Discourse) -> None:
        self.knowledge = knowledge
        self.discourse = discourse
        self.toks: List[Tok] = []
        self.question = False
        self.constructions = _build_constructions()
        self._observed: Dict[str, str] = {}
        self._defining = False

    # -- entry point -------------------------------------------------------
    def listen(self, utterance: str) -> Heard:
        raw = utterance.strip()
        if not raw:
            return Heard(call("Unintelligible"), raw)

        if ":=" in raw:
            try:
                head, body = parse_definition(raw)
                return Heard(
                    call("Teach", concept=head, meaning=body), raw, "definition"
                )
            except (ParseError, ValueError, IndexError):
                pass

        if looks_like_expression(raw):
            try:
                return Heard(parse(raw), raw, "concept-syntax")
            except (ParseError, ValueError, IndexError):
                pass

        self.toks, self.question = tokenize(raw)
        self._observed = {}
        if not self.toks:
            return Heard(call("Unintelligible"), raw)

        best: Optional[Tuple[int, Expr, str]] = None
        for c in self.constructions:
            if c.needs_question is True and not (self.question or self._question_word()):
                continue
            if c.needs_question is False and self.question and self._question_word():
                continue
            for bindings, end, literals in self._match(c.elements, 0, {}):
                if end != len(self.toks):
                    continue
                built = c.build(bindings, self)
                if built is None:
                    continue
                score = literals * 10 + c.bonus
                if best is None or score > best[0]:
                    best = (score, built, c.pattern)
                break

        if best is not None:
            self._commit_observations(best[1])
            self.discourse.note_meaning(best[1])
            return Heard(best[1], raw, best[2])

        fallback = self._desperate_parse()
        if fallback is not None:
            self._commit_observations(fallback)
            self.discourse.note_meaning(fallback)
            return Heard(fallback, raw, "fallback", confidence=0.5)

        return Heard(call("Unintelligible", text=Lit(raw)), raw, "", confidence=0.0)

    # -- pattern matching --------------------------------------------------
    def _match(
        self, elements: Sequence[Any], i: int, bindings: Dict[str, Expr]
    ) -> Iterator[Tuple[Dict[str, Expr], int, int]]:
        """Yield (bindings, end index, literal words consumed)."""
        if not elements:
            yield bindings, i, 0
            return
        head, rest = elements[0], elements[1:]

        if isinstance(head, _Words):
            for option in head.options:
                if self._words_at(option, i):
                    for b, j, lits in self._match(rest, i + len(option), bindings):
                        yield b, j, lits + len(option)
            return

        if isinstance(head, _Optional):
            for b, j, lits in self._match(head.inner, i, bindings):
                for b2, k, lits2 in self._match(rest, j, b):
                    yield b2, k, lits + lits2
            for b, j, lits in self._match(rest, i, bindings):
                yield b, j, lits
            return

        assert isinstance(head, _Slot)
        for value, j in self._parse_slot(head.kind, i):
            nb = dict(bindings)
            nb[head.name] = value
            for b2, k, lits in self._match(rest, j, nb):
                yield b2, k, lits

    def _words_at(self, words: Sequence[str], i: int) -> bool:
        if i + len(words) > len(self.toks):
            return False
        for offset, w in enumerate(words):
            if self.toks[i + offset].word != w:
                return False
        return True

    def _parse_slot(self, kind: str, i: int) -> Iterator[Tuple[Expr, int]]:
        if kind == "value":
            return self.value(i)
        if kind == "np":
            return self.noun_phrase(i)
        if kind == "number":
            return self.number(i)
        if kind == "word":
            return self.single_word(i)
        if kind == "concept":
            return self.concept_word(i)
        if kind == "text":
            return self.rest_text(i)
        if kind == "prop":
            return self.proposition(i)
        if kind == "comp":
            return self.complement(Var(_SUBJ), i)
        if kind == "defn":
            return self.definition_body(i)
        if kind == "action":
            return self.action(i)
        if kind == "collection":
            return self.collection(i)
        raise ValueError("unknown slot kind %r" % kind)

    # -- token helpers -----------------------------------------------------
    def word(self, i: int) -> Optional[str]:
        return self.toks[i].word if 0 <= i < len(self.toks) else None

    def raw(self, i: int) -> Optional[str]:
        return self.toks[i].raw if 0 <= i < len(self.toks) else None

    def _observe(self, name: str, common: bool) -> None:
        """Note how a noun was used. Only committed if this parse wins.

        This is surface trivia for the mouth ("the car" keeps its article,
        "Greg" does not). It deliberately does not teach Soup the concept, so
        an unrecognised word is still a gap the Teacher can ask about.
        """
        self._observed.setdefault(name, "thing" if common else "entity")

    def _commit_observations(self, expr: Expr) -> None:
        from .expr import concept_names

        for name in concept_names(expr):
            kind = self._observed.get(name)
            if kind:
                self.knowledge.note_noun(name, kind == "thing")

    def _question_word(self) -> bool:
        first = self.word(0)
        return first in (
            "what",
            "who",
            "how",
            "why",
            "when",
            "where",
            "which",
            "is",
            "are",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "should",
            "am",
        )

    # -- slot parsers ------------------------------------------------------
    def single_word(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w:
            yield Lit(self.raw(i)), i + 1

    def concept_word(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w and w not in _STOP_FOR_NOUN:
            yield call(concept_name(w)), i + 1

    def rest_text(self, i: int) -> Iterator[Tuple[Expr, int]]:
        if i < len(self.toks):
            yield Lit(" ".join(t.raw for t in self.toks[i:])), len(self.toks)

    def number(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w is None:
            return
        if _NUMERIC.match(w):
            yield Lit(float(w) if "." in w else int(w)), i + 1
            return
        if w in _NUMBER_WORDS:
            total = _NUMBER_WORDS[w]
            j = i + 1
            # "twenty five", "three hundred"
            while True:
                nxt = self.word(j)
                if nxt in ("hundred", "thousand", "million"):
                    total *= _NUMBER_WORDS[nxt]
                    j += 1
                    continue
                if nxt in _NUMBER_WORDS and _NUMBER_WORDS[nxt] < 100 and total % 10 == 0:
                    total += _NUMBER_WORDS[nxt]
                    j += 1
                    continue
                break
            yield Lit(total), j

    def bracketed(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """`[1, 2, 3]`, written out literally."""
        if self.word(i) != "[":
            return
        items: List[Expr] = []
        j = i + 1
        while j < len(self.toks) and self.word(j) != "]":
            got = next(self.value(j), None)
            if got is None:
                return
            items.append(got[0])
            j = got[1]
            if self.word(j) == "and":
                j += 1
        if self.word(j) == "]":
            yield Seq(tuple(items)), j + 1

    def collection(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """A bracketed list, a run of values, or a reference to one."""
        found = False
        for items, j in self.bracketed(i):
            found = True
            yield items, j
        if found:
            return

        run: List[Expr] = []
        j = i
        while True:
            got = next(self.atom(j), None)
            if got is None:
                break
            run.append(got[0])
            j = got[1]
            if self.word(j) == "and" and next(self.atom(j + 1), None):
                j += 1
                continue
            if j < len(self.toks) and self.word(j) in ("and",):
                break
            nxt = next(self.atom(j), None)
            if nxt is None:
                break
        if len(run) >= 2:
            yield Seq(tuple(run)), j

        for expr, j2 in self.noun_phrase(i):
            yield expr, j2

    # -- value grammar -----------------------------------------------------
    def value(self, i: int) -> Iterator[Tuple[Expr, int]]:
        return self.additive(i)

    def definition_body(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """A value parsed as a definition, where "it" means the parameter.

        "double means multiply it by 2" is a rule about any number, not about
        whatever we happened to be discussing.
        """
        was = self._defining
        self._defining = True
        try:
            for item in list(self.value(i)):
                yield item
            for item in list(self.action(i)):
                yield item
        finally:
            self._defining = was

    def additive(self, i: int) -> Iterator[Tuple[Expr, int]]:
        for left, j in self.multiplicative(i):
            chain: List[Tuple[Expr, int]] = [(left, j)]
            cur, k = left, j
            while True:
                op = self.word(k)
                if op in ("plus", "+"):
                    concept = "Add"
                elif op in ("minus", "-"):
                    concept = "Subtract"
                else:
                    break
                nxt = next(self.multiplicative(k + 1), None)
                if nxt is None:
                    break
                cur = call(concept, cur, nxt[0])
                k = nxt[1]
                chain.append((cur, k))
            for item in reversed(chain):
                yield item

    def multiplicative(self, i: int) -> Iterator[Tuple[Expr, int]]:
        for left, j in self.unary(i):
            chain: List[Tuple[Expr, int]] = [(left, j)]
            cur, k = left, j
            while True:
                op = self.word(k)
                step = 1
                if op in ("times", "*", "x"):
                    concept = "Multiply"
                elif op == "multiplied" and self.word(k + 1) == "by":
                    concept, step = "Multiply", 2
                elif op == "divided" and self.word(k + 1) == "by":
                    concept, step = "Divide", 2
                elif op in ("over", "/"):
                    concept = "Divide"
                elif op in ("mod", "modulo"):
                    concept = "Modulo"
                elif op in ("to",) and self.word(k + 1) == "the" and self.word(k + 2) == "power":
                    concept, step = "Power", 4 if self.word(k + 3) == "of" else 3
                else:
                    break
                nxt = next(self.unary(k + step), None)
                if nxt is None:
                    break
                cur = call(concept, cur, nxt[0])
                k = nxt[1]
                chain.append((cur, k))
            for item in reversed(chain):
                yield item

    def unary(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        prefix = {
            "double": "Double",
            "twice": "Double",
            "triple": "Triple",
            "half": "Half",
            "square": "Square",
            "negative": "Negate",
        }.get(w or "")
        if prefix:
            skip = 1
            if self.word(i + 1) == "of":
                skip = 2
            for inner, j in self.unary(i + skip):
                yield call(prefix, inner), j
        for item in self.atom(i):
            yield item

    def atom(self, i: int) -> Iterator[Tuple[Expr, int]]:
        if self.word(i) == "(":
            for inner, j in self.additive(i + 1):
                if self.word(j) == ")":
                    yield inner, j + 1
        if self.word(i) == "[":
            for items, j in self.bracketed(i):
                yield items, j
        for item in self.number(i):
            yield item
        for item in self.noun_phrase(i):
            yield item

    # -- noun phrases ------------------------------------------------------
    def noun_phrase(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w is None:
            return

        # "this list", "those numbers", plain pronouns
        if w in _PRONOUNS:
            if self._defining and w in ("it", "that", "this", "them", "those", "one"):
                yield Var(_PARAM), i + 1
            nxt = self.word(i + 1)
            if w in ("this", "that", "these", "those", "the") and nxt in (
                "list",
                "numbers",
                "collection",
                "set",
            ):
                # "this list" with no list in the conversation is a dangling
                # reference, not an excuse to grab the last number we said.
                ref = self.discourse.last_collection
                yield (ref if ref is not None else call("Ref", Lit("%s %s" % (w, nxt)))), i + 2
            if w in ("my", "your", "his", "her", "our", "their"):
                owner = self.discourse.resolve_pronoun(w) or call("User")
                for attr, j in self.attribute_word(i + 1):
                    yield call(attr, subject=owner), j
            resolved = self.discourse.resolve_pronoun(w)
            if resolved is not None:
                yield resolved, i + 1
            else:
                yield call("Ref", Lit(w)), i + 1

        # "<name> 's <attr>"
        if self.word(i + 1) == "'s":
            for owner, j in self.bare_noun(i):
                for attr, k in self.attribute_word(j + 1):
                    yield call(attr, subject=owner), k
                for inner, k in self.bare_noun(j + 1):
                    yield call("Of", owner=owner, thing=inner), k

        # "the biggest file in that folder", "the oldest person"
        j = i + 1 if w in _DETERMINERS else i
        sup = self.word(j)

        # "the most adorable pokemon" is one concept, `MostAdorable`, applied
        # to a collection. Soup may not have it yet; that is the point.
        if sup in ("most", "least") and self.word(j + 1) not in (None, "of", "in"):
            quality = self.word(j + 1)
            if quality not in _STOP_FOR_NOUN and not _NUMERIC.match(quality):
                composed = concept_name(sup) + concept_name(quality)
                for inner, k in self.bare_noun(j + 2):
                    yield call(composed, collection=_pluralise_concept(inner)), k
                if self.word(j + 2) in (None, "?"):
                    yield call(composed, collection=call("Ref", Lit("them"))), j + 2

        if sup in _SUPERLATIVES:
            op, by = _SUPERLATIVES[sup]
            if self.word(j + 1) in ("of", "in"):
                for items, k in self.collection(j + 2):
                    yield call(op, collection=items), k
            for inner, k in self.bare_noun(j + 1):
                collection: Expr = _pluralise_concept(inner)
                k2 = k
                if self.word(k) in ("in", "of", "from"):
                    for scope, k3 in self.noun_phrase(k + 1):
                        scoped = _scope_collection(collection, scope)
                        yield _extreme_call(op, scoped, by), k3
                        k2 = k3
                        break
                yield _extreme_call(op, collection, by), k2
            if self.word(j + 1) in ("one", "thing", "item"):
                yield _extreme_call(op, call("Ref", Lit("them")), by), j + 2

        # "the age of alice", "the average of [1,2,3]", "the sum of those"
        if w in _DETERMINERS or w in _WORD_CONCEPT:
            k = i + 1 if w in _DETERMINERS else i
            for attr, m in self.attribute_word(k):
                if self.word(m) != "of":
                    continue
                for items, m2 in self.collection(m + 1):
                    if isinstance(items, Seq):
                        yield call(attr, collection=items), m2
                for owner, m2 in self.noun_phrase(m + 1):
                    yield call(attr, subject=owner), m2
                for v, m2 in self.value(m + 1):
                    yield call(attr, v), m2

        for item in self.bare_noun(i):
            yield item

    def bare_noun(self, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w is None:
            return
        start = i
        if w in _DETERMINERS:
            start = i + 1
            w = self.word(start)
            if w is None:
                return
        if w in _STOP_FOR_NOUN or w in _SUPERLATIVES:
            return
        if w in _PRONOUNS and start == i:
            return
        if _NUMERIC.match(w) or w in _NUMBER_WORDS:
            return
        name = concept_name(w)
        self._observe(name, common=start != i)
        yield call(name), start + 1

    def attribute_word(self, i: int) -> Iterator[Tuple[str, int]]:
        w = self.word(i)
        if w is None:
            return
        if w in _WORD_CONCEPT:
            yield _WORD_CONCEPT[w], i + 1
            return
        if w in _STOP_FOR_NOUN:
            return
        yield concept_name(w), i + 1

    # -- clauses -----------------------------------------------------------
    def proposition(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """A statement of fact: `alice is 30`, `greg has the car`."""
        for subject, j in self.noun_phrase(i):
            for prop, k in self.predicate(subject, j):
                yield prop, k

    def predicate(self, subject: Expr, j: int) -> Iterator[Tuple[Expr, int]]:
        """Everything after the subject, attached to it."""
        w = self.word(j)
        if w is None:
            return

        if w in ("is", "are", "was", "were", "am", "be"):
            negated = self.word(j + 1) == "not"
            k = j + 2 if negated else j + 1
            for prop, end in self.complement(subject, k):
                yield (call("Not", prop) if negated else prop), end
            return

        if w in _TRANSITIVE:
            concept = _IRREGULAR_STEM.get(w) or concept_name(_lemma(w))
            for obj, k in self.noun_phrase(j + 1):
                yield call(concept, subject=subject, object=obj), k
            return

        if w in ("lives", "live") and self.word(j + 1) == "in":
            for obj, k in self.noun_phrase(j + 2):
                yield call("LivesIn", subject=subject, object=obj), k
            return

        if w in ("works", "work") and self.word(j + 1) in ("at", "for"):
            for obj, k in self.noun_phrase(j + 2):
                yield call("WorksAt", subject=subject, object=obj), k
            return

        if _could_be_verb(w):
            # generic transitive verb: "greg drives the truck"
            concept = _IRREGULAR_STEM.get(w) or concept_name(_lemma(w))
            for obj, k in self.noun_phrase(j + 1):
                yield call(concept, subject=subject, object=obj), k

    def complement(self, subject: Expr, i: int) -> Iterator[Tuple[Expr, int]]:
        w = self.word(i)
        if w is None:
            return

        # "30 years old", "170 tall", "12 pounds heavy"
        for n, j in self.number(i):
            after = self.word(j)
            if after in ("years", "year") and self.word(j + 1) in ("old", None):
                end = j + 2 if self.word(j + 1) == "old" else j + 1
                yield call("Age", subject=subject, value=n), end
            elif after in _MEASURE_WORDS:
                unit = self.word(j + 1)
                end = j + 2 if unit in _MEASURE_WORDS else j + 1
                yield call(_WORD_CONCEPT[after], subject=subject, value=n), end
            elif after in _UNIT_WORDS and self.word(j + 1) in _MEASURE_WORDS:
                measure = self.word(j + 1)
                yield call(_WORD_CONCEPT[measure], subject=subject, value=n), j + 2
            elif _is_attribute_call(subject):
                yield _attach_value(subject, n), j
            else:
                yield call("Value", subject=subject, value=n), j

        # "older than bob"
        if w in _COMPARATIVES and self.word(i + 1) == "than":
            for right, j in self.value(i + 2):
                yield call(_COMPARATIVES[w], left=subject, right=right), j

        # "a teacher", "an animal"
        if w in ("a", "an"):
            for kind, j in self.bare_noun(i + 1):
                yield call("IsA", subject=subject, kind=kind), j

        # "named keal", "called keal"
        if w in ("named", "called"):
            nxt = self.raw(i + 1)
            if nxt:
                yield call("Name", subject=subject, value=Lit(nxt)), i + 2

        # "in paris", "from iowa"
        if w in ("in", "at", "from"):
            for place, j in self.noun_phrase(i + 1):
                yield call("LivesIn" if w == "in" else "From", subject=subject, object=place), j

        # bare complement: "alice is tall", "my name is keal"
        for other, j in self.value(i):
            if _is_attribute_call(subject):
                yield _attach_value(subject, other), j
            elif isinstance(other, Call) and not other.args:
                yield call("Is", subject=subject, value=other), j
            else:
                yield call("EqualTo", left=subject, right=other), j
        if w and w not in _STOP_FOR_NOUN and not any(True for _ in self.value(i)):
            yield call(concept_name(w), subject=subject), i + 1

    def action(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """An imperative clause: `double it`, `add 3 and 4`, `sort them`."""
        w = self.word(i)
        if w is None:
            return

        if w in ("add", "sum") :
            for a, j in self.value(i + 1):
                if self.word(j) in ("and", "to", "plus", "+"):
                    for b, k in self.value(j + 1):
                        yield call("Add", a, b), k
                yield call("Add", a), j

        if w in ("multiply", "times"):
            for a, j in self.value(i + 1):
                if self.word(j) in ("by", "and", "times"):
                    for b, k in self.value(j + 1):
                        yield call("Multiply", a, b), k

        if w in ("subtract", "minus"):
            for a, j in self.value(i + 1):
                if self.word(j) == "from":
                    for b, k in self.value(j + 1):
                        yield call("Subtract", left=b, right=a), k

        if w in ("divide",):
            for a, j in self.value(i + 1):
                if self.word(j) == "by":
                    for b, k in self.value(j + 1):
                        yield call("Divide", left=a, right=b), k

        if w in ("remember", "note") :
            k = i + 2 if self.word(i + 1) == "that" else i + 1
            for prop, j in self.proposition(k):
                yield call("Remember", proposition=prop), j

        if w in ("forget",):
            for np, j in self.noun_phrase(i + 1):
                yield call("Forget", about=np), j

        # "double every number in this list"
        if w in ("double", "triple", "halve", "square", "negate"):
            concept = {"halve": "Half"}.get(w, concept_name(w))
            for target, j in self.each_phrase(i + 1):
                yield call("Map", collection=target, transformation=call(concept)), j
            for target, j in self.value(i + 1):
                yield call(concept, target), j

        # generic imperative verb with an object
        if _could_be_verb(w) and w not in _OWN_BRANCH:
            concept = _WORD_CONCEPT.get(w) or concept_name(_lemma(w))
            for target, j in self.each_phrase(i + 1):
                yield call("Map", collection=target, transformation=call(concept)), j
            for target, j in self.collection(i + 1):
                yield call(concept, target), j
            for target, j in self.value(i + 1):
                yield call(concept, target), j
            yield call(concept), i + 1

    def each_phrase(self, i: int) -> Iterator[Tuple[Expr, int]]:
        """`every number in this list`, `all of them`, `each item`."""
        w = self.word(i)
        if w not in ("every", "each", "all"):
            return
        j = i + 1
        if self.word(j) == "of":
            j += 1
        noun_end = j
        if self.word(j) is not None and self.word(j) not in _PRONOUNS:
            for _noun, k in self.bare_noun(j):
                noun_end = k
                break
        if self.word(noun_end) in ("in", "of", "from"):
            for scope, k in self.collection(noun_end + 1):
                yield scope, k
            return
        for scope, k in self.noun_phrase(noun_end if noun_end > j else j):
            yield scope, k
            return
        ref = self.discourse.last_collection
        if ref is not None:
            yield ref, noun_end

    # -- last resort -------------------------------------------------------
    def _desperate_parse(self) -> Optional[Expr]:
        """No construction fit. Salvage whatever meaning is lying around."""
        for expr, end in self.value(0):
            if end == len(self.toks):
                return call("Question", about=expr) if self.question else expr
        for expr, end in self.proposition(0):
            if end == len(self.toks):
                return call("Remember", proposition=expr)
        for expr, end in self.action(0):
            if end == len(self.toks):
                return call("Request", action=expr)
        return None


_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# word -> concept
# ---------------------------------------------------------------------------


def _could_be_verb(word: Optional[str]) -> bool:
    """Brackets, operators and numbers are not verbs, however hopeful we are."""
    return bool(
        word
        and word[0].isalpha()
        and word not in _STOP_FOR_NOUN
        and not _NUMERIC.match(word)
    )


def concept_name(word: str) -> str:
    w = word.strip().strip("'")
    if not w:
        return "Thing"
    if w.lower() in _WORD_CONCEPT:
        return _WORD_CONCEPT[w.lower()]
    if w[:1].isupper() and w[1:2].islower():
        return w
    return "".join(part.capitalize() for part in re.split(r"[-_ ]", w) if part)


def _lemma(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR_STEM:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and w[-3:-2] in ("s", "x", "z", "h"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _singular(name: str) -> str:
    lowered = name.lower()
    stem = _lemma(lowered)
    return "".join(p.capitalize() for p in stem.split())


def _pluralise_concept(noun: Expr) -> Expr:
    """`file` in "the biggest file in that folder" means the set of files."""
    if isinstance(noun, Call) and not noun.args:
        return call("AllOf", kind=call(_singular(noun.concept)))
    return noun


def _scope_collection(collection: Expr, scope: Expr) -> Expr:
    if isinstance(collection, Call) and collection.concept == "AllOf":
        return call("AllOf", kind=collection.first("kind"), within=scope)
    return collection


def _extreme_call(op: str, collection: Expr, by: Optional[str]) -> Expr:
    if by:
        return call(op, collection=collection, by=call(by))
    return call(op, collection=collection)


def _is_attribute_call(expr: Expr) -> bool:
    return isinstance(expr, Call) and expr.has("subject") and not expr.has("value")


def _attach_value(attr: Expr, value: Expr) -> Expr:
    assert isinstance(attr, Call)
    return Call(attr.concept, attr.args + (Arg("value", value),))


# ---------------------------------------------------------------------------
# the construction inventory
# ---------------------------------------------------------------------------


def _build_constructions() -> List[Construction]:
    C = Construction
    out: List[Construction] = []

    def add(pattern: str, build, question=None, bonus: int = 0):
        out.append(C(pattern, build, question, bonus))

    # -- social ------------------------------------------------------------
    add("(hi|hello|hey|yo|sup|howdy|greetings)", lambda b, e: call("Greeting"))
    add("(good) (morning|evening|afternoon)", lambda b, e: call("Greeting"))
    add("(hi|hello|hey|yo) (there|soup|you)", lambda b, e: call("Greeting"))
    add(
        "(bye|goodbye|later|cya|peace) [now]",
        lambda b, e: call("Farewell"),
    )
    add("(see) (you|ya) [later]", lambda b, e: call("Farewell"))
    add("(thanks|thank|thx|ty) [you]", lambda b, e: call("Thanks"))
    add("(how) (are) (you|ya) [doing]", lambda b, e: call("Question", about=call("State", subject=call("Assistant"))))
    add("(what) (is) (up)", lambda b, e: call("Greeting"))
    add("(yes|yeah|correct|right|sure|ok|okay)", lambda b, e: call("Affirm"))
    add("(no|nope|wrong|incorrect)", lambda b, e: call("Deny"))

    # -- identity / meta ---------------------------------------------------
    add(
        "(who|what) (are) (you)",
        lambda b, e: call("Question", about=call("Identity", subject=call("Assistant"))),
    )
    add(
        "(what) (can) (you) (do)",
        lambda b, e: call("Question", about=call("Capability", subject=call("Assistant"))),
    )
    add(
        "(what) (do) (you) (know) (about) {x:np}",
        lambda b, e: call("Recall", about=b["x"]),
    )
    add(
        "(tell) (me) (about) (yourself|you)",
        lambda b, e: call("Question", about=call("Identity", subject=call("Assistant"))),
    )
    add(
        "(tell) (me) (about) {x:np}",
        lambda b, e: call("Recall", about=b["x"]),
    )
    add(
        "(what) (are) (you)",
        lambda b, e: call("Question", about=call("Identity", subject=call("Assistant"))),
    )
    add(
        "(what) (does) {w:word} (mean)",
        lambda b, e: call("Explain", concept=call(concept_name(_text(b["w"])))),
    )
    add(
        "(what) (is) (a|an) {w:word}",
        lambda b, e: call("Explain", concept=call(concept_name(_text(b["w"])))),
    )
    add(
        "(do) (you) (know) (what) {w:word} (is|means)",
        lambda b, e: call("Explain", concept=call(concept_name(_text(b["w"])))),
    )
    add("(why) [is] (that|it|this)", lambda b, e: call("Explain", concept=call("Ref", Lit("that"))))
    add("(explain|why)", lambda b, e: call("Explain", concept=call("Ref", Lit("that"))))
    add("(how) (do) (you) (know) [that]", lambda b, e: call("Explain", concept=call("Ref", Lit("that"))))

    # -- teaching ----------------------------------------------------------
    add(
        "(to) {v:word} (something) (means|is) (to) {a:defn}",
        lambda b, e: _teaching(b["v"], b["a"], always_parameterised=True),
    )
    add(
        "{w:word} (means) {a:defn}",
        lambda b, e: _teaching(b["w"], b["a"]),
    )
    add(
        "(to) {w:word} (means|is) (to) {a:defn}",
        lambda b, e: _teaching(b["w"], b["a"]),
    )
    add(
        "(define) {w:word} (as) {a:defn}",
        lambda b, e: _teaching(b["w"], b["a"]),
    )
    add(
        "{w:word} (is) (short) (for) {a:defn}",
        lambda b, e: _teaching(b["w"], b["a"]),
    )

    # -- questions ---------------------------------------------------------
    add(
        "(what|how much|how many) (is|are|was) {v:value}",
        lambda b, e: call("Question", about=b["v"]),
    )
    add(
        "(how) (many) {x:np} (are) (there)",
        lambda b, e: call("Question", about=call("Count", collection=b["x"])),
    )
    add(
        "(how) (many) {x:word} (are|is) (in|of) {c:collection}",
        lambda b, e: call("Question", about=call("Count", collection=b["c"])),
    )
    add(
        "(how) (many) (are|is) (in|of) {c:collection}",
        lambda b, e: call("Question", about=call("Count", collection=b["c"])),
    )
    add(
        "(how) (many) {x:word} (in) {c:collection}",
        lambda b, e: call("Question", about=call("Count", collection=b["c"])),
    )
    add(
        "(how) (old) (is|are|am) {x:np}",
        lambda b, e: call("Question", about=call("Age", subject=b["x"])),
        bonus=2,
    )
    add(
        "(how) {w:word} (is|are|am) {x:np}",
        lambda b, e: call("Question", about=call(_attr_for(_text(b["w"])), subject=b["x"])),
    )
    add(
        "(who|what) {v:word} {x:np}",
        lambda b, e: _who_verb(b, e),
        question=True,
    )
    add(
        "(who|what) (is|are) {x:np}",
        lambda b, e: call("Question", about=_identity_of(b["x"])),
    )
    add(
        "(where) (is|are) {x:np}",
        lambda b, e: call("Question", about=call("Location", subject=b["x"])),
    )
    add(
        "(is|are|was|were) {x:value} {c:comp}",
        lambda b, e: call("Ask", proposition=_attach_subject(b["c"], b["x"])),
        bonus=3,
    )
    add(
        "(is|are|was|were) {p:prop}",
        lambda b, e: call("Ask", proposition=b["p"]),
    )
    add(
        "(is|are) (it|that|this) (true) (that) {p:prop}",
        lambda b, e: call("Ask", proposition=b["p"]),
    )
    add(
        "(do|does|did) {p:prop}",
        lambda b, e: call("Ask", proposition=b["p"]),
    )
    add(
        "(do) (you) (know) {p:prop}",
        lambda b, e: call("Ask", proposition=b["p"]),
    )
    add(
        "(can) (you) {a:action}",
        lambda b, e: call("Request", action=b["a"]),
    )
    add(
        "(could|would|will) (you) [please] {a:action}",
        lambda b, e: call("Request", action=b["a"]),
    )
    add(
        "(i) (want) (you) (to) {a:action}",
        lambda b, e: call("Request", action=b["a"]),
    )
    add(
        "(what) (about) {v:value}",
        lambda b, e: call("Question", about=b["v"]),
    )
    add(
        "(and) {v:value}",
        lambda b, e: call("Question", about=b["v"]),
        question=True,
    )

    # -- statements --------------------------------------------------------
    add(
        "(my|the) (name) (is) {w:word}",
        lambda b, e: call(
            "Remember",
            proposition=call("Name", subject=call("User"), value=b["w"]),
        ),
    )
    add(
        "(i) (am) (called|named) {w:word}",
        lambda b, e: call(
            "Remember", proposition=call("Name", subject=call("User"), value=b["w"])
        ),
    )
    add(
        "(i) (am) {n:number} [years old]",
        lambda b, e: call(
            "Remember", proposition=call("Age", subject=call("User"), value=b["n"])
        ),
    )
    add(
        "(remember) [that] {p:prop}",
        lambda b, e: call("Remember", proposition=b["p"]),
    )
    add(
        "(forget) [about] {x:np}",
        lambda b, e: call("Forget", about=b["x"]),
    )
    add(
        "(if) {c:prop} (then) {t:prop}",
        lambda b, e: call("Learn", rule=call("Conditional", condition=b["c"], then=b["t"])),
    )
    add("{p:prop}", lambda b, e: call("Remember", proposition=b["p"]), question=False, bonus=-5)

    # -- imperatives -------------------------------------------------------
    add("{a:action}", lambda b, e: call("Request", action=b["a"]), question=False, bonus=-7)
    add("{v:value}", lambda b, e: _bare_value(b["v"]), bonus=-9)

    return out


def _text(expr: Expr) -> str:
    if isinstance(expr, Lit):
        return str(expr.value)
    if isinstance(expr, Call):
        return expr.concept
    return str(expr)


def _attr_for(word: str) -> str:
    return _WORD_CONCEPT.get(word.lower(), concept_name(word))


def _identity_of(x: Expr) -> Expr:
    """"who is Greg" asks for Greg, and whatever Soup can say about him."""
    return x


_NOT_A_VERB = {
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "and",
    "or",
    "not",
    "than",
    "about",
}


def _who_verb(bindings: Dict[str, Expr], ears: "Ears") -> Optional[Expr]:
    word = _text(bindings["v"]).lower()
    if word in _NOT_A_VERB or _NUMERIC.match(word):
        return None
    concept = _IRREGULAR_STEM.get(word) or concept_name(_lemma(word))
    return call(
        "Query",
        pattern=call(concept, subject=call("Who"), object=bindings["x"]),
    )


_PARAM = "x"


def _teaching(word: Expr, body: Expr, always_parameterised: bool = False) -> Expr:
    """Build a `Teach` whose head takes a parameter only if the body uses one."""
    from .expr import walk

    name = concept_name(_text(word))
    uses_param = always_parameterised or any(
        isinstance(node, Var) and node.name == _PARAM for node in walk(body)
    )
    head = call(name, Var(_PARAM)) if uses_param else call(name)
    return call("Teach", concept=head, meaning=body)


_SUBJ = "_subj"


def _attach_subject(predicate: Expr, subject: Expr) -> Expr:
    """Undo subject-aux inversion: "is alice older than bob"."""
    from .expr import substitute

    return substitute(predicate, {_SUBJ: subject})


def _bare_value(v: Expr) -> Optional[Expr]:
    if isinstance(v, Call) and v.concept in ("Ref", "Unintelligible"):
        return None
    return call("Question", about=v)
