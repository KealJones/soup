"""The seat a model sits in, and the three things it is allowed to be asked.

Reading English is the one part of this design that is genuinely solved better
elsewhere, so a model does it. The seat is where it sits, and the point of
having a named place for it is that the place is small. Everything the model
may be asked is here, in three methods, and nothing else in Soup talks to it.

  hear()    what did this sentence mean? shape only, never an answer.
  define()  what does this concept mean, in terms of ones we have?
  answer()  what is the value of this, if nothing else could tell us?

They are in order of how much they are worth. `define` is the valuable one: a
definition is bought once and kept as an ordinary rule, so it makes Soup
permanently better, while an answer helps exactly once and is often the thing
a small model gets wrong. Asked to quintuple ten, qwen guesses a number.
Asked what quintupling is, it says Multiply(x, 5).

Whatever comes back is checked rather than believed. Concepts invented out of
thin air are rejected, restatements of the question are rejected, and a
definition made of things Soup does not have is rejected by the Teacher. An
unknown concept that survives is not a failure: it is exactly what the Teacher
and the lookup exist for.

Talks to Ollama's native /api/chat or any OpenAI-compatible
/v1/chat/completions endpoint over stdlib urllib, so Ollama and LM Studio
work out of the box and nothing gets installed. No server running means no
seat, and the ears carry on alone.

Ollama's native endpoint is the default because it is the only one of the two
that can actually turn thinking off, and a reasoning model will otherwise
spend fifty seconds deliberating over one line of syntax.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

from .expr import Call, Expr, Lit, Seq, render
from .parse import ParseError, parse

__all__ = ["Seat", "seat_from_env"]

DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"

_SYSTEM = '''You translate English into Soup concept expressions. You are the
ears of a system, not its brain. Reply with exactly one expression and nothing
else: no prose, no code fences, no explanation.

SYNTAX
  Concept(name=Value, name=Value)   a concept applied to named arguments
  Concept()                          a concept on its own
  "text"  42  3.14  [1, 2, 3]        literal text, numbers, lists

Concept names are CapitalizedCamelCase. Argument names are lowercase.

MOOD: wrap the whole thing in exactly one of these.
  Question(about=...)      they asked something
  Request(action=...)      they told you to do something
  Remember(proposition=...) they stated something
  Greeting() Farewell() Thanks() Affirm() Deny() Laugh()

RULES
1. Say what the sentence MEANS, never how to accomplish it. "delete the
   biggest file" is Delete(target=Maximum(collection=Files(), by=FileSize())),
   never a loop or a sequence of steps.
1b. Do not work anything out, and above all do not do arithmetic. If a word
   names an idea then that idea is the concept, even when you happen to know
   what it unpacks to. "sextupled" is Sextuple(x). It is not Multiply(x, 6)
   and it is certainly not Power(x, 6). Unpacking concepts is somebody
   else's job here and you will be asked about it separately; guessing at it
   now only replaces a word we could have looked up with a number we cannot.
2. If you do not know a concept name, invent a clear one. Inventing
   MostAdorable() is correct and useful. Do not substitute something you do
   know but that means something else.
3. Keep the structure of the original sentence. Prepositions make good
   argument names: "the cat sat on the mat" is
   Remember(proposition=Sat(subject=Cat(), on=Mat())).
4. Use Ref("it") / Ref("that") for pronouns pointing at earlier turns.
5. Verbs go in their base form: "jumps" and "jumped" are both Jump, "gave"
   is Give. A concept is the same concept whatever tense it arrived in.
6. Keep every clause. "sort these and show me the top three" is two things
   being asked for, so it is Seq(Sort(...), Show(...)), not just the first
   one. Dropping half a sentence is worse than an awkward reading of it.
7. Name an attribute as fully as the sentence does. "when was X born" is
   DateOfBirth(subject=X), never Date(subject=X); "how tall" is Height, not
   Size. A vague name collides with a different concept and gets answered
   confidently wrongly, which is the worst outcome available.

EXAMPLES
what is 6 times 4
Question(about=Multiply(6, 4))

make this picture look spooky but still cute
Request(action=Make(target=Ref("this"), look=AllOf(Spooky(), Cute())))

Alice is 30 years old
Remember(proposition=Age(subject=Alice(), value=30))

who has the car
Question(about=Query(pattern=Has(subject=Who(), object=Car())))

when was albert einstein born
Question(about=DateOfBirth(subject=AlbertEinstein()))

delete the biggest file in my downloads folder
Request(action=Delete(target=Maximum(collection=AllOf(kind=File(), within=Downloads()), by=FileSize())))

why is the sky blue
Question(about=Why(proposition=Is(subject=Sky(), value=Blue())))

spooky means creepy and dark
Teach(concept=Spooky(), meaning=AllOf(Creepy(), Dark()))

double it
Request(action=Double(Ref("it")))

lmaooo
Laugh()

she gave him a book yesterday
Remember(proposition=Give(subject=She(), object=Book(), to=He(), time=Yesterday()))

sort these by size and show me the top three
Request(action=Seq(Sort(collection=Ref("these"), by=Size()), Show(recipient=User(), object=Top(3))))'''


# What each kind of concept is for. The model does much better at reusing a
# concept when it can see what sort of thing it is looking at.
_KIND_BLURB = (
    ("operation", "things that compute a result"),
    ("relation", "things that hold between two things"),
    ("attribute", "properties of one thing"),
    ("quality", "adjectives"),
    ("entity", "particular named things"),
    ("thing", "kinds of thing"),
    ("modality", "how certain or obliged something is"),
    ("speech", "asking, answering, acknowledging"),
    ("value", "literals and units"),
)

_CONVENTIONS = '''ARGUMENT NAMES
Use these, consistently. They are how the rest of the system reads you.
  subject=   who or what the thing is about
  value=     the value of an attribute      Age(subject=Alice(), value=30)
  object=    what a verb was done to        Has(subject=Greg(), object=Car())
  kind=      what something is a kind of    IsA(subject=Greg(), kind=Person())
  quality=   an adjective on a noun         Dog(quality=Lazy())
  target=    what an action acts on         Delete(target=File())
  collection= a group being operated over   Count(collection=Files())
  of=, in=, on=, with=, from=, over=, to=, for=
             a preposition from the sentence, used as written

A multi-word name is ONE concept: "new zealand" is NewZealand(), "the eiffel
tower" is EiffelTower(), "albert einstein" is AlbertEinstein(). Never split a
name into a concept per word, and never make a list out of it.

An adjective on a noun stays an argument: "the lazy dog" is Dog(quality=Lazy()).

"who is X" and "what is X" ask what X is: Question(about=Identity(subject=X)).
"how tall is X" asks an attribute: Question(about=Height(subject=X)).
"the P of X" is P(subject=X): Question(about=Capital(subject=France())).'''


_ANSWERING = '''You answer questions written as Soup concept expressions.
Reply with exactly one concept expression and nothing else: no prose, no code
fences, no explanation.

SYNTAX
  "text"  42  3.14  [1, 2, 3]        literal text, numbers, lists
  Concept(name=Value)                a concept applied to named arguments

Answer with a literal wherever a literal will do. Keep it to one short
sentence of text at most.

Never repeat the question back. Never wrap the answer in the concept you
were asked about. Just the answer on its own.

If you do not know, or the question has no factual answer, reply exactly:
Unknown()

EXAMPLES
  asked Identity(subject=AlbertEinstein())
  reply "a german-born physicist who came up with relativity"

  asked Capital(of=France())
  reply "Paris"

  asked Age(subject=User())
  reply Unknown()

  asked Population(subject=Tokyo())
  reply 37000000'''


_DEFINING = '''You are given a concept Soup does not know, and the concepts it
does know. Say what the unknown one MEANS, in terms of the known ones.

Reply with exactly one concept expression and nothing else: no prose, no code
fences, no explanation.

This is a definition, not an answer. You are not being asked what the value is,
you are being asked what the concept is the same as, so that Soup can work the
value out for itself from now on.

RULES
1. Use ONLY concepts from the known list. A definition made of things Soup
   also does not know teaches it nothing and will be thrown away.
2. Keep the parameter names from the signature you were given, and pass them
   on. A parameter is passed by writing its bare name:
     Quadruple(x)  means  Multiply(x, 4)        <- passes x. correct.
     Quadruple(x)  means  Multiply(x=4)         <- names an argument x. wrong.
   Every parameter in the signature must appear in the definition.
3. Define the concept, not this one case. Given Nephew(subject) do not reply
   with somebody's actual nephew.
4. If nothing in the known list comes close, reply exactly: Unknown()
   That is a perfectly good answer and much better than a wrong one.

EXAMPLES
  unknown  Quadruple(x)
  means    Multiply(x, 4)

  unknown  MilitaryTime()
  means    TwentyFourHourTime()

  unknown  Grandparent(subject)
  means    Parent(subject=Parent(subject=subject))

  unknown  Cheapest(collection)
  means    Minimum(collection=collection, by=Price())

  unknown  ShorterThan(left, right)
  means    LessThan(left=Height(subject=left), right=Height(subject=right))

  unknown  Spooky()
  means    AllOf(Creepy(), Dark())

  unknown  Quixotic(subject)
  means    Unknown()'''


class Seat:
    """A model sitting in the ears, on a short leash."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.model = model
        self.key = key
        self.timeout = timeout
        self.calls = 0
        self.failures = 0
        # One dead connection is enough. Nobody wants every turn to wait for
        # the same refused socket.
        self.available = True
        self.last_error: str = ""
        # The concept vocabulary, rendered once. It changes slowly and the
        # prompt is sent on every utterance, so rebuilding it per turn would
        # be waste.
        self.brief: str = ""

    def learn_vocabulary(self, knowledge) -> None:
        self.brief = vocabulary_brief(knowledge)

    def hear(self, utterance: str, vocabulary: Optional[List[str]] = None) -> Optional[Expr]:
        if not self.available:
            return None
        prompt = _SYSTEM + "\n\n" + _CONVENTIONS
        if self.brief:
            prompt += "\n\n" + self.brief
        elif vocabulary:
            prompt += "\n\nCONCEPTS ALREADY KNOWN (prefer these when they fit):\n"
            prompt += ", ".join(sorted(vocabulary)[:220])
        reply = self._ask(prompt, utterance)
        if reply is None:
            return None
        try:
            heard = parse(_clean(reply))
        except (ParseError, ValueError, IndexError):
            self.last_error = "unparseable: %s" % reply[:120]
            return None
        invented = _unfaithful(heard, utterance, vocabulary or [])
        if invented:
            # Small models will cheerfully answer "who is albert einstein"
            # with a concept named Alice. The ears are allowed to be baffled;
            # they are not allowed to make things up.
            self.last_error = "invented %s, not in the utterance" % ", ".join(invented)
            return None
        return heard

    @property
    def native(self) -> bool:
        """Ollama's own endpoint, as opposed to the OpenAI-shaped one."""
        return self.url.rstrip("/").endswith("/api/chat")

    def _payload(self, system: str, utterance: str) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterance},
        ]
        if self.native:
            return {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # One line of syntax needs no deliberation. Only Ollama's
                # native API actually honours this; the compatibility layer
                # quietly ignores it and burns the budget on reasoning.
                "think": False,
                "options": {"temperature": 0, "num_predict": 300},
            }
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def answer(self, expr: Expr) -> Optional[Expr]:
        """Ask the model to resolve a concept expression we could not.

        Separate from `hear` on purpose. Hearing is about shape and must not
        answer anything; this one is about the world and must not reshape
        the question.
        """
        if not self.available:
            return None
        expression = render(expr, False)
        reply = self._ask(_ANSWERING, expression)
        if reply is None:
            return None
        answer = _answer_in(reply, expression)
        if answer is None:
            self.last_error = "unparseable answer: %s" % reply[:120]
            return None
        if isinstance(answer, Call) and answer.concept in ("Unknown", "Nothing"):
            return None
        head = expression.split("(", 1)[0]
        if isinstance(answer, Call) and answer.concept == head:
            # Models like to restate the question with the answer filled in.
            # Take the filling; refuse the restatement.
            inner = answer.get("value") or answer.get("answer")
            if inner is None:
                self.last_error = "echoed the question"
                return None
            answer = inner
        if render(answer, False) == expression:
            self.last_error = "echoed the question"
            return None
        if _same_shape(answer, expression) or _says_nothing_new(answer, expression):
            # Relabelling the question is not answering it, and neither is
            # wrapping a word from it in a concept: asked what military time
            # is, qwen replies Concept(name="Military").
            self.last_error = "restated the question"
            return None
        return answer

    def define(self, signature: str, vocabulary: List[str]) -> Optional[Expr]:
        """Ask for a concept's meaning, not its value.

        The third thing a seat can be asked, and the most useful one. `hear`
        buys a single sentence and `answer` buys a single fact, but a
        definition is bought once and kept: it goes through the Teacher like
        any human answer, becomes an ordinary rule, and every later sentence
        using the concept is then resolved by Soup itself, offline and for
        free. Being told what "military time" means is worth more than being
        told what time it is.
        """
        if not self.available:
            return None
        prompt = _DEFINING
        if self.brief:
            prompt += "\n\n" + self.brief
        elif vocabulary:
            prompt += "\n\nKNOWN CONCEPTS:\n" + ", ".join(sorted(vocabulary)[:220])
        reply = self._ask(prompt, "unknown  %s\nmeans" % signature)
        if reply is None:
            return None
        try:
            meaning = parse(_clean(reply))
        except (ParseError, ValueError, IndexError):
            self.last_error = "unparseable definition: %s" % reply[:120]
            return None
        if isinstance(meaning, Call) and meaning.concept in ("Unknown", "Nothing"):
            return None
        if render(meaning, False) == signature:
            self.last_error = "defined the concept as itself"
            return None
        return meaning

    def _ask(self, system: str, utterance: str) -> Optional[str]:
        body = json.dumps(self._payload(system, utterance)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer %s" % self.key
        request = urllib.request.Request(self.url, data=body, headers=headers)
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if self.native:
                return payload["message"]["content"]
            return payload["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError) as exc:
            self.failures += 1
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                # Slow, not absent. Worth another go on the next sentence.
                self.last_error = "timed out after %gs" % self.timeout
                return None
            # Nothing listening, so stop trying for the rest of the session.
            self.available = False
            self.last_error = "%s (%s)" % (exc, self.url)
            return None
        except (KeyError, IndexError, ValueError) as exc:
            self.failures += 1
            self.last_error = "bad response: %s" % exc
            return None


# Shapes the ears are always allowed to reach for, whether or not the word
# appears in what was said. Mood and structure are ours, not the speaker's.
_STRUCTURAL = frozenset(
    [
        "Question", "Request", "Remember", "Teach", "Seq", "AllOf", "AnyOf",
        "Ref", "Query", "Unknown", "Identity", "Assertion", "Not", "Value",
    ]
)


def _camel_words(name: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)]


# Words with no content of their own. A model that renders "him" as He() or
# "a bit" as Some() is normalising, not inventing.
_FUNCTION_WORDS = frozenset(
    """he she it they them him her his hers its their theirs i me my we us our
    you your this that these those one ones some any all none both each every
    other another same self thing things someone somebody something anyone
    anybody anything everyone everything nobody nothing here there now then
    today yesterday tomorrow true false yes no""".split()
)


def _leaves(expr: Expr) -> List[str]:
    """Concept names used as bare values, with no arguments of their own.

    These are the names of *things*: entities, qualities, references. A
    wrongly invented one is a thing the speaker never mentioned, which is
    the invention that matters.
    """
    if isinstance(expr, Call):
        if not expr.args:
            return [expr.concept]
        found: List[str] = []
        for arg in expr.args:
            found.extend(_leaves(arg.value))
        return found
    if isinstance(expr, Seq):
        found = []
        for item in expr.items:
            found.extend(_leaves(item))
        return found
    return []


def _unfaithful(expr: Expr, utterance: str, vocabulary: List[str]) -> List[str]:
    """Things the model introduced that the speaker never mentioned.

    Only the leaves are policed, and deliberately so. A model that reads
    "turn the volume down" as Decrease(target=Volume()) has named a relation
    we did not say, which is exactly the paraphrasing we want from it: the
    sentence has no word "decrease" in it and Decrease is still the right
    answer. A model that reads "who is albert einstein" as
    Who(subject=Alice()) has put a person into the sentence who was never in
    it, and no amount of good intent makes that recoverable.

    So heads may be invented freely and things may not.
    """
    said = set(re.findall(r"[a-z0-9]+", utterance.lower()))
    known = set(vocabulary) | _STRUCTURAL
    invented = []
    for name in _leaves(expr):
        if name in known or name in invented:
            continue
        words = _camel_words(name)
        if all(w in _FUNCTION_WORDS or _echoes(w, said) for w in words):
            continue
        invented.append(name)
    return invented


def _echoes(word: str, said: set) -> bool:
    """Was this word really spoken, allowing for the fact that words inflect?

    "sextupled" is a perfectly ordinary way to say Sextuple, and "cities" is
    City. Demanding the letters match exactly would have the guard rejecting
    the very paraphrasing it exists to permit, so a shared stem counts. The
    length floor is what keeps that from becoming a loophole: short words
    would otherwise collide their way through on a prefix or two.
    """
    if word in said:
        return True
    if len(word) < 4:
        return False
    stem = word[:-1]
    return any(
        spoken.startswith(stem) or word.startswith(spoken[:-1])
        for spoken in said
        if len(spoken) >= 4
    )


def vocabulary_brief(knowledge, per_kind: int = 70) -> str:
    """The concepts soup already has, grouped so a model can reuse them.

    This is the difference between a model that invents `Times` and `HowOld`
    alongside the `Multiply` and `Age` we already own, and one that lands on
    the vocabulary the realizer can actually do something with. A flat
    alphabetical dump does not achieve that; showing what sort of thing each
    concept is does.
    """
    grouped = {}
    for name, defined in knowledge.concepts.items():
        grouped.setdefault(defined.kind, []).append(name)

    lines = ["CONCEPTS SOUP ALREADY KNOWS. Reuse these names wherever they fit."]
    for kind, blurb in _KIND_BLURB:
        names = sorted(grouped.get(kind, []))
        if not names:
            continue
        lines.append("%s (%s):" % (kind, blurb))
        lines.append("  " + ", ".join(names[:per_kind]))
    lines.append(
        "Anything not on that list, invent a clear CamelCase name for. "
        "Inventing is expected; reaching for a concept that means something "
        "else is not."
    )
    return "\n".join(lines)


def _answer_in(reply: str, question: str) -> Optional[Expr]:
    """The first line of the reply that is an answer rather than the question.

    Models trained on question/answer pairs will happily restate the question
    before answering it, so walk the lines instead of trusting the first.
    """
    body = _clean(reply, single=False)
    for line in body.splitlines():
        line = line.strip().lstrip("-*> ").strip()
        if line.startswith("reply "):
            line = line[6:].strip()
        if not line or line == question:
            continue
        try:
            return parse(_balance(line))
        except (ParseError, ValueError, IndexError):
            plain = line.rstrip(".")
            if plain and len(plain) < 200 and not _refusal(plain):
                # "100 degrees celsius" is a perfectly good answer that
                # simply forgot its quotation marks.
                return Lit(plain)
    return None


def _refusal(line: str) -> bool:
    low = line.lower()
    return any(
        phrase in low
        for phrase in ("i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai")
    )


def _says_nothing_new(answer: Expr, question: str) -> bool:
    """An answer built entirely out of words that were in the question.

    Not applied to plain literals, because "Paris" answering the capital of
    France is a literal and telling us something. This is about a concept
    wrapped around a word we already had.
    """
    if not isinstance(answer, Call) or not answer.args:
        return False
    asked = set(re.findall(r"[a-z0-9]+", question.lower()))
    pieces = re.findall(r"[a-z0-9]+", render(answer, False).lower())
    return bool(pieces) and all(p in asked or p in _STOP for p in pieces)


_STOP = frozenset(["concept", "name", "value", "meaning", "thing", "label", "of", "is"])


def _same_shape(answer: Expr, question: str) -> bool:
    """Same arguments under a different name. A rename, not an answer."""
    if not isinstance(answer, Call) or not answer.args:
        return False
    inside = question.split("(", 1)[-1].rsplit(")", 1)[0]
    return render(answer, False).split("(", 1)[-1].rsplit(")", 1)[0] == inside


def _clean(reply: str, single: bool = True) -> str:
    """Models like to wrap things in fences and add a sentence of pride.

    With `single` off, the surviving lines come back whole so the caller can
    choose between them rather than taking the first that looks right.
    """
    text = reply.strip()
    while "<think>" in text and "</think>" in text:
        head, _, rest = text.partition("<think>")
        text = (head + rest.partition("</think>")[2]).strip()
    if "```" in text:
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.lower().startswith(("python", "soup", "text")):
                text = text.split("\n", 1)[-1]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not single:
        return "\n".join(lines)
    for line in lines:
        if "(" in line and line[:1].isupper():
            return _balance(line)
    return _balance(lines[0]) if lines else text


def _balance(line: str) -> str:
    """Small models drop a closing bracket about one time in ten.

    Counting them back up is a repair, not a guess: there is exactly one way
    to close an expression that is otherwise well formed, and refusing a
    parse over a missing keystroke helps nobody.
    """
    depth = 0
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"' and (i == 0 or line[i - 1] != "\\"):
            quoted = not quoted
        elif not quoted:
            depth += (ch == "(") - (ch == ")")
    if depth > 0:
        return line + ")" * depth
    while depth < 0 and line.endswith(")"):
        line, depth = line[:-1], depth + 1
    return line


def seat_from_env() -> Optional[Seat]:
    """A seat if the environment asks for one, otherwise nothing at all."""
    url = os.environ.get("SOUP_LLM_URL")
    model = os.environ.get("SOUP_LLM_MODEL")
    if not url and not model:
        return None
    return Seat(
        url=url or DEFAULT_URL,
        model=model or DEFAULT_MODEL,
        key=os.environ.get("SOUP_LLM_KEY"),
        timeout=float(os.environ.get("SOUP_LLM_TIMEOUT", "30")),
    )
