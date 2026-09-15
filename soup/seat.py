"""The seat a model sits in, and the three things it is allowed to be asked.

Reading English is the one part of this design that is genuinely solved better
elsewhere, so a model does it. The seat is where they sit. Hear stays on a small fast model; define and
answer may sit in a larger one. Everything they may be asked is here, in
three methods, and nothing else in Soup talks to them.

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

from .expr import Call, Expr, Lit, Seq, render, walk
from .parse import ParseError, parse

__all__ = ["Seat", "seat_from_env"]

DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_TEACHER = "qwen3.8:27b"

_SYSTEM = '''You translate English into Soup concept expressions. You are the
ears of a system, not its brain. Reply with exactly one expression and nothing
else: no prose, no code fences, no explanation.

SYNTAX
  Foo(bar=Value, baz=Value)          a concept applied to named arguments
  Foo()                              a concept on its own
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
   "I" / "me" / "my" is User(), never Developer() or Person(). The speaker
   already has a name.
5. Verbs go in their base form: "jumps" and "jumped" are both Jump, "gave"
   is Give. A concept is the same concept whatever tense it arrived in.
6. Keep every clause. "sort these and show me the top three" is two things
   being asked for, so it is Seq(Sort(...), Show(...)), not just the first
   one. Dropping half a sentence is worse than an awkward reading of it.
7. Name an attribute as fully as the sentence does. "when was X born" is
   DateOfBirth(subject=X), never Date(subject=X); "how tall" is Height, not
   Size. A vague name collides with a different concept and gets answered
   confidently wrongly, which is the worst outcome available.
8. Keep every detail, and keep it as structure. When a sentence sets up a
   situation and then asks about it, the situation is a list of
   propositions and it goes in given=[...] on the question. Four people
   doing four things is four propositions; it is not a list of four names,
   and the four things are the entire point of the question. A detail you
   drop cannot be reasoned about by anybody downstream, and being long is
   not a problem here: being lossy is.
8b. Numeric inputs to a computation are NAMED ARGUMENTS on the stem concept,
   not items in a given list. "beginning inventory $30,000, purchases
   $87,500, net sales $102,000, gross profit rate 40%" is four named
   arguments: EndingInventory(beginningInventory=30000, purchases=87500,
   netSales=102000, grossProfitRate=40). Use given=[...] only for scenes
   where multiple actors do different things (the sisters example). Named
   arguments become parameter names downstream, and a formula that says
   Plus(beginningInventory, purchases) is useful where Nth(given, 1) is not.
9. A clause is nested concepts. Never one CamelCase sentence, and never
   Concept(name="a whole phrase"). "ensuring that one person does not
   carry a whole task" is Ensure(that=Not(Carry(subject=Person(),
   object=Task(quality=Whole())))). Identity(subject=X) is only for a
   named person or thing (Einstein, Chess), never for a description of
   an act. Packed names cannot be realized; nested ones can.

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

what is the min number of players for chess
Question(about=Players(subject=Chess()))

delete the biggest file in my downloads folder
Request(action=Delete(target=Maximum(collection=AllOf(kind=File(), within=Downloads()), by=FileSize())))

search the web for concept-based AI assistants
Request(action=WebSearch(query="concept-based AI assistants"))

read https://example.org/guide
Request(action=VisitWebpage(url="https://example.org/guide"))

look up Ada Lovelace on Wikipedia
Request(action=WikipediaSearch(query="Ada Lovelace"))

transcribe the recording at /tmp/meeting.wav
Request(action=TranscribeAudio(path="/tmp/meeting.wav"))

create greet.py and main.py in /tmp/scratch
Request(action=Create(target=Directory(path="/tmp/scratch"), files=[File(path="greet.py", contents="def hello():\\n    return 'hi'\\n"), File(path="main.py", contents="from greet import hello\\nprint(hello())\\n")]))

change hello.py so it prints goodbye instead of hello
Request(action=Change(target=File(path="hello.py"), from="hello", to="goodbye"))

why is the sky blue
Question(about=Why(proposition=Is(subject=Sky(), value=Blue())))

spooky means creepy and dark
Teach(concept=Spooky(), meaning=AllOf(Creepy(), Dark()))

how do you play chess
Question(about=HowTo(Play(object=Chess())))

double it
Request(action=Double(Ref("it")))

lmaooo
Laugh()

she gave him a book yesterday
Remember(proposition=Give(subject=She(), object=Book(), to=He(), time=Yesterday()))

sort these by size and show me the top three
Request(action=Seq(Sort(collection=Ref("these"), by=Size()), Show(recipient=User(), object=Top(3))))

there are three sisters in a room: ann is reading, kate is playing chess. what is the third sister doing
Question(about=Doing(subject=Third(of=Sisters(count=3))), given=[Read(subject=Ann()), Play(subject=Kate(), object=Chess())])

A total of 30 players, 5 per team. Which statement explains how to find the number of teams?

A. Multiply 5 by 5 to find 25 teams.
B. Divide 30 by 5 to find 6 teams.
Question(about=Choose(among=[Option(letter="A", text="Multiply 5 by 5 to find 25 teams."), Option(letter="B", text="Divide 30 by 5 to find 6 teams.")], by=Teams(players=30, per=5)))

A company had beginning inventory $30,000, purchases $87,500, net sales $102,000 and a gross profit rate of 40%. What is the ending inventory?

A. $50,200
B. $45,100
C. $56,300
Question(about=Choose(among=[Option(letter="A", text="$50,200"), Option(letter="B", text="$45,100"), Option(letter="C", text="$56,300")], by=EndingInventory(beginningInventory=30000, purchases=87500, netSales=102000, grossProfitRate=40)))

as what is ensuring that one person does not carry a whole task referred to
Question(about=Ensure(that=Not(Carry(subject=Person(), object=Task(quality=Whole())))))'''


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

"who is X" and "what is X" ask what X is, when X is a named person or thing:
Question(about=Identity(subject=AlbertEinstein())). Never Identity of a
clause, and never Concept(name="..."). A description of an act stays nested.
"how tall is X" asks an attribute: Question(about=Height(subject=X)).
"the P of X" is P(subject=X): Question(about=Capital(subject=France())).'''


_ANSWERING = '''You answer questions written as Soup concept expressions.
Reply with exactly one concept expression and nothing else: no prose, no code
fences, no explanation.

SYNTAX
  "text"  42  3.14  [1, 2, 3]        literal text, numbers, lists
  Foo(bar=Value)                     a concept applied to named arguments

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


_DEFINING = '''You are given a concept expression Soup cannot yet realize, and
the concepts it does know. Say what it MEANS, in terms of the known ones.

Reply with exactly one concept expression and nothing else: no prose, no code
fences, no explanation.

This is a definition, not an answer. You are not being asked what the value is,
you are being asked what the concept is the same as, so that Soup can work the
value out for itself from now on. The definition is kept and reused.

RULES
1. Prefer composing concepts from the known list. A definition made of things
   Soup also does not know teaches it nothing and will be thrown away.
2. Keep the parameter names from the signature you were given, and pass them
   on. A parameter is passed by writing its bare name:
     Quadruple(x)  means  Multiply(x, 4)        <- passes x. correct.
     Quadruple(x)  means  Multiply(x=4)         <- names an argument x. wrong.
   Every parameter in the signature must appear in the definition.
3. Define the concept, not this one case. Given Nephew(subject) do not reply
   with somebody's actual nephew.
3b. An "in" line is where the word actually turned up. Read the word the way
   that sentence uses it, then define it generally. Never answer the "in"
   line; it is context, not the question.
3c. Never define a concept using itself. Wrapping it in something else is
   still using itself, and the definition will be thrown away.
3d. Argument names are lowercase. Concept names are capitalised. RealizedBy,
   HasProperty and InverseOf describe how Soup files knowledge away and are
   never a definition of anything; do not reach for them.
4. If the machine itself can work the value out (this computer's clock, files,
   environment) and no known concept already does, Python("...") or
   Shell("...") is a real realization. Keep it. Pass parameters as arguments,
   not inside the string:
     ExpandHome(path)  means  Python("os.path.expanduser(path)", path=path)
   Do not use Python or Shell for facts about people.
5. If only a person would know, or nothing is close, reply exactly: Unknown()
6. Do not define a concept as Concat, Join, or Equals(..., True). Those teach nothing.

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

  in       Doing(subject=Fifth(of=Sisters(count=5)))
  unknown  Fifth(of)
  means    Nth(collection=of, index=5)

  unknown  HomeDirectory()
  means    Python("os.path.expanduser('~')")

  unknown  KernelName()
  means    Shell("uname -s")

  unknown  Spooky()
  means    AllOf(Creepy(), Dark())

  unknown  Quixotic(subject)
  means    Unknown()'''


_SPEAKING = '''You are the mouth of a system. You are given its answer as a
concept expression. Say that answer in English, the way a person would say it
out loud.

Reply with exactly one short line and nothing else: no quotes, no preamble, no
explanation, no markdown.

RULES
1. Say only what the expression says. Never add a fact, a number, a name, a
   caveat or a unit that is not in it. You are choosing words, not answering.
2. Never mention concept names, argument names, brackets or the word
   "expression". Nobody wants to hear "Answer value 240".
3. Answer(value=V, to=Q) is the reply to a question: say V as a natural
   answer, not as a restatement of Q.
4. Answer(letter="B", value=V) is a multiple choice pick. Start the line with
   that letter, a full stop, then V. Exactly: B. <V>
5. Acknowledged() means something was written down: say so briefly.
6. Unknown(about=X) means it genuinely could not be worked out: say you do not
   know, and name X.
7. Ask(of=User(), about=X) is a question for the person you are talking to.
   Ask it, in the second person, and ask nothing else.
8. Numbers stay as digits, with whatever unit they came with. 240 W is not
   "two hundred and forty watts" and 30 is not "thirty".
9. Casual and short. Lowercase is fine. No exclamation marks.

EXAMPLES
  Answer(value=24, to=Multiply(6, 4))
  it's 24

  Answer(letter="B", value="Divide 30 by 5 to find 6 teams.")
  B. Divide 30 by 5 to find 6 teams.

  Answer(value="Paris", to=Capital(subject=France()))
  Paris

  Assertion(truth=True, proposition=OlderThan(left=Ada(), right=Nia()))
  yep, ada's older than nia

  Acknowledged(proposition=Age(subject=Ada(), value=30))
  got it, wrote that down

  Unknown(about=Age(subject=Greg()))
  no idea how old greg is

  Ask(of=User(), about=Age(subject=User()))
  how old are you?

  Ask(of=User(), about=Location(subject=User()))
  where do you live?'''


_SETTLING = '''You are given a multiple choice question and its options.
Reply with exactly one letter and nothing else: no prose, no punctuation, no
explanation.

Always answer. There is no option for not knowing: pick the one most likely to
be right, even when you are unsure. A letter that turns out wrong is worth
more than no letter at all.'''


class Seat:
    """Two models on a short leash: a fast pair of ears, a slower teacher.

    `hear` has to feel instant, so it stays on a small model. `define` is
    the one that makes Soup permanently smarter, so it may sit in a larger
    chair. `answer` and property-name guesses use the teacher too. If the
    teacher is missing, the ears fill in; Soup does not go deaf over a
    download.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        teacher: str = DEFAULT_TEACHER,
        key: Optional[str] = None,
        timeout: float = 30.0,
        embed: bool = False,
        weights: Optional[str] = None,
        teacher_weights: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.teacher = teacher or model
        self.key = key
        self.timeout = timeout
        # In-process llama.cpp, lazy. Tests construct a Seat and stub `_ask`;
        # they must not mmap three gigabytes on import.
        self.embed = embed
        self.weights = weights
        self.teacher_weights = teacher_weights
        self._engines: dict = {}
        self._gone: set = set()
        self.calls = 0
        self.failures = 0
        # One dead connection is enough. Nobody wants every turn to wait for
        # the same refused socket.
        self.available = True
        self.last_error: str = ""
        self.last_guess = None
        # The concept vocabulary, rendered once. It changes slowly and the
        # prompt is sent on every utterance, so rebuilding it per turn would
        # be waste.
        self.brief: str = ""

    @property
    def _local(self):
        """The ears' in-process engine, if it has been loaded."""
        return self._engines.get(self.model)

    def learn_vocabulary(self, knowledge) -> None:
        self.brief = vocabulary_brief(knowledge)

    def hear(
        self,
        utterance: str,
        vocabulary: Optional[List[str]] = None,
        context: str = "",
    ) -> Optional[Expr]:
        if not self.available:
            return None
        prompt = _SYSTEM + "\n\n" + _CONVENTIONS
        if self.brief:
            prompt += "\n\n" + self.brief
        elif vocabulary:
            prompt += "\n\nCONCEPTS ALREADY KNOWN (prefer these when they fit):\n"
            prompt += ", ".join(sorted(vocabulary)[:220])
        user_msg = utterance
        if context:
            user_msg = context + "\n\n" + utterance
        heard = self._read_hear(
            self._ask(prompt, user_msg, model=self.model),
            utterance,
            vocabulary or [],
        )
        if heard is not None and _packed_clause(heard):
            retried = self._read_hear(
                self._ask(prompt, user_msg + "\n\n" + _CLAUSE_HINT, model=self.model),
                utterance,
                vocabulary or [],
            )
            if retried is not None and not _packed_clause(retried):
                return retried
            self.last_error = self.last_error or "packed a clause into one name"
        return heard

    def _read_hear(
        self, reply: Optional[str], utterance: str, vocabulary: List[str]
    ) -> Optional[Expr]:
        if reply is None:
            return None
        try:
            heard = parse(_clean(reply))
        except (ParseError, ValueError, IndexError):
            self.last_error = "unparseable: %s" % reply[:120]
            self.last_guess = None
            return None
        self.last_guess = heard
        swapped = _swapped_person(heard, utterance, vocabulary)
        if swapped:
            # "who is albert einstein" as Who(subject=Alice()) put the wrong
            # person in the subject slot. A paraphrase like Developer() for
            # "I" is extra structure, not a swap, and has to get through or
            # the rest of the sentence dies with it.
            self.last_error = "invented %s, not in the utterance" % ", ".join(swapped)
            return None
        extras = _unfaithful(heard, utterance, vocabulary)
        if extras:
            self.last_error = "paraphrased %s" % ", ".join(extras)
        else:
            self.last_error = ""
        return heard

    @property
    def native(self) -> bool:
        """Ollama's own endpoint, as opposed to the OpenAI-shaped one."""
        return self.url.rstrip("/").endswith("/api/chat")

    def _payload(self, system: str, utterance: str, model: Optional[str] = None, max_tokens: int = 300) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterance},
        ]
        name = model or self.model
        if self.native:
            return {
                "model": name,
                "messages": messages,
                "stream": False,
                # One line of syntax needs no deliberation. Only Ollama's
                # native API actually honours this; the compatibility layer
                # quietly ignores it and burns the budget on reasoning.
                "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            }
        return {
            "model": name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterance},
        ]
        name = model or self.model
        if self.native:
            return {
                "model": name,
                "messages": messages,
                "stream": False,
                # One line of syntax needs no deliberation. Only Ollama's
                # native API actually honours this; the compatibility layer
                # quietly ignores it and burns the budget on reasoning.
                "think": False,
                "options": {"temperature": 0, "num_predict": 300},
            }
        return {
            "model": name,
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
        reply = self._ask(_ANSWERING, expression, model=self.teacher)
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

    def define(
        self, signature: str, vocabulary: List[str], context: Optional[str] = None
    ) -> Optional[Expr]:
        """Ask for a concept's meaning, not its value.

        The third thing a seat can be asked, and the most useful one. `hear`
        buys a single sentence and `answer` buys a single fact, but a
        definition is bought once and kept: it goes through the Teacher like
        any human answer, becomes an ordinary rule, and every later sentence
        using the concept is then resolved by Soup itself, offline and for
        free. Being told what "military time" means is worth more than being
        told what time it is.

        `context` is where the word actually turned up. A signature on its
        own strips the sentence away, and some words need it: asked what
        `Fifth(of)` means with nothing else to go on, a model says a fifth
        of it, which is a fine reading of the word and the wrong one for
        the fifth sister in the room.
        """
        if not self.available:
            return None
        prompt = _DEFINING
        if self.brief:
            prompt += "\n\n" + self.brief
        elif vocabulary:
            prompt += "\n\nKNOWN CONCEPTS:\n" + ", ".join(sorted(vocabulary)[:220])
        asked = "unknown  %s\nmeans" % signature
        if context and context != signature:
            asked = "in       %s\n%s" % (context, asked)
        reply = self._ask(prompt, asked, model=self.teacher)
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
        if _shouted_arguments(meaning):
            # `RealizedBy(Subject=subject)`. Argument names are lowercase,
            # and a rule carrying `Subject=` matches nothing for the rest of
            # time. Worth failing on rather than filing: a reply that got
            # the syntax wrong has usually got the meaning wrong too.
            self.last_error = "capitalised argument names: %s" % reply[:120]
            return None
        return meaning

    def speak(self, expression: str, style: str = "") -> Optional[str]:
        """Say a finished answer out loud, in English.

        Turning a concept expression into a sentence is a translation job, and
        a model is better at it in every direction that matters: it is less
        brittle than a renderer full of special cases, it can be asked for a
        different register or a different language, and it does not need a new
        branch every time a concept shape is invented.

        Soup's answer is the concept expression. This only chooses words for
        it, so it is given nothing to reason about and told to add nothing. A
        mouth that decides facts is a second brain disagreeing with the first.
        """
        if not self.available or not expression:
            return None
        prompt = _SPEAKING if not style else "%s\n\nSTYLE\n%s" % (_SPEAKING, style)
        reply = self._ask(prompt, expression, model=self.model)
        if reply is None:
            return None
        said = _clean(reply).strip().strip('"').strip()
        if not said or "\n" in said.strip():
            said = said.splitlines()[0].strip() if said else ""
        return said or None

    def settle(self, question: str, options: List[tuple]) -> Optional[str]:
        """Which of these, when everything Soup knows has come up empty.

        The fourth and cheapest thing a seat can be asked, and deliberately
        the last one tried. `define` buys a rule worth keeping and `answer`
        buys a fact worth filing; this buys one letter and teaches nothing.
        It exists because the alternative is not a better answer, it is no
        answer, and a shrug is the only outcome that is never any use.

        Handed the options, a model that could not define the concept can
        still usually recognise the right one, so a question Soup could not
        work out gets answered rather than abandoned. What comes back is
        recorded as a guess from the chair, not as something Soup worked out,
        so nobody is misled about which is which.
        """
        if not self.available or not options:
            return None
        lines = ["%s. %s" % (letter, text) for letter, text in options if text]
        if not lines:
            return None
        asked = "%s\n\n%s" % (question.strip(), "\n".join(lines))
        reply = self._ask(_SETTLING, asked, model=self.teacher)
        if reply is None:
            return None
        letters = {letter.upper() for letter, _text in options}
        found = re.findall(r"\b([A-Za-z])\b", _clean(reply))
        for candidate in found:
            if candidate.upper() in letters:
                return candidate.upper()
        return None

    def property_names(
        self,
        concept: str,
        subject: Optional[str] = None,
        specifically: Optional[str] = None,
    ) -> List[str]:
        """What a record keeper is likely to call this property.

        The narrowest possible use of a model, and the safest. It is not
        asked for the fact, it is asked for a word, and the word still has
        to match a real Wikidata property exactly before anything is
        believed. A bad suggestion finds nothing and we learn nothing; it
        cannot turn into a wrong fact with a citation stapled to it.

        Worth having because the vocabulary gap is real and one-sided:
        Wikidata files chess under "minimum number of players", nobody asks
        a question that way, and `Players` matches no label at all.
        """
        if not self.available:
            return []
        asked = "concept  %s%s%s\nnames" % (
            concept,
            ", of %s" % subject if subject else "",
            ", specifically the %s" % specifically if specifically else "",
        )
        reply = self._ask(_NAMING_PROPERTY, asked, model=self.teacher)
        if reply is None:
            return []
        out: List[str] = []
        for line in _clean(reply).splitlines():
            name = line.strip().strip("-*\u2022").strip().strip('"')
            if not name or name.lower() in ("none", "names"):
                continue
            if name.lower() not in [o.lower() for o in out]:
                out.append(name)
        return out[:3]

    def _engine(self, name: Optional[str] = None):
        """An in-process model for this name, or None to use HTTP.

        Failed loads are remembered as False so a missing llama-cpp-python
        does not get re-imported on every sentence. None here means try HTTP
        next. A missing teacher must not sit the ears in that chair before
        ollama has had a go; 404 on HTTP is when the ears fill in.
        """
        if not self.embed:
            return None
        name = name or self.model
        if name in self._engines:
            cached = self._engines[name]
            if cached is False:
                return None
            return cached
        from .local import load

        weights = self.weights if name == self.model else self.teacher_weights
        try:
            loaded = load(name, weights)
        except Exception as exc:
            self.last_error = "could not load %s: %s" % (name, exc)
            self._engines[name] = False
            return None
        if loaded is None:
            self._engines[name] = False
            return None
        self._engines[name] = loaded
        return loaded

    def _ask(self, system: str, utterance: str, model: Optional[str] = None, max_tokens: int = 300) -> Optional[str]:
        model = model or self.model
        if model in self._gone:
            model = self.model
        engine = self._engine(model)
        if engine is not None:
            self.calls += 1
            from .local import chat

            try:
                return chat(engine, system, utterance, max_tokens=max_tokens)
            except Exception as exc:
                self.failures += 1
                self.last_error = str(exc)
                return None
        body = json.dumps(self._payload(system, utterance, model=model, max_tokens=max_tokens)).encode("utf-8")
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
        except urllib.error.HTTPError as exc:
            self.failures += 1
            if exc.code == 404 and model != self.model:
                self._gone.add(model)
                self.last_error = "no %s, using %s" % (model, self.model)
                return self._ask(system, utterance, self.model, max_tokens=max_tokens)
            self.available = False
            self.last_error = "%s (%s)" % (exc, self.url)
            return None
        except (urllib.error.URLError, OSError) as exc:
            self.failures += 1
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                # Slow, not absent. Worth another go on the next sentence.
                self.last_error = "timed out after %gs" % self.timeout
                return None
            if model != self.model:
                self._gone.add(model)
                self.last_error = "no %s (%s), using %s" % (model, exc, self.model)
                return self._ask(system, utterance, self.model, max_tokens=max_tokens)
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
        "Greeting", "Farewell", "Laugh", "Thanks", "Affirm", "Deny",
        "Unintelligible", "Opinion", "Acknowledged",
        "User", "Assistant",
    ]
)


def _camel_words(name: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)]


# A named person is two or three humps. A glued-together clause is a sentence.
_PACKED_HUMPS = 6

_CLAUSE_HINT = (
    "A clause is nested concepts. Never Concept(name=...). Never a CamelCase "
    "sentence. Identity is only for a named person or thing."
)


def _packed_name(name: str) -> bool:
    return len(_camel_words(name)) >= _PACKED_HUMPS


def _packed_clause(expr: Expr) -> bool:
    """The whole question smashed into one name, which nothing can realize."""
    for node in walk(expr):
        if not isinstance(node, Call):
            continue
        if node.concept == "Concept":
            return True
        if not node.args and _packed_name(node.concept):
            return True
        if _packed_name(node.concept) and node.concept not in _STRUCTURAL:
            return True
    return False


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

    So heads may be invented freely. Extra leaves are paraphrases and are
    allowed through. What still dies is a *swap*: asking about Einstein and
    hearing Alice in the subject slot.
    """
    said = set(re.findall(r"[a-z0-9]+", utterance.lower()))
    known = set(vocabulary) | _STRUCTURAL
    invented = []
    for name in _leaves(expr):
        if name in known or name in invented:
            continue
        words = _camel_words(name)
        smashed = "".join(words)
        # "typescript" is one word. TypeScript is two. They are the same name.
        if smashed in said:
            continue
        if all(w in _FUNCTION_WORDS or _echoes(w, said) for w in words):
            continue
        invented.append(name)
    return invented


_PERSON_ASK = frozenset(
    ["Who", "Identity", "Age", "DateOfBirth", "Name", "Location"]
)


def _swapped_person(expr: Expr, utterance: str, vocabulary: List[str]) -> List[str]:
    """Alice in a sentence about Einstein. Extra leaves are not this."""
    invented = set(_unfaithful(expr, utterance, vocabulary))
    if not invented:
        return []
    swapped = []
    for node in walk(expr):
        if not isinstance(node, Call) or node.concept not in _PERSON_ASK:
            continue
        subject = node.first("subject", "of", "who")
        if (
            isinstance(subject, Call)
            and not subject.args
            and subject.concept in invented
        ):
            swapped.append(subject.concept)
    return swapped


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


_NAMING_PROPERTY = '''Wikidata keeps facts under properties with fixed English
names. You are given a concept Soup could not find a property for. Say what
that property is most likely called on Wikidata.

Reply with up to three names, one per line, best guess first. Nothing else:
no numbering, no prose, no P-numbers, no explanation.

Use Wikidata's own wording. It is lowercase, and it is often longer and more
specific than the word you were handed. If nothing plausible exists, reply
with the single word: none

EXAMPLES
  concept  Players, of Chess
  names    minimum number of players
  maximum number of players
  number of players

  concept  Players, of Chess, specifically the minimum
  names    minimum number of players
  maximum number of players
  number of players

  concept  Height, of EiffelTower
  names    height

  concept  Director, of Jaws
  names    director

  concept  Vibe, of Tuesday
  names    none
'''


def _shouted_arguments(expr: Expr) -> bool:
    """An argument name starting with a capital, which is a concept name."""
    return any(
        isinstance(n, Call) and any(a.name and a.name[:1].isupper() for a in n.args)
        for n in walk(expr)
    )


def vocabulary_brief(knowledge, per_kind: int = 120) -> str:
    """The concepts soup already has, grouped so a model can reuse them.

    This is the difference between a model that invents `Times` and `HowOld`
    alongside the `Multiply` and `Age` we already own, and one that lands on
    the vocabulary the realizer can actually do something with. A flat
    alphabetical dump does not achieve that; showing what sort of thing each
    concept is does. Short glosses also tell the model what each name means,
    so it can choose tools and concepts by purpose rather than by name alone.
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
        lines.extend(
            "  %s: %s" % (name, knowledge.concepts[name].gloss)
            if knowledge.concepts[name].gloss
            else "  " + name
            for name in names[:per_kind]
        )
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
    # A bare </think> means the model opened thinking before we saw it.
    text = text.replace("</think>", "").strip()
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
    teacher = os.environ.get("SOUP_TEACHER_MODEL")
    weights = os.environ.get("SOUP_LLM_WEIGHTS")
    if not url and not model and not weights:
        return None
    return Seat(
        url=url or DEFAULT_URL,
        model=model or DEFAULT_MODEL,
        teacher=teacher or DEFAULT_TEACHER,
        key=os.environ.get("SOUP_LLM_KEY"),
        timeout=float(os.environ.get("SOUP_LLM_TIMEOUT", "30")),
        # A URL means they already have a server. Weights or a model name
        # without a URL means load them ourselves.
        embed=url is None,
        weights=weights,
    )
