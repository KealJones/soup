"""Realization: concepts in, concepts out.

A realization is anything that turns one concept expression into another. It
might be Python code, a taught rewrite rule, a lookup in what we believe, or a
composition of all three. Execution is just the case where a realization
happens to touch the outside world.

When nothing can realize a concept we do not throw. We record a `Gap` and keep
going, because the half-resolved expression is exactly the diagnostic surface
the Teacher needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .expr import Arg, Call, Expr, Lit, Seq, Var, is_name, render, substitute
from .knowledge import Evidence, Knowledge

__all__ = [
    "Gap",
    "Result",
    "Realizer",
    "Context",
    "native",
    "only_you_know",
    "NATIVES",
    "SPECIAL_FORMS",
]

MAX_DEPTH = 48

# Concepts that would rather see an `Unknown` argument than be short-circuited
# by it. Everything else passes the ignorance straight up the tree, so the
# answer names the thing we actually failed to resolve.
UNKNOWN_TOLERANT = {
    "Unknown",
    "Ask",
    "If",
    "Conditional",
    "Choose",
    "Select",
    "Pick",
    "Answer",
    "Assertion",
    "Explain",
    "Teach",
    "Remember",
    "Recall",
    "Query",
    "Say",
    "Request",
    "Acknowledged",
    "Hear",
    "Speak",
    "Meaning",
    "Turn",
    "Think",
    "Consult",
    "Conversation",
    "Subagent",
    "Turn",
}

# This computer, not the world. Asking a model what time it is, or filing
# the answer as a fact, is how "9:09 pm in military time" became a memory.
_CLOCK = frozenset(
    {
        "Time",
        "Now",
        "Date",
        "Today",
        "Day",
        "Weekday",
        "Year",
        "Timezone",
        "TwentyFourHourTime",
    }
)


@dataclass
class Gap:
    """A concept we could not resolve, and the shape it was used in."""

    concept: str
    expr: Expr
    reason: str = "no realization"

    @property
    def arity(self) -> int:
        return len(self.expr.args) if isinstance(self.expr, Call) else 0

    def signature(self) -> str:
        if not isinstance(self.expr, Call) or not self.expr.args:
            return "%s()" % self.concept
        parts = []
        for a in self.expr.args:
            parts.append(a.name if a.name else "_")
        return "%s(%s)" % (self.concept, ", ".join(parts))


@dataclass
class Result:
    value: Expr
    gaps: List[Gap] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    # Definitions worked out on the spot and kept. Worth telling the person
    # about: it is the difference between Soup answering them and Soup
    # getting permanently better at their kind of question.
    learned: List[str] = field(default_factory=list)
    heard: Optional[object] = None
    said: Optional[str] = None
    # Answered by recognition in the chair rather than worked out here. Not a
    # failure, but not the same thing as knowing, and worth being able to
    # count separately.
    guessed: bool = False

    @property
    def resolved(self) -> bool:
        return not self.gaps

    def __str__(self) -> str:
        return render(self.value, multiline=False)


NativeFn = Callable[["Context", Call], Optional[Expr]]
NATIVES: Dict[str, NativeFn] = {}
SPECIAL_FORMS: Dict[str, NativeFn] = {}


def native(*names: str, special: bool = False) -> Callable[[NativeFn], NativeFn]:
    """Register a Python realization for one or more concepts.

    `special=True` means "do not realize my arguments first" - needed for
    binding forms, conditionals and anything that treats an argument as a
    pattern rather than a value.
    """

    def deco(fn: NativeFn) -> NativeFn:
        table = SPECIAL_FORMS if special else NATIVES
        for n in names:
            table[n] = fn
        return fn

    return deco


class Context:
    """What a native realization gets to play with."""

    def __init__(self, realizer: "Realizer", env: Dict[str, Expr], depth: int, result: Result):
        self.realizer = realizer
        self.env = env
        self.depth = depth
        self.result = result

    @property
    def knowledge(self) -> Knowledge:
        return self.realizer.knowledge

    def realize(self, expr: Expr) -> Expr:
        return self.realizer._realize(expr, self.env, self.depth + 1, self.result)

    def note(self, message: str) -> None:
        self.result.trace.append(message)

    def effect(self, message: str) -> None:
        self.result.effects.append(message)

    def gap(self, concept: str, expr: Expr, reason: str = "no realization") -> None:
        self.realizer._record_gap(self.result, Gap(concept, expr, reason))

    @property
    def thinking(self) -> bool:
        return bool(getattr(self.realizer, "_thinking", False))


# Argument names that hold a whole thought rather than a thing.
_PROPOSITIONAL = frozenset(["proposition", "about", "pattern", "action", "to", "given"])

# Do not realize these. `given` is a scene, not a to-do list: realizing
# Play(Kate, Chess) as an operation is how "list the activities" turned
# into "teach me Do".
_QUOTED = frozenset(["given"])




class Realizer:
    """Walks a concept expression and resolves as much meaning as it can."""

    def __init__(
        self,
        knowledge: Knowledge,
        seat=None,
        sources=None,
        discourse=None,
        agents=None,
        think_budget: int = 3,
    ) -> None:
        self.knowledge = knowledge
        # Places to ask about facts nothing here could resolve, in order of
        # trust. These are generic answer sources; Wikidata, for example,
        # builds its requests and filters results from ordinary concepts.
        self.sources = list(sources or [])
        self.seat = seat
        self.discourse = discourse
        self.agents = agents or {}
        self.think_budget = think_budget
        self.lesson = None
        self.last_result: Optional[Result] = None
        self._outside_budget = 0
        self._thinking = False
        self._think_depth = 0
        # The English of the turn in progress, for the last resort that needs
        # the sentence rather than the concept tree.
        self._utterance = ""

    def realize(
        self,
        expr: Expr,
        env: Optional[Dict[str, Expr]] = None,
        thinking: bool = False,
    ) -> Result:
        result = Result(value=expr)
        environment: Dict[str, Expr] = dict(env or {})
        # One utterance should not turn into a dozen round trips.
        self._outside_budget = 2
        self._thinking = thinking
        self._think_depth = 0
        # Learning is worth more than answering, so it gets a budget of its
        # own rather than competing with one.
        self._learn_budget = 2
        # Asking outright, once everything we know has failed. Its own budget
        # too: it must not be spent by the ordinary lookups earlier in a pass,
        # because this is the one that stops us shrugging.
        self._resort_budget = 3
        # Concepts the seat was asked to define and had nothing for. Worth
        # remembering for the length of a pass: putting the identical
        # signature to the identical model twice buys the identical nothing.
        self._declined = set()
        # A model is a thing to ask, not a thing to obey. It gets consulted
        # about questions and stays out of requests and assertions, which
        # are ours to carry out or to write down.
        self._asking = isinstance(expr, Call) and expr.concept in ("Question", "Query")
        result.value = self._realize(expr, environment, 0, result)
        return result

    # -- core --------------------------------------------------------------
    def _realize(self, expr: Expr, env: Dict[str, Expr], depth: int, result: Result) -> Expr:
        if depth > MAX_DEPTH:
            result.trace.append("stopped: expression kept unfolding past depth %d" % MAX_DEPTH)
            return expr

        if isinstance(expr, Lit):
            return expr

        if isinstance(expr, Var):
            bound = env.get(expr.name)
            if bound is None:
                return expr
            return self._realize(bound, env, depth + 1, result) if bound != expr else expr

        if isinstance(expr, Seq):
            return Seq(tuple(self._realize(i, env, depth + 1, result) for i in expr.items))

        assert isinstance(expr, Call)
        ctx = Context(self, env, depth, result)

        special = SPECIAL_FORMS.get(expr.concept)
        if special is not None:
            out = special(ctx, expr)
            return out if out is not None else expr

        evaluated = Call(
            expr.concept,
            tuple(
                Arg(
                    a.name,
                    a.value
                    if a.name in _QUOTED
                    else self._realize(a.value, env, depth + 1, result),
                )
                for a in expr.args
            ),
        )

        cd = self.knowledge.concept(evaluated.concept)
        known_concept = cd is not None and cd.kind != "concept"
        if known_concept and evaluated.concept not in UNKNOWN_TOLERANT:
            # Pass ignorance upward so the reply names the thing we actually
            # lack. Concepts we do not know keep going: their own gap is the
            # more useful thing to report.
            for a in evaluated.args:
                if isinstance(a.value, Call) and a.value.concept == "Unknown":
                    return a.value

        # A word can mean several things, and the argument names are how the
        # sentence said which. `Power(voltage, current)` is not exponentiation
        # however many numbers it has in it, so a meaning taught for exactly
        # this shape outranks the native reading of the same word.
        by_name = self._apply_rules(evaluated, result, loose=False)
        if by_name is not None and not _contains(by_name, evaluated):
            result.trace.append(
                "%s -> %s  (taught, this shape)"
                % (render(evaluated, False), render(by_name, False))
            )
            return self._realize(by_name, env, depth + 1, result)

        fn = NATIVES.get(evaluated.concept)
        if fn is not None:
            out = fn(ctx, evaluated)
            if out is not None:
                if out != evaluated:
                    result.trace.append(
                        "%s -> %s" % (render(evaluated, False), render(out, False))
                    )
                    return self._realize(out, env, depth + 1, result)
                return out

        rewritten = self._apply_rules(evaluated, result)
        if rewritten is not None and _contains(rewritten, evaluated):
            # A rule that hands back what it was given will hand it back
            # forever. Definitions arrive from people and from models and
            # some of them are circular; this is where that stops being
            # fatal rather than merely wrong.
            result.trace.append(
                "ignored a rule for %s: it rewrites to itself" % evaluated.concept
            )
            rewritten = None
        if rewritten is not None:
            result.trace.append(
                "%s -> %s  (taught)" % (render(evaluated, False), render(rewritten, False))
            )
            return self._realize(rewritten, env, depth + 1, result)

        from_facts = self._from_facts(evaluated)
        if from_facts is not None:
            result.trace.append(
                "%s -> %s  (known)" % (render(evaluated, False), render(from_facts, False))
            )
            return self._realize(from_facts, env, depth + 1, result)

        inherited = self._inherited_rule(evaluated, result)
        if inherited is not None:
            return self._realize(inherited, env, depth + 1, result)

        spread = self._distribute(evaluated, env, depth, result)
        if spread is not None:
            return spread

        # Nothing resolved it, so decide what kind of not-knowing this is.
        if is_name(evaluated):
            # A name, bare or with adjectives on it. An entity, a quality, an
            # operation referred to as a value. Names stand for themselves;
            # you cannot fail to evaluate "Greg", and "the lazy dog" is no
            # more a failed computation than "dog" is.
            return evaluated

        if self._is_value(evaluated):
            return evaluated

        if not known_concept and not self._looks_like_fact(evaluated):
            # A missing fact is not a missing word. Players(subject=Chess())
            # should hit the record before the Teacher asks you to define
            # Players as a verb.
            worked_out = self._learn_concept(evaluated, env, depth, result)
            if worked_out is not None:
                return worked_out
            settled = self._classify_subject(evaluated, env, depth, result)
            if settled is not None:
                return settled

        asked = self._from_outside(evaluated, result)
        if asked is not None:
            return asked

        if known_concept and evaluated.concept in _CLOCK:
            # Known word, unknown *shape*. Time(subject, format=Military())
            # is not a missing fact about France; it is a method we have not
            # been taught yet. Ask what it means, keep the rule, reuse it.
            worked_out = self._learn_concept(evaluated, env, depth, result)
            if worked_out is not None:
                return worked_out

        if known_concept:
            # Power(voltage, current) is not 120**2. The native declined this
            # shape; that is a method to learn, not a missing fact.
            if cd is not None and cd.kind == "operation" and evaluated.concept in NATIVES:
                worked_out = self._learn_concept(evaluated, env, depth, result)
                if worked_out is not None:
                    return worked_out
            asked = self._last_resort(evaluated, result)
            if asked is not None:
                return asked
            return Call("Unknown", (Arg("about", evaluated),))

        # Something is being done to arguments and we have no idea what.
        # That is a hole in the language, and the Teacher can fill exactly it.
        self._record_gap(result, Gap(evaluated.concept, evaluated))
        return evaluated

    # -- resolution strategies --------------------------------------------
    def _apply_rules(self, expr: Call, result: Result, loose: bool = True) -> Optional[Expr]:
        for rule in self.knowledge.rules_for(expr.concept):
            out = rule.apply(expr, loose=loose)
            if out is not None:
                return out
        return None

    def _inherited_rule(self, expr: Call, result: Result) -> Optional[Expr]:
        """If `Sprint IsA Run`, try Run's realizations for a Sprint."""
        for parent in self.knowledge.ancestors(expr.concept):
            for rule in self.knowledge.rules_for(parent):
                out = rule.apply(Call(parent, expr.args))
                if out is not None:
                    result.trace.append(
                        "%s is a %s, so: %s" % (expr.concept, parent, render(out, False))
                    )
                    return out
            fn = NATIVES.get(parent)
            if fn is not None:
                return Call(parent, expr.args)
        return None

    def _distribute(
        self, expr: Call, env: Dict[str, Expr], depth: int, result: Result
    ) -> Optional[Expr]:
        """An operation handed a collection where it wanted one thing.

        "add 10 to [1, 2, 3]" means do it to each of them. Only for operations
        we actually know, and only when exactly one argument is a collection,
        so this never quietly papers over a real mismatch.
        """
        cd = self.knowledge.concept(expr.concept)
        if cd is None or cd.kind != "operation" or len(expr.args) < 2:
            return None
        positions = [i for i, a in enumerate(expr.args) if isinstance(a.value, Seq)]
        if len(positions) != 1:
            return None
        index = positions[0]
        collection = expr.args[index].value
        assert isinstance(collection, Seq)
        spread: List[Expr] = []
        for item in collection.items:
            args = list(expr.args)
            args[index] = Arg(args[index].name, item)
            spread.append(self._realize(Call(expr.concept, tuple(args)), env, depth + 1, result))
        result.trace.append(
            "%s applied to each of %d" % (expr.concept, len(collection.items))
        )
        return Seq(tuple(spread))

    def _from_facts(self, expr: Call) -> Optional[Expr]:
        """Answer straight out of what we believe.

        `Age(subject=Alice)` finds the fact `Age(subject=Alice, value=30)`.
        `Has(subject=Greg, object=Car)` finds itself and answers True.
        """
        if not expr.has("value"):
            probe = Call(expr.concept, expr.args + (Arg("value", Var("_v")),))
            hits = self.knowledge.query(probe)
            for fact, bindings in hits:
                if fact.truth and "_v" in bindings:
                    return bindings["_v"]
            subject = _subject_of(expr)
            if subject is not None:
                # Same question, different slot name. What we wrote down as
                # `Players(subject=Chess(), value=2)` is the answer to
                # `Players(inGame=Chess())`, and the ears get to invent a
                # fresh preposition every time they read a sentence.
                probe = Call(
                    expr.concept, (Arg("subject", subject), Arg("value", Var("_v")))
                )
                for fact, bindings in self.knowledge.query(probe):
                    if fact.truth and "_v" in bindings:
                        return bindings["_v"]

        hits = self.knowledge.query(expr)
        for fact, _ in hits:
            if _same_shape(fact.proposition, expr):
                return Lit(bool(fact.truth))
        return None

    def _is_value(self, expr: Call) -> bool:
        """Entities, qualities and other nouns realize to themselves."""
        cd = self.knowledge.concept(expr.concept)
        if cd is None:
            return not expr.args
        return cd.kind in ("entity", "thing", "quality", "value")

    def _learn_concept(
        self, expr: Call, env: Dict[str, Expr], depth: int, result: Result
    ) -> Optional[Expr]:
        """Work out what a concept means, rather than what its answer is.

        Deliberately ahead of asking outright, because the two are not the
        same purchase. An answer is good once. A definition is good forever,
        and turns every later sentence built on the concept back into
        ordinary local resolution, offline and free.

        Asked to quintuple ten, a model hands back a number, and a small one
        will hand back the wrong number. Asked what quintupling *is*, the
        same model says Multiply(x, 5), which is both right and permanent.
        Prefer the question that gets easier to answer every time.

        The definition is not trusted for having come from a model. It goes
        through the same Teacher a human answer would, which throws it out
        unless it is built from concepts we already have, and files what
        survives as an ordinary rule anyone can inspect or contradict.
        """
        if self.seat is None or not getattr(self.seat, "available", True):
            return None
        if self._learn_budget <= 0:
            return None
        self._learn_budget -= 1
        # Imported here because the Teacher is built on top of us.
        from .teacher import Teacher

        teacher = Teacher(self.knowledge)
        lesson = teacher.ask(Gap(expr.concept, expr))
        meaning = self.seat.define(
            lesson.signature(), list(self.knowledge.concepts), context=render(expr, False)
        )
        if meaning is None:
            self._declined.add(expr.concept)
            return None
        if teacher.learn(lesson, meaning) is None:
            self._declined.add(expr.concept)
            return None
        rewritten = self._apply_rules(expr, result)
        if rewritten is None:
            # A definition we cannot then apply taught us nothing usable.
            return None
        definition = "%s := %s" % (lesson.signature(), render(meaning, False))
        result.learned.append(definition)
        result.trace.append("%s  (worked it out)" % definition)
        return self._realize(rewritten, env, depth + 1, result)

    def _classify_subject(
        self, expr: Call, env: Dict[str, Expr], depth: int, result: Result
    ) -> Optional[Expr]:
        """Before asking what X of a thing is, find out what the thing is.

        `Players(inGame=Chess())` is an unknown word applied to a name we
        hold nothing about. Asked the question straight, a model answers in
        prose - "white and black pieces on a board" - which gets filed as
        the value and poisons every later reading of it. Asked what chess
        *is*, the same model says `Game(players=2)`, which is structure, is
        reusable, and answers the question on the way past.

        Only for words we do not know. `Age(subject=Alice())` is already a
        question we understand and goes straight to the source.
        """
        subject = _subject_of(expr)
        if subject is None or self._learn_budget <= 0:
            return None
        assert isinstance(subject, Call)
        if self.knowledge.ancestors(subject.concept):
            return None
        self._learn_budget -= 1
        self._realize(Call("Kind", (Arg("of", subject),)), env, depth + 1, result)
        settled = self._from_facts(expr)
        if settled is None:
            return None
        result.trace.append(
            "%s -> %s  (once it knew what %s was)"
            % (render(expr, False), render(settled, False), subject.concept)
        )
        return self._realize(settled, env, depth + 1, result)

    def _from_outside(self, expr: Call, result: Result) -> Optional[Expr]:
        """Last resort: ask somewhere else, then write down what came back.

        Only reached once everything we actually know has failed, so the
        outside world fills in facts rather than doing arithmetic we can do
        ourselves. Sources are tried in order of how much they deserve to be
        believed: a record of a fact beats a guess at one, so Wikidata goes
        first and a model only sees what Wikidata had no answer for.

        Whatever comes back is kept as an ordinary fact, tagged with where it
        came from. So the same question is only asked once, `:facts` shows
        you which answers were ours and which were borrowed, and you can
        contradict any of them.
        """
        if self._outside_budget <= 0:
            return None
        if expr.concept in _CLOCK:
            return None
        if not self._looks_like_fact(expr):
            return None
        if not self._asking and not self._fact_shaped(expr):
            return None
        if any(arg.name in _PROPOSITIONAL for arg in expr.args):
            # A wrapper around the real question. Whatever it wraps has
            # already had its turn; asking again just asks worse.
            return None
        self._outside_budget -= 1
        for source, tag, note in self._sources():
            answer = source.answer(expr)
            if answer is None:
                continue
            if isinstance(answer, Call) and answer.concept in ("Unknown", "Nothing"):
                continue
            result.trace.append(
                "%s -> %s  (%s)" % (render(expr, False), render(answer, False), note)
            )
            if not expr.has("value"):
                self.knowledge.assert_fact(
                    Call(expr.concept, expr.args + (Arg("value", answer),)),
                    True,
                    Evidence(source=tag, confidence=0.9 if tag == "wikidata" else 0.6),
                )
            return answer
        return None

    def _last_resort(self, expr: Call, result: Result) -> Optional[Expr]:
        """Everything we know has failed. Ask anyway rather than shrug.

        `_from_outside` is choosy on purpose: it will not send a verb to a
        model that wants a fact, because a bad answer gets written down as
        one. This is the end of the line, where the alternative is not a
        better answer but no answer, and no answer is the one outcome that
        is never useful. An answer from outside is still tagged with where
        it came from, so a guess stays visibly a guess.

        The exception is a question only the person in front of us can
        settle. Nothing on the internet knows how old you are.
        """
        if _mentions_user(expr):
            return None
        if expr.concept in _CLOCK:
            return None
        if self._resort_budget <= 0:
            return None
        self._resort_budget -= 1
        for source, tag, note in self._sources():
            answer = source.answer(expr)
            if answer is None:
                continue
            if isinstance(answer, Call) and answer.concept in ("Unknown", "Nothing"):
                continue
            result.trace.append(
                "%s -> %s  (%s, last resort)"
                % (render(expr, False), render(answer, False), note)
            )
            if not expr.has("value"):
                self.knowledge.assert_fact(
                    Call(expr.concept, expr.args + (Arg("value", answer),)),
                    True,
                    Evidence(
                        source=tag,
                        confidence=0.8 if tag == "wikidata" else 0.4,
                        note="last resort",
                    ),
                )
            return answer
        return None

    def _fact_shaped(self, expr: Call) -> bool:
        """A missing *value* of an attribute, even inside a request.

        Requests must not ask the model to *do* the action. They may ask
        what a file's Source is, the way a question asks Alice's Age.
        """
        cd = self.knowledge.concept(expr.concept)
        if cd is not None and cd.kind == "attribute":
            return True
        if cd is not None and cd.kind == "operation":
            return False
        # Players(subject=Chess()) is a missing fact even before anyone has
        # told soup that Players is an attribute, and even when the question
        # arrived wrapped in Minimum so `_asking` is looking at the wrapper.
        return _subject_of(expr) is not None

    def _looks_like_fact(self, expr: Call) -> bool:
        """Should we ask the outside world, or the Teacher?

        Capital(subject=France()) is a missing fact. Choose(between=A, or=B)
        is a missing verb. Asking a model to 'answer' Choose restates the
        tree badly and then we try to teach the wreckage.
        """
        if expr.concept in _CLOCK:
            return False
        if _mentions_user(expr):
            return False
        if self._fact_shaped(expr) or expr.concept in ("Identity", "Name"):
            return True
        if _subject_of(expr) is not None:
            return True
        names = expr.arg_names()
        if "subject" in names or "of" in names:
            return not any(n in _DOING_ARGS for n in names)
        return False

    def _sources(self):
        for source in self.sources:
            if isinstance(source, tuple):
                values = list(source) + [None, None]
                resolver, tag, note = values[:3]
            else:
                resolver = source
                tag = getattr(source, "tag", None) or "outside"
                note = getattr(source, "note", None) or "looked it up"
            if resolver is not None and getattr(resolver, "available", True):
                yield resolver, tag, note
        if self.seat is not None and getattr(self.seat, "available", True):
            yield self.seat, "llm", "asked the model"

    def _record_gap(self, result: Result, gap: Gap) -> None:
        for existing in result.gaps:
            if existing.concept == gap.concept and existing.expr == gap.expr:
                return
        result.gaps.append(gap)


def _same_shape(a: Expr, b: Expr) -> bool:
    return isinstance(a, Call) and isinstance(b, Call) and len(a.args) == len(b.args)


def _mentions_user(expr: Expr) -> bool:
    from .expr import walk

    return any(isinstance(n, Call) and n.concept == "User" for n in walk(expr))


def only_you_know(about: Optional[Expr]) -> Optional[Call]:
    """A missing fact about the person in front of us, as a question for them.

    The one class of ignorance no amount of looking things up can fix: there
    is nothing on the internet about how old you are, and a model asked will
    invent a number. So it comes back as `Ask(of=User(), about=...)`, which
    is a concept like everything else here, and the words for it are chosen
    where all the other words are chosen.

    Deliberately about the shape rather than a list of blessed attributes. A
    table of Age, Name and LivesIn only ever answers for the three things
    somebody thought of; "can you drive" is equally yours to answer.
    """
    if not isinstance(about, Call) or not _mentions_user(about):
        return None
    return Call("Ask", (Arg("of", Call("User", ())), Arg("about", about)))


# Argument names that mean the call is an action being set up, not an
# attribute of something.
_DOING_ARGS = frozenset({"between", "or", "for", "action", "options", "reason"})

# The slots a one-argument fact is allowed to arrive in. `target` is how
# you aim a Make or a Delete, and looking that up as a property of a kettle
# is how a request got sent to the model.
_ABOUT_SLOTS = frozenset({"subject", "of", "in", "for", "about", "object", "inGame"})


def _subject_of(expr: Call) -> Optional[Expr]:
    """The one thing a call is about, whatever the ears called the slot.

    `Players(inGame=Chess())`, `Players(in=Chess())` and
    `Players(subject=Chess())` are the same question asked three ways. The
    slot name is invented fresh by whatever read the sentence, so hanging
    "is this a fact we could look up" on it meant one improvised
    preposition sent a perfectly answerable question to the human instead.
    An attribute of one named thing is the commonest shape there is.
    """
    if len(expr.args) != 1:
        return None
    arg = expr.args[0]
    if arg.name and arg.name not in _ABOUT_SLOTS:
        return None
    return arg.value if isinstance(arg.value, Call) and is_name(arg.value) else None


def _contains(haystack: Expr, needle: Expr) -> bool:
    from .expr import walk

    return any(node == needle for node in walk(haystack))
