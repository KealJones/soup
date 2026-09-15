"""Tests for soup. Plain unittest, no dependencies.

    python3 -m unittest discover -s tests -v

English goes through a scripted seat. The tests never touch a network. Soup's
own notation (`Multiply(6, 4)`, `Vibe(x) := Count(x)`) is parsed directly,
because that is this system's language, not natural language understanding.
"""

from __future__ import annotations

import json
import io
import os
import random
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Seat, Session, fresh_knowledge, parse, parse_definition, render
from soup.cli import _command
from soup.discourse import Discourse
from soup.ears import Ears
from soup.expr import Arg, Call, Lit, Seq, Var, is_name, match, substitute
from soup.knowledge import SYMMETRIC, TAXONOMIC, TRANSITIVE, Knowledge
from soup.lookup import Wikidata, _subject_text
from soup.local import find_weights, mlx_id
from soup.realize import Gap, Realizer
from soup.seat import vocabulary_brief
from soup.teacher import Lesson, Teacher


def reply(session: Session, text: str) -> str:
    return session.respond(text).said


def meaning(session: Session, text: str) -> str:
    heard = session.respond(text).heard
    assert heard is not None
    return render(heard.expr, multiline=False)


def fresh(seed: int = 1, llm=None, sources=()) -> Session:
    # Most unit tests exercise Soup without external sources. Production
    # Sessions enable Wikidata by default; lookup tests opt in explicitly.
    return Session(memory_path=None, seed=seed, llm=llm, sources=list(sources or []))


def talking(heard, answers=None, definitions=None, seed: int = 1, sources=None) -> Session:
    """A session whose ears recites a script instead of calling a model."""
    return fresh(seed=seed, llm=Script(heard, answers, definitions), sources=sources)


class Script:
    """A stand-in for a model, so the tests never touch a network.

    Three jobs, same as the real seat: hear a sentence, define a concept,
    answer a fact. Anything not in the script is a refusal, which is how a
    real model that does not know also comes back.
    """

    available = True

    def __init__(self, heard=None, answers=None, definitions=None) -> None:
        self.heard = {_key(k): v for k, v in (heard or {}).items()}
        self.answers = answers or {}
        self.definitions = definitions or {}
        self.asked = []
        self.defined = []
        self.context = []
        self.last_error = ""

    def hear(self, utterance, vocabulary=None):
        text = self.heard.get(_key(utterance))
        if text is None:
            self.last_error = "unscripted: %s" % utterance
            return None
        return parse(text) if isinstance(text, str) else text

    def answer(self, expr):
        self.asked.append(render(expr, multiline=False))
        text = self.answers.get(expr.concept)
        return Lit(text) if text is not None else None

    def define(self, signature, vocabulary, context=None):
        self.defined.append(signature)
        self.context.append(context)
        head = signature.split("(", 1)[0]
        # Keyed by full signature when a word has more than one meaning,
        # by bare name when it has one.
        body = self.definitions.get(signature) or self.definitions.get(head)
        return parse(body) if body else None


def _key(utterance: str) -> str:
    return utterance.strip().rstrip("?!.").lower()


class Voice:
    """A seat that only talks, for checking what Speak hands it."""

    available = True
    last_error = ""

    def __init__(self, says: str = "okay") -> None:
        self.says = says
        self.spoken = []
        self.styles = []

    def speak(self, expression, style=""):
        self.spoken.append(expression)
        self.styles.append(style)
        return self.says

    def hear(self, utterance, vocabulary=None):
        return None

    def define(self, signature, vocabulary, context=None):
        return None

    def answer(self, expr):
        return None


def _is_unknown_answer(answer) -> bool:
    """Ignorance, however it got wrapped: a hole, or a question back at you."""
    from soup.expr import walk

    return any(
        isinstance(n, Call) and n.concept in ("Unknown", "Ask") for n in walk(answer)
    )


# Fresh numbers and names each run, so a test that secretly only knows
# "6 times 4 is 24" cannot hide. Failures print the values they used.
_rng = random.Random()
_PEOPLE = (
    "Ada", "Nia", "Omar", "Priya", "Jules", "Ren", "Samir", "Wren",
    "Theo", "Mira", "Lina", "Kai",
)
_THINGS = ("Bike", "Hat", "Kettle", "Violin", "Lamp", "Coat", "Mug")
_VERBS = ("Flumph", "Yeet", "Blorp", "Snerk", "Wibble", "Zorp")
_OPS = (("Quadruple", 4), ("Quintuple", 5), ("Sextuple", 6), ("Heptuple", 7), ("Octuple", 8))
_ADJECTIVES = ("Lazy", "Rusty", "Tiny", "Damp", "Loud")
_NOUNS = ("Dog", "Cat", "Boat", "Boot", "Goat")


def _n(lo: int = 2, hi: int = 12) -> int:
    return _rng.randint(lo, hi)


def _pick(options):
    return _rng.choice(options)


def _person() -> str:
    return _pick(_PEOPLE)


def _two_people():
    return tuple(_rng.sample(_PEOPLE, 2))


def _thing() -> str:
    return _pick(_THINGS)


def _verb() -> str:
    return _pick(_VERBS)


def _op():
    return _pick(_OPS)


def _items(k: int = 3, lo: int = 1, hi: int = 9):
    return [_n(lo, hi) for _ in range(k)]


def _list(xs) -> str:
    return "[%s]" % ", ".join(str(x) for x in xs)


def _q(s) -> str:
    """A string literal the concept parser will accept."""
    return '"%s"' % str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class TestExpressions(unittest.TestCase):
    def test_round_trip(self):
        a, b, c = _items()
        who = _person()
        name = _pick(("Keal", "Ash", "Rin", "Noor"))
        for source in [
            "Double(value=%s)" % a,
            "%s()" % who,
            "Map(collection=%s, transformation=Double())" % _list([a, b, c]),
            'Name(subject=User(), value="%s")' % name,
            "Let(file=Maximum(collection=Files(), by=FileSize()), then=Delete(target=file))",
        ]:
            self.assertEqual(render(parse(source), multiline=False), source)

    def test_case_decides_concept_versus_binding(self):
        who = _person()
        self.assertIsInstance(parse(who), Call)
        self.assertIsInstance(parse(who.lower()), Var)

    def test_definition_parses(self):
        head, body = parse_definition("MostAdorable(items) := Maximum(collection=items)")
        self.assertEqual(head.concept, "MostAdorable")
        self.assertEqual(body.concept, "Maximum")

    def test_match_and_substitute(self):
        x = _n()
        pattern = parse("Double(x)")
        bindings = match(pattern, parse("Double(%s)" % x))
        self.assertIsNotNone(bindings)
        self.assertEqual(bindings["x"], Lit(x))
        self.assertEqual(
            render(substitute(parse("Multiply(x, 2)"), bindings), multiline=False),
            "Multiply(%s, 2)" % x,
        )

    def test_match_rejects_different_concepts(self):
        x = _n()
        self.assertIsNone(match(parse("Double(x)"), parse("Triple(%s)" % x)))

    def test_multiline_render_is_reparseable(self):
        source = (
            "Let(file=Maximum(collection=Files(folder=Downloads()), by=FileSize()), "
            "then=Delete(target=file))"
        )
        pretty = render(parse(source), multiline=True)
        self.assertIn("\n", pretty)
        self.assertEqual(render(parse(pretty), multiline=False), source)

    def test_a_noun_with_an_adjective_is_still_a_name(self):
        who = _person()
        noun, adj = _pick(_NOUNS), _pick(_ADJECTIVES)
        op, n = _op()
        x = _n()
        self.assertTrue(is_name(parse("%s()" % who)))
        self.assertTrue(is_name(parse("%s(quality=%s())" % (noun, adj))))
        self.assertFalse(is_name(parse("Age(subject=%s())" % who)))
        self.assertFalse(is_name(parse("%s(%s)" % (op, x))))


class TestRealization(unittest.TestCase):
    def setUp(self):
        self.knowledge = fresh_knowledge()
        self.realizer = Realizer(self.knowledge)

    def realize(self, source: str):
        return self.realizer.realize(parse(source))

    def test_arithmetic(self):
        a, b, c = _n(), _n(), _n()
        self.assertEqual(self.realize("Multiply(%s, %s)" % (a, b)).value, Lit(a * b), (a, b))
        self.assertEqual(
            self.realize("Add(%s, Multiply(%s, %s))" % (a, b, c)).value, Lit(a + b * c), (a, b, c)
        )

    def test_taught_rules_are_realizations(self):
        x = _n()
        self.assertEqual(self.realize("Double(%s)" % x).value, Lit(x * 2), x)

    def test_rules_bind_by_position_when_names_differ(self):
        x = _n()
        even = _n(2, 10) * 2
        self.assertEqual(self.realize("Double(value=%s)" % x).value, Lit(x * 2), x)
        self.assertEqual(self.realize("Half(subject=%s)" % even).value, Lit(even // 2), even)

    def test_binding_survives_composition(self):
        a, b = _n(), _n()
        result = self.realize("Let(n=Add(%s, %s), then=Multiply(n, n))" % (a, b))
        self.assertEqual(result.value, Lit((a + b) ** 2), (a, b))

    def test_map_applies_a_concept_to_each_item(self):
        xs = _items()
        result = self.realize("Map(collection=%s, transformation=Double())" % _list(xs))
        self.assertEqual(result.value, Seq(tuple(Lit(x * 2) for x in xs)), xs)

    def test_partial_application_in_map(self):
        xs = _items(2)
        k = _n(3, 9)
        result = self.realize("Map(collection=%s, transformation=Multiply(%s))" % (_list(xs), k))
        self.assertEqual(result.value, Seq(tuple(Lit(x * k) for x in xs)), (xs, k))

    def test_operations_distribute_over_collections(self):
        xs = _items()
        k = _n()
        result = self.realize("Add(%s, %s)" % (_list(xs), k))
        self.assertEqual(result.value, Seq(tuple(Lit(x + k) for x in xs)), (xs, k))

    def test_distribution_leaves_single_argument_operations_alone(self):
        xs = _items()
        self.assertEqual(self.realize("Add(collection=%s)" % _list(xs)).value, Lit(sum(xs)), xs)

    def test_unknown_concept_becomes_a_gap(self):
        xs = _items(2)
        result = self.realize("MostAdorable(collection=%s)" % _list(xs))
        self.assertEqual([g.concept for g in result.gaps], ["MostAdorable"])

    def test_names_are_never_gaps(self):
        who = _person()
        result = self.realize("%s()" % who)
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.value, parse("%s()" % who))

    def test_an_adjective_on_a_noun_is_not_a_gap(self):
        noun, adj = _pick(_NOUNS), _pick(_ADJECTIVES)
        source = "%s(quality=%s())" % (noun, adj)
        result = self.realize(source)
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.value, parse(source))

    def test_missing_data_is_not_a_gap(self):
        who = _person()
        result = self.realize("Age(subject=%s())" % who)
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.value.concept, "Unknown")

    def test_ignorance_propagates_to_the_thing_actually_missing(self):
        who = _person()
        n = _n()
        result = self.realize("Add(%s, Age(subject=%s()))" % (n, who))
        self.assertEqual(render(result.value.get("about"), False), "Age(subject=%s())" % who)

    def test_facts_answer_questions(self):
        who = _person()
        age = _n(18, 80)
        self.knowledge.assert_fact(parse("Age(subject=%s(), value=%s)" % (who, age)))
        self.assertEqual(self.realize("Age(subject=%s())" % who).value, Lit(age), (who, age))

    def test_composed_rule_over_facts(self):
        older, younger = _two_people()
        hi, lo = _n(40, 80), _n(18, 35)
        self.knowledge.assert_fact(parse("Age(subject=%s(), value=%s)" % (older, hi)))
        self.knowledge.assert_fact(parse("Age(subject=%s(), value=%s)" % (younger, lo)))
        self.assertEqual(
            self.realize("OlderThan(left=%s(), right=%s())" % (older, younger)).value, Lit(True)
        )
        self.assertEqual(
            self.realize("OlderThan(left=%s(), right=%s())" % (younger, older)).value, Lit(False)
        )

    def test_query_fills_a_hole_from_memory(self):
        who, thing = _person(), _thing()
        self.knowledge.assert_fact(parse("Has(subject=%s(), object=%s())" % (who, thing)))
        result = self.realize("Query(pattern=Has(subject=Who(), object=%s()))" % thing)
        self.assertEqual(render(result.value, False), "%s()" % who)

    def test_learned_concepts_compose_with_builtins(self):
        op, factor = _op()
        x = _n()
        head, body = parse_definition("%s(x) := Multiply(x, %s)" % (op, factor))
        self.knowledge.add_rule(head, body)
        self.assertEqual(self.realize("%s(%s)" % (op, x)).value, Lit(x * factor), (op, x, factor))

    def test_a_rule_that_rewrites_to_itself_is_refused(self):
        head, body = parse_definition("Loop(x) := Loop(x)")
        self.knowledge.add_rule(head, body)
        result = self.realize("Loop(1)")
        self.assertTrue(
            any("rewrites to itself" in line for line in result.trace), result.trace
        )

    def test_a_rule_that_wraps_itself_is_refused_too(self):
        # The shape a model actually produced: not `X := X`, which the seat
        # already caught, but `X := Something(X)`.
        head, body = parse_definition("Loop(x) := Not(Loop(x))")
        self.knowledge.add_rule(head, body)
        result = self.realize("Loop(1)")
        self.assertTrue(
            any("rewrites to itself" in line for line in result.trace), result.trace
        )

    def test_recursion_that_shrinks_is_still_allowed(self):
        head, body = parse_definition("Countdown(x) := Countdown(Decrement(x))")
        self.knowledge.add_rule(head, body)
        result = self.realize("Countdown(3)")
        self.assertTrue(any("depth" in line for line in result.trace), result.trace)


class TestKnowledge(unittest.TestCase):
    def test_round_trip_through_disk(self):
        who = _person()
        age = _n(18, 80)
        thing = _thing()
        op = _pick(("Vibe", "Aura", "Mood"))
        k = fresh_knowledge()
        k.assert_fact(parse("Age(subject=%s(), value=%s)" % (who, age)))
        head, body = parse_definition("%s(x) := Count(x)" % op)
        k.add_rule(head, body)
        k.note_noun(thing, common=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            k.save(path)
            loaded = Knowledge()
            self.assertTrue(loaded.load(path))
        self.assertIn(thing, loaded.common_nouns)
        self.assertEqual(len(loaded.query(parse("Age(subject=%s(), value=v)" % who))), 1)
        self.assertEqual(loaded.rules_for(op)[0].source_text(), "%s(x) := Count(x)" % op)

    def test_memory_does_not_dump_the_seed(self):
        who = _person()
        k = fresh_knowledge()
        k.define("Developer", kind="entity")
        k.assert_fact(parse("Age(subject=%s(), value=30)" % who))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            k.save(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        names = [c["name"] for c in data["concepts"]]
        self.assertNotIn("Add", names)
        self.assertNotIn("Developer", names)
        self.assertTrue(any("Age" in f["proposition"] for f in data["facts"]))
        self.assertFalse(any(e["evidence"]["source"] == "builtin" for e in data["edges"]))
        self.assertFalse(any(r["evidence"]["source"] == "builtin" for r in data["rules"]))

    def test_reloading_does_not_clone_the_seed(self):
        k = fresh_knowledge()
        before_rules = len(k.rules_for("AtLeast"))
        before_edges = len(k.edges)
        fat = {
            "version": 1,
            "concepts": [],
            "edges": [e.to_json() for e in k.edges],
            "facts": [],
            "rules": [r.to_json() for bucket in k.rules.values() for r in bucket],
            "notes": [],
            "common_nouns": [],
        }
        k.load_json(fat)
        k.load_json(fat)
        self.assertEqual(len(k.rules_for("AtLeast")), before_rules)
        self.assertEqual(len(k.edges), before_edges)

    def test_the_same_entry_is_not_added_twice(self):
        k = fresh_knowledge()
        who = _person()
        age = _n(18, 80)
        op = _pick(("Vibe", "Aura", "Mood"))
        fact = parse("Age(subject=%s(), value=%s)" % (who, age))
        head, body = parse_definition("%s(x) := Count(x)" % op)
        k.assert_fact(fact)
        k.assert_fact(fact)
        k.add_rule(head, body)
        k.add_rule(head, body)
        k.relate(op, "RealizedBy", "Count")
        k.relate(op, "RealizedBy", "Count")
        self.assertEqual(len(k.query(fact)), 1)
        self.assertEqual(len(k.rules_for(op)), 1)
        self.assertEqual(
            len([e for e in k.edges if e.source == op and e.relation == "RealizedBy" and e.target == "Count"]),
            1,
        )

    def test_an_unlearned_stub_is_not_written_down(self):
        k = fresh_knowledge()
        head, body = parse_definition("Choose(x) := Seq(Add(x, 1))")
        k.add_rule(head, body)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            k.save(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        names = [c["name"] for c in data["concepts"]]
        self.assertIn("Choose", names)
        self.assertNotIn("Seq", names)

    def test_inheritance(self):
        k = fresh_knowledge()
        k.relate("Sprint", "IsA", "Run")
        self.assertTrue(k.is_a("Sprint", "Run"))
        self.assertFalse(k.is_a("Run", "Sprint"))

    def test_symmetry_comes_from_an_edge_not_from_python(self):
        k = fresh_knowledge()
        k.relate("Big", "RhymesWith", "Fig")
        self.assertEqual(k.related("Fig", "RhymesWith"), [])
        k.relate("RhymesWith", "HasProperty", SYMMETRIC)
        self.assertEqual(k.related("Fig", "RhymesWith"), ["Big"])

    def test_transitivity_chases_the_chain(self):
        k = fresh_knowledge()
        k.relate("Wheel", "PartOf", "Car")
        k.relate("Car", "PartOf", "Fleet")
        self.assertEqual(k.related("Wheel", "PartOf"), ["Car", "Fleet"])
        self.assertEqual(k.related("Wheel", "Owns"), [])

    def test_a_new_relation_can_join_the_taxonomy(self):
        k = fresh_knowledge()
        k.relate("Corgi", "BreedOf", "Dog")
        self.assertFalse(k.is_a("Corgi", "Dog"))
        k.relate("BreedOf", "HasProperty", TAXONOMIC)
        self.assertTrue(k.is_a("Corgi", "Dog"))

    def test_properties_can_be_asserted_as_ordinary_facts(self):
        k = fresh_knowledge()
        k.relate("Beside", "Nudges", "Near")
        k.assert_fact(parse("IsA(subject=Nudges(), kind=Symmetric())"))
        self.assertIn(SYMMETRIC, k.properties_of("Nudges"))
        self.assertEqual(k.related("Near", "Nudges"), ["Beside"])

    def test_inverse_relations_answer_each_other(self):
        k = fresh_knowledge()
        k.relate("ParentOf", "InverseOf", "ChildOf")
        parent, child = _two_people()
        k.relate(parent, "ParentOf", child)
        self.assertEqual(k.related(child, "ChildOf"), [parent])

    def test_seeded_relations_keep_their_old_behaviour(self):
        k = fresh_knowledge()
        self.assertEqual(k.related("Farewell", "OppositeOf"), ["Greeting"])
        self.assertEqual(k.related("Add", "SimilarTo"), ["Sum"])
        self.assertIn(TRANSITIVE, k.properties_of("IsA"))


class TestEars(unittest.TestCase):
    """Reading English is the model's job. Soup's own notation is still ours."""

    def test_english_without_a_model_is_admitted_to(self):
        said = reply(fresh(), "hello")
        self.assertEqual(said, "i have no model to hear you with")

    def test_empty_input_is_unintelligible(self):
        self.assertEqual(meaning(fresh(), "   "), "Unintelligible()")

    def test_raw_concept_syntax_is_accepted(self):
        a, b = _n(), _n()
        source = "Multiply(%s, %s)" % (a, b)
        self.assertEqual(meaning(fresh(), source), source)

    def test_a_definition_in_soup_notation_is_a_teach(self):
        op, factor = _op()
        source = "%s(x) := Multiply(x, %s)" % (op, factor)
        self.assertEqual(
            meaning(fresh(), source),
            "Teach(concept=%s(x), meaning=Multiply(x, %s))" % (op, factor),
        )

    def test_a_scripted_seat_is_what_the_ears_return(self):
        session = talking({"heya soup": "Greeting()"})
        self.assertEqual(meaning(session, "heya soup"), "Greeting()")
        self.assertEqual(session.respond("heya soup").heard.source, "llm")

    def test_an_unscripted_sentence_is_admitted_to_not_guessed(self):
        session = talking({"hi": "Greeting()"})
        said = reply(session, "asdkjh qwe zzz")
        self.assertIn("unscripted", said)

    def test_the_ears_do_not_resolve_anything(self):
        x = _n()
        english = "double %s" % x
        knowledge = fresh_knowledge()
        ears = Ears(
            knowledge,
            Discourse(),
            seat=Script({english: "Request(action=Double(%s))" % x}),
        )
        heard = ears.listen(english)
        self.assertEqual(render(heard.expr, False), "Request(action=Double(%s))" % x)
        self.assertEqual(knowledge.rules_for("Double")[0].source_text(), "Double(x) := Multiply(x, 2)")


class TestSeat(unittest.TestCase):
    """The LLM seat, exercised without a model anywhere near it."""

    def seat(self, reply_text: str) -> Seat:
        s = Seat()
        s._ask = lambda system, utterance, model=None: reply_text
        return s

    def test_a_reply_becomes_a_concept_expression(self):
        heard = self.seat("Remember(proposition=Sat(subject=Cat(), on=Mat()))").hear(
            "the cat sat on the mat"
        )
        self.assertEqual(
            render(heard, multiline=False),
            "Remember(proposition=Sat(subject=Cat(), on=Mat()))",
        )

    def test_fences_and_chatter_are_stripped(self):
        a, b = _n(), _n()
        expr = "Question(about=Multiply(%s, %s))" % (a, b)
        asked = "what is %s times %s" % (a, b)
        for wrapper in [
            "```\n%s\n```" % expr,
            "```python\n%s\n```" % expr,
            "Sure! Here you go:\n%s" % expr,
            "<think>hmm, a product</think>%s" % expr,
        ]:
            heard = self.seat(wrapper).hear(asked, ["Multiply"])
            self.assertEqual(render(heard, multiline=False), expr, wrapper)

    def test_a_dropped_bracket_is_repaired(self):
        a, b = _n(), _n()
        expr = "Question(about=Multiply(%s, %s))" % (a, b)
        heard = self.seat(expr[:-1]).hear("what is %s times %s" % (a, b), ["Multiply"])
        self.assertEqual(render(heard, multiline=False), expr)

    def test_nonsense_is_declined_rather_than_invented(self):
        self.assertIsNone(self.seat("I'm sorry, I can't help with that.").hear("x"))

    def test_the_ears_may_be_baffled_but_may_not_make_things_up(self):
        fake = _person()
        seat = self.seat("Question(about=Who(subject=%s(), name=Einstein()))" % fake)
        self.assertIsNone(seat.hear("who is albert einstein?"))
        self.assertIn(fake, seat.last_error)

    def test_a_name_that_was_actually_said_is_allowed_through(self):
        first, last = _pick(("Marie", "Niels", "Ada", "Alan")), _pick(("Curie", "Bohr", "Lovelace", "Turing"))
        spoken = "who is %s %s?" % (first.lower(), last.lower())
        concept = "%s%s" % (first, last)
        heard = self.seat("Question(about=Identity(subject=%s()))" % concept).hear(spoken)
        self.assertEqual(
            render(heard, multiline=False),
            "Question(about=Identity(subject=%s()))" % concept,
        )

    def test_a_packed_clause_is_heard_as_nested_concepts(self):
        blob = (
            'Question(about=Identity(subject=Concept(name='
            '"EnsuringOneIndividualDoesNotCarryTheBurdenOfAWholeWorkTask")))'
        )
        nested = (
            "Question(about=Ensure(that=Not(Carry(subject=Person(), "
            "object=Task(quality=Whole())))))"
        )
        replies = [blob, nested]
        seat = Seat()
        seat._ask = lambda system, utterance, model=None: replies.pop(0)
        heard = seat.hear(
            "as what is ensuring that one person does not carry a whole task referred to"
        )
        self.assertEqual(render(heard, multiline=False), nested)

    def test_an_inflected_word_is_the_same_word(self):
        op, _factor = _op()
        x = _n()
        heard = self.seat("Question(about=%s(%s))" % (op, x)).hear(
            "what is %s %sd" % (x, op.lower())
        )
        self.assertEqual(render(heard, multiline=False), "Question(about=%s(%s))" % (op, x))

    def test_an_answer_is_taken_but_a_restatement_is_not(self):
        seat = self.seat('Capital(subject=France(), value="Paris")')
        self.assertEqual(render(seat.answer(parse("Capital(subject=France())")), False), '"Paris"')
        seat = self.seat("Capital(subject=France())")
        self.assertIsNone(seat.answer(parse("Capital(subject=France())")))

    def test_a_model_that_does_not_know_says_so(self):
        self.assertIsNone(self.seat("Unknown()").answer(parse("Age(subject=User())")))

    def test_an_unquoted_answer_is_still_an_answer(self):
        seat = self.seat("100 degrees celsius")
        self.assertEqual(
            render(seat.answer(parse("BoilingPoint(subject=Water())")), False),
            '"100 degrees celsius"',
        )

    def test_a_question_restated_before_the_answer_is_seen_through(self):
        seat = self.seat('Capital(subject=France())\n"Paris"')
        self.assertEqual(render(seat.answer(parse("Capital(subject=France())")), False), '"Paris"')

    def test_a_definition_is_parsed(self):
        op, factor = _op()
        meaning = self.seat("Multiply(x, %s)" % factor).define("%s(x)" % op, ["Multiply"])
        self.assertEqual(render(meaning, False), "Multiply(x, %s)" % factor)

    def test_unknown_as_a_definition_is_declined(self):
        self.assertIsNone(self.seat("Unknown()").define("Quixotic(subject)", []))

    def test_defining_a_concept_as_itself_is_declined(self):
        op, _factor = _op()
        seat = self.seat("%s(x)" % op)
        self.assertIsNone(seat.define("%s(x)" % op, ["Multiply"]))
        self.assertIn("itself", seat.last_error)

    def test_endpoint_shape_follows_the_url(self):
        self.assertTrue(Seat(url="http://h/api/chat").native)
        self.assertFalse(Seat(url="http://h/v1/chat/completions").native)
        payload = Seat(url="http://h/api/chat")._payload("s", "u")
        self.assertIs(payload["think"], False)

    def test_the_ears_are_small_and_the_teacher_is_not(self):
        seat = Seat(url="http://h/api/chat")
        self.assertEqual(seat.model, "qwen3.5:4b")
        self.assertEqual(seat.teacher, "qwen3.8:27b")
        self.assertEqual(seat._payload("s", "u")["model"], "qwen3.5:4b")
        self.assertEqual(seat._payload("s", "u", model=seat.teacher)["model"], "qwen3.8:27b")

    def test_define_asks_the_teacher_not_the_ears(self):
        seat = Seat()
        asked = []

        def capture(system, utterance, model=None):
            asked.append(model)
            return "Multiply(x, 5)"

        seat._ask = capture
        seat.hear("hi", [])
        seat.define("Quintuple(x)", ["Multiply"])
        seat.answer(parse("Capital(subject=France())"))
        self.assertEqual(asked[0], "qwen3.5:4b")
        self.assertEqual(asked[1], "qwen3.8:27b")
        self.assertEqual(asked[2], "qwen3.8:27b")

    def test_the_teacher_can_be_named_from_the_environment(self):
        previous = {k: os.environ.get(k) for k in ("SOUP_LLM_MODEL", "SOUP_TEACHER_MODEL")}
        os.environ["SOUP_LLM_MODEL"] = "qwen3.5:4b"
        os.environ["SOUP_TEACHER_MODEL"] = "qwen3.8:27b"
        try:
            from soup.seat import seat_from_env

            seat = seat_from_env()
            self.assertEqual(seat.model, "qwen3.5:4b")
            self.assertEqual(seat.teacher, "qwen3.8:27b")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_a_missing_inprocess_teacher_is_not_the_ears(self):
        seat = Seat(embed=True, model="qwen3.5:4b", teacher="qwen3.8:27b")
        seat._engines["qwen3.8:27b"] = False
        self.assertIsNone(seat._engine("qwen3.8:27b"))

    def test_a_seat_does_not_load_weights_until_asked(self):
        seat = Seat(embed=True, model="no-such-model")
        self.assertIsNone(seat._local)


class TestEmbeddedWeights(unittest.TestCase):
    """Finding a GGUF. Loading one is a different test, and a slow one."""

    def test_a_gguf_path_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "toy.gguf")
            with open(path, "wb") as fh:
                fh.write(b"GGUF" + b"\0" * 8)
            self.assertEqual(find_weights(path), path)

    def test_a_non_gguf_file_is_not_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "toy.bin")
            with open(path, "wb") as fh:
                fh.write(b"NOPE")
            self.assertIsNone(find_weights(path))

    def test_an_ollama_name_resolves_to_its_blob(self):
        with tempfile.TemporaryDirectory() as d:
            blob = os.path.join(d, "blobs", "sha256-abc")
            os.makedirs(os.path.dirname(blob))
            with open(blob, "wb") as fh:
                fh.write(b"GGUF" + b"\0" * 8)
            manifest = os.path.join(
                d, "manifests", "registry.ollama.ai", "library", "qwen3.5", "4b"
            )
            os.makedirs(os.path.dirname(manifest))
            with open(manifest, "w") as fh:
                json.dump(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:abc",
                            }
                        ]
                    },
                    fh,
                )
            self.assertEqual(find_weights("qwen3.5:4b", root=d), blob)

    def test_qwen_names_map_onto_mlx_repos(self):
        self.assertEqual(mlx_id("qwen3.5:4b"), "mlx-community/Qwen3.5-4B-MLX-4bit")
        self.assertEqual(mlx_id("qwen3.5:9b"), "mlx-community/Qwen3.5-9B-MLX-4bit")
        self.assertEqual(
            mlx_id("mlx-community/Qwen3.5-4B-MLX-4bit"),
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        self.assertIsNone(mlx_id("/tmp/nope.gguf"))

    def test_the_real_qwen_gguf_is_already_on_disk(self):
        path = find_weights("qwen3.5:4b")
        if path is None:
            self.skipTest("qwen3.5:4b is not in ~/.ollama")
        self.assertTrue(os.path.isfile(path))


class TestVocabularyBrief(unittest.TestCase):
    def test_the_brief_groups_concepts_by_what_they_are(self):
        brief = vocabulary_brief(fresh_knowledge())
        self.assertIn("operation", brief)
        self.assertIn("Multiply", brief)
        self.assertIn("relation", brief)
        self.assertIn("IsA", brief)
        self.assertIn("TwentyFourHourTime", brief)

    def test_the_brief_says_inventing_is_allowed(self):
        self.assertIn("invent", vocabulary_brief(fresh_knowledge()).lower())

    def test_builtin_concepts_have_glosses_and_the_brief_includes_them(self):
        knowledge = fresh_knowledge()
        missing = sorted(name for name, concept in knowledge.concepts.items() if not concept.gloss)
        self.assertEqual([], missing)
        brief = vocabulary_brief(knowledge)
        self.assertIn("Add: combine values", brief)
        self.assertIn("VisitWebpage: fetch a page into structured sections", brief)

    def test_filtered_concepts_command_shows_glosses(self):
        session = Session(knowledge=fresh_knowledge(), memory_path=None, sources=[])
        output = io.StringIO()
        with redirect_stdout(output):
            _command(session, "concepts", "VisitWebpage")
        self.assertIn("VisitWebpage — fetch a page into structured sections", output.getvalue())


class TestFaithfulEars(unittest.TestCase):
    """A model may name a relation you did not say. It may not add things."""

    def seat(self, reply_text: str) -> Seat:
        s = Seat()
        s._ask = lambda system, utterance, model=None: reply_text
        return s

    def test_an_invented_relation_is_allowed(self):
        heard = self.seat("Request(action=Decrease(target=Volume()))").hear(
            "turn the volume down"
        )
        self.assertIsNotNone(heard)

    def test_an_invented_person_is_not(self):
        fake = _person()
        seat = self.seat("Question(about=Who(subject=%s(), name=Einstein()))" % fake)
        self.assertIsNone(seat.hear("who is albert einstein?"))
        self.assertIn(fake, seat.last_error)
        self.assertIsNotNone(seat.last_guess, "the rejected tree should still be sitting there")
        self.assertEqual(seat.last_guess.concept, "Question")

    def test_a_pronoun_may_be_tidied_up(self):
        heard = self.seat("Remember(proposition=Give(subject=She(), to=He()))").hear(
            "she gave him a book"
        )
        self.assertIsNotNone(heard)

    def test_a_smashed_name_is_the_word_they_typed(self):
        heard = self.seat("Question(about=Choose(between=TypeScript(), or=Rust()))").hear(
            "should i use typescript or rust?"
        )
        self.assertIsNotNone(heard)

    def test_a_paraphrase_leaf_is_not_a_swap(self):
        heard = self.seat(
            "Question(about=Choose(between=TypeScript(), or=Rust(), for=Developer()))"
        ).hear("should i use typescript or rust?")
        self.assertIsNotNone(heard, "Developer() is a paraphrase of the speaker, not Alice")
        self.assertIn("Developer", render(heard, False))

    def test_a_rejected_guess_is_still_on_the_heard(self):
        fake = _person()
        seat = self.seat("Question(about=Who(subject=%s(), name=Einstein()))" % fake)
        knowledge = fresh_knowledge()
        ears = Ears(knowledge, Discourse(), seat=seat)
        heard = ears.listen("who is albert einstein?")
        self.assertFalse(heard.understood)
        self.assertIsNotNone(heard.guessed)
        self.assertEqual(heard.guessed.concept, "Question")

    def test_affirm_is_mood_not_an_invented_person(self):
        heard = self.seat("Affirm()").hear("cool", [])
        self.assertIsNotNone(heard, "Affirm() is a speech act, not a fake Alice")
        self.assertEqual(heard.concept, "Affirm")

    def test_a_name_that_got_through_is_known_next_turn(self):
        knowledge = fresh_knowledge()
        first = Script({"who is ada": "Question(about=Identity(subject=Ada()))"})
        ears = Ears(knowledge, Discourse(), seat=first)
        ears.listen("who is ada")
        self.assertTrue(knowledge.knows_concept("Ada"), "heard names should join the session")
        later = Seat()
        later._ask = lambda system, utterance, model=None: "Question(about=Age(subject=Ada()))"
        self.assertIsNotNone(later.hear("how old is she", list(knowledge.concepts)))


class TestSpeaking(unittest.TestCase):
    """Talking is a concept: Speak(of=...). The seat chooses the words."""

    def test_the_seat_does_the_talking(self):
        session = fresh(llm=Voice("it's twenty four"))
        self.assertEqual(reply(session, "Speak(of=Answer(value=24))"), "it's twenty four")
        self.assertIn("Answer(value=24)", session.ears.seat.spoken)

    def test_a_style_is_a_concept_and_reaches_the_chair(self):
        voice = Voice("arr, twenty four")
        session = fresh(llm=voice)
        reply(session, "Speak(of=Answer(value=24), style=Pirate())")
        self.assertIn("Pirate()", voice.styles)

    def test_a_letter_survives_however_it_gets_worded(self):
        session = fresh(llm=Voice("divide 30 by 5 to find 6 teams"))
        said = reply(session, 'Speak(of=Answer(value="x", letter="B", to=Teams()))')
        self.assertTrue(said.startswith("B."), said)

    def test_with_nothing_in_the_chair_the_expression_goes_out_as_it_is(self):
        self.assertIn("24", reply(fresh(), "Speak(of=Answer(value=24))"))

    def test_wikipedia_concepts_are_spoken_instead_of_reducing_to_the_query(self):
        results = parse(
            'SearchResults(query="parakeet", results=['
            'SearchResult(title="Parakeet", '
            'url="https://en.wikipedia.org/wiki/Parakeet", '
            'snippet="A parakeet is a small parrot.")])'
        )
        script = Script(
            {
                "can you lookup what a parakeet is": (
                    'Question(about=WikipediaSearch(query="parakeet"))'
                )
            }
        )
        script.spoken = []

        def broken_mouth(expression, style=""):
            script.spoken.append(expression)
            return "parakeet"

        script.speak = broken_mouth
        session = fresh(llm=script)
        with patch("soup.web.wikipedia_search", return_value=results):
            said = reply(session, "can you lookup what a parakeet is?")

        self.assertEqual(said, "A parakeet is a small parrot.")
        self.assertEqual([], script.spoken)


class TestTheClock(unittest.TestCase):
    """Two concepts, not a list of the words people use for them."""

    def test_the_plain_time_is_a_twelve_hour_clock(self):
        said = reply(fresh(), "Question(about=Time())")
        self.assertRegex(said, r"\d?\d:\d\d [ap]m")

    def test_the_other_clock_face_is_its_own_concept(self):
        told = fresh().respond("Question(about=TwentyFourHourTime())")
        clock = told.answer.first("value")
        self.assertRegex(str(clock.value), r"^[012]\d:\d\d$")

    def test_time_with_an_unknown_argument_is_not_quietly_the_wrong_clock(self):
        said = reply(fresh(), "Question(about=Time(in=Swahili()))")
        self.assertNotRegex(said, r"\d\d:\d\d")

    def test_the_date_answers(self):
        self.assertRegex(reply(fresh(), "Question(about=Date())"), r"\d{4}")

    def test_timezone_is_not_where_you_live(self):
        said = reply(
            fresh(),
            "Question(about=Location(subject=User(), attribute=Timezone()))",
        )
        self.assertNotIn("live", said.lower())
        self.assertNotIn("Where", said)
        self.assertNotIn("never told", said.lower())
        self.assertTrue(said.strip(), said)

    def test_the_clock_is_not_a_fact_the_model_gets_to_file(self):
        session = talking({}, answers={"Time": "21:09"})
        reply(session, 'Question(about=Time(subject="9:09 pm", format=Military()))')
        props = [render(f.proposition, False) for f in session.knowledge.facts]
        self.assertFalse(any("Time(" in p for p in props), props)

    def test_a_missing_clock_shape_can_be_taught_as_python(self):
        session = talking({}, definitions={"Time": 'Python("21")'})
        said = reply(
            session, 'Question(about=Time(subject="9:09 pm", format=Military()))'
        )
        self.assertIn("21", said)
        rules = session.knowledge.rules_for("Time")
        self.assertTrue(rules)
        self.assertEqual(rules[0].method, "python")
        self.assertTrue(any("Python(" in r["rule"] for r in session.knowledge.to_json()["rules"]))

    def test_where_you_live_is_asked(self):
        # Nothing can look this up, so it comes back as a question for you.
        told = fresh().respond("Question(about=Location(subject=User()))")
        self.assertEqual(told.answer.concept, "Ask", told.said)
        self.assertEqual(told.answer.get("of"), Call("User", ()))

    def test_a_place_name_fills_the_ask(self):
        city = _pick(("Phoenix", "Oslo", "Lisbon"))
        session = fresh()
        reply(session, "Question(about=Location(subject=User()))")
        said = reply(session, "%s()" % city)
        self.assertIn(city.lower(), said.lower())

    def test_a_new_question_does_not_steal_the_ask(self):
        a, b = _n(), _n()
        session = fresh()
        reply(session, "Question(about=Location(subject=User()))")
        said = reply(session, "Question(about=Add(%s, %s))" % (a, b))
        self.assertIn(str(a + b), said)
        filled = reply(session, "Phoenix()")
        self.assertIn("phoenix", filled.lower())


class TestTeacher(unittest.TestCase):
    def test_a_positional_gap_is_teachable_as_x(self):
        op, _factor = _op()
        x = _n()
        lesson = Lesson(Gap(op, parse("%s(%s)" % (op, x))))
        self.assertEqual(lesson.params, ["x"])
        self.assertEqual(lesson.signature(), "%s(x)" % op)

    def test_a_named_gap_keeps_the_name(self):
        op = _pick(("Vibe", "Aura", "Mood"))
        xs = _items()
        lesson = Lesson(Gap(op, parse("%s(collection=%s)" % (op, _list(xs)))))
        self.assertEqual(lesson.params, ["collection"])
        self.assertEqual(lesson.signature(), "%s(collection=collection)" % op)

    def test_a_positional_among_named_args_is_x_not_z(self):
        lesson = Lesson(
            Gap(
                "Choose",
                parse(
                    "Choose(subject=Developer(), between=TypeScript(), Rust(), reason=Why())"
                ),
            )
        )
        self.assertEqual(
            lesson.signature(),
            "Choose(subject=subject, between=between, x, reason=reason)",
        )

    def test_a_grounded_definition_becomes_a_rule(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        op, factor = _op()
        x = _n()
        while x == factor:
            x = _n()
        lesson = teacher.ask(Gap(op, parse("%s(%s)" % (op, x))))
        taught = teacher.learn(lesson, parse("Multiply(x, %s)" % factor))
        self.assertIsNotNone(taught)
        self.assertEqual(
            knowledge.rules_for(op)[0].source_text(),
            "%s(x) := Multiply(x, %s)" % (op, factor),
        )

    def test_a_definition_made_of_mysteries_is_thrown_out(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        op = _pick(("Vibe", "Aura", "Mood"))
        mystery = _verb()
        lesson = teacher.ask(Gap(op, parse("%s(collection=%s)" % (op, _list(_items(1))))))
        self.assertIsNone(teacher.learn(lesson, parse("%s(Grooble())" % mystery)))
        self.assertEqual(knowledge.rules_for(op), [])

    def test_unrelated_arithmetic_is_not_an_answer(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = Lesson(
            Gap(
                "Choose",
                parse(
                    "Choose(subject=Developer(), between=TypeScript(), Rust(), reason=Why())"
                ),
            )
        )
        offered = parse(
            'Request(action=Seq(Add(5, 5), Add(Ref("it"), 100), Divide(Ref("it"), 2)))'
        )
        self.assertFalse(teacher.is_answer(lesson, offered))
        self.assertIsNone(teacher.learn(lesson, offered))
        self.assertEqual(knowledge.rules_for("Choose"), [])

    def test_a_rewrite_using_the_params_is_an_answer(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        op, factor = _op()
        x = _n()
        lesson = teacher.ask(Gap(op, parse("%s(%s)" % (op, x))))
        self.assertTrue(teacher.is_answer(lesson, parse("Multiply(x, %s)" % factor)))

    def test_python_is_a_grounded_definition(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        op = _pick(("HomeDir", "UserHome", "MyHome"))
        lesson = teacher.ask(Gap(op, parse("%s()" % op)))
        taught = teacher.learn(lesson, parse('Python("1")'))
        self.assertIsNotNone(taught)
        self.assertEqual(knowledge.rules_for(op)[0].method, "python")
        self.assertIn("Python(", knowledge.concept(op).gloss)

    def test_equals_true_is_not_a_definition(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = teacher.ask(Gap("IsTrue", parse("IsTrue(subject=X())")))
        self.assertIsNone(teacher.learn(lesson, parse("Equals(subject=subject, value=True)")))
        self.assertEqual(knowledge.rules_for("IsTrue"), [])

    def test_concat_is_not_a_definition_of_a_mystery(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = teacher.ask(Gap("Describe", parse("Describe(subject=X())")))
        self.assertIsNone(teacher.learn(lesson, parse('Concat(subject, " is a thing")')))
        self.assertEqual(knowledge.rules_for("Describe"), [])


class TestChoose(unittest.TestCase):
    """Pick among options by realizing the stem. Keep a definition that actually answers."""

    def test_a_computed_stem_picks_the_matching_option(self):
        said = reply(
            fresh(),
            "Question(about=Choose(among=[%s, %s], by=Divide(30, 5)))"
            % (_q("25 teams"), _q("6 teams")),
        )
        self.assertIn("6 teams", said.lower())
        self.assertNotIn("don't know", said.lower())

    def test_named_divide_still_computes(self):
        said = reply(fresh(), "Question(about=Divide(players=30, by=5))")
        self.assertIn("6", said)

    def test_a_new_shape_of_a_known_op_is_learned(self):
        session = talking(
            {
                "wattage": "Question(about=Choose(among=[%s, %s], by=Power(voltage=120, current=2)))"
                % (_q("240 W"), _q("120 W"))
            },
            definitions={"Power": "Multiply(voltage, current)"},
        )
        said = reply(session, "wattage")
        self.assertIn("240 w", said.lower(), said)
        self.assertTrue(session.knowledge.rules_for("Power"))


class TestManyMeanings(unittest.TestCase):
    """One word, several realizations. The shape of the call picks which."""

    def test_a_second_meaning_does_not_replace_the_first(self):
        session = talking(
            {"wattage": "Question(about=Power(voltage=120, current=2))"},
            definitions={"Power": "Multiply(voltage, current)"},
        )
        self.assertIn("240", reply(session, "wattage"))
        # The native meaning is still there for two bare numbers.
        self.assertIn("1024", reply(session, "Question(about=Power(2, 10))"))
        # And the taught one is still there for the electrical shape.
        self.assertIn("240", reply(session, "Question(about=Power(voltage=120, current=2))"))

    def test_two_taught_shapes_of_one_word_both_survive(self):
        session = talking(
            {},
            definitions={
                "Power(voltage=voltage, current=current)": "Multiply(voltage, current)",
                "Power(work=work, time=time)": "Divide(work, time)",
            },
        )
        self.assertIn("240", reply(session, "Question(about=Power(voltage=120, current=2))"))
        self.assertIn("25", reply(session, "Question(about=Power(work=100, time=4))"))
        heads = sorted(r.source_text() for r in session.knowledge.rules_for("Power"))
        self.assertEqual(len(heads), 2, heads)
        # And the first meaning still answers after the second was filed.
        self.assertIn("240", reply(session, "Question(about=Power(voltage=120, current=2))"))

    def test_scenery_in_the_call_does_not_block_the_meaning(self):
        # The live MMLU shape: the ears kept the microwave, which is scenery.
        # Voltage times current is still the meaning of this Power.
        session = talking(
            {
                "wattage": "Question(about=Choose(among=[%s, %s], "
                "by=Power(source=MicrowaveOven(), voltage=120, current=2)))"
                % (_q("240 W"), _q("120 W"))
            },
            definitions={"Power": "Multiply(voltage, current)"},
        )
        said = reply(session, "wattage")
        self.assertIn("240 w", said.lower(), said)

    def test_a_shape_that_failed_does_not_unfile_a_sibling_meaning(self):
        session = talking(
            {},
            definitions={"Power": "Multiply(voltage, current)"},
        )
        reply(session, "Question(about=Power(voltage=120, current=2))")
        kept = [r.source_text() for r in session.knowledge.rules_for("Power")]
        # A later miss on a different shape must not wipe the electrical rule.
        reply(session, "Question(about=Power(subject=Nation()))")
        self.assertEqual([r.source_text() for r in session.knowledge.rules_for("Power")], kept)


class TestNeverIDontKnow(unittest.TestCase):
    """Soup answers. The only admitted ignorance is a question only you can settle."""

    def test_a_fact_nobody_taught_is_asked_of_the_world(self):
        script = Script(
            {"how tall is the eiffel tower": "Question(about=Height(subject=EiffelTower()))"},
            answers={"Height": "330 metres"},
        )
        session = fresh(llm=script)
        said = reply(session, "how tall is the eiffel tower")
        self.assertIn("330", said, said)
        self.assertNotIn("don't know", said.lower())

    def test_an_operation_with_no_definition_still_answers(self):
        # Nothing defines it and nothing computes it, so the last resort is
        # an answer from outside. Being wrong is allowed; abstaining is not.
        script = Script(
            {"whats the vibe of this": "Question(about=Vibe(subject=Room()))"},
            answers={"Vibe": "pretty good"},
        )
        session = fresh(llm=script)
        said = reply(session, "whats the vibe of this")
        self.assertIn("pretty good", said.lower(), said)
        self.assertNotIn("don't know", said.lower())
        self.assertNotIn("no idea", said.lower())

    def test_only_a_question_about_you_is_handed_back(self):
        told = fresh().respond("Question(about=Age(subject=User()))")
        self.assertEqual(told.answer.concept, "Ask", told.said)
        self.assertEqual(told.answer.get("about"), parse("Age(subject=User())"))

    def test_a_known_operation_it_cannot_compute_asks_rather_than_shrugs(self):
        # Power is a known operation, the native declines this shape, and no
        # definition survives. Last resort is the world, not a shrug.
        script = Script(
            {"wattage": "Question(about=Power(source=MicrowaveOven(), voltage=120, current=2))"},
            answers={"Power": "240 W"},
        )
        session = fresh(llm=script)
        said = reply(session, "wattage")
        self.assertIn("240", said, said)
        self.assertNotIn("don't know", said.lower())
        self.assertNotIn("don't have", said.lower())

    def test_an_unteachable_verb_is_answered_not_handed_back(self):
        verb, thing = _verb(), _thing()
        script = Script(
            {"what now": "Question(about=%s(subject=%s()))" % (verb, thing)},
            answers={verb: "about six"},
        )
        session = fresh(llm=script)
        said = reply(session, "what now")
        self.assertIn("six", said.lower(), said)
        self.assertNotIn("teach me", said.lower())

    def test_power_is_voltage_times_current(self):
        said = reply(
            fresh(),
            "Question(about=Choose(among=[%s], by=Multiply(120, 2)))"
            % ", ".join(_q(w) for w in ("240 W", "120 W", "10 W", "480 W")),
        )
        self.assertIn("240 w", said.lower(), said)

    def test_a_taught_stem_is_kept_when_it_answers(self):
        first = "how many groups of 5 in 30"
        second = "how many groups of 4 in 12"
        script = Script(
            {
                first: "Question(about=Choose(among=[%s, %s], by=Teams(players=30, per=5)))"
                % (_q("25 teams"), _q("6 teams")),
                second: "Question(about=Choose(among=[%s, %s], by=Teams(players=12, per=4)))"
                % (_q("8 teams"), _q("3 teams")),
            },
            definitions={"Teams": "Divide(players, per)"},
        )
        session = fresh(llm=script)
        said = reply(session, first)
        self.assertIn("6 teams", said.lower(), said)
        self.assertTrue(session.knowledge.rules_for("Teams"), "the definition that answered should stay")
        said = reply(session, second)
        self.assertIn("3 teams", said.lower(), said)
        self.assertEqual(script.defined, ["Teams(players=players, per=per)"])

    def test_a_junk_definition_is_not_kept(self):
        english = "blot thirty and five"
        session = talking(
            {
                english: "Question(about=Choose(among=[%s, %s], by=Blot(30, 5)))"
                % (_q("0"), _q("2"))
            },
            definitions={"Blot": "Range(start=0, end=100)"},
        )
        reply(session, english)
        self.assertEqual(session.knowledge.rules_for("Blot"), [])

    def test_exam_english_hears_the_stem_and_picks(self):
        item = (
            "A total of 30 players will play basketball at a park. "
            "There will be exactly 5 players on each team. "
            "Which statement correctly explains how to find the number of teams needed?\n\n"
            "A. Multiply 5 by 5 to find 25 teams.\n"
            "B. Divide 30 by 5 to find 6 teams."
        )
        stem = item.split("\n\n", 1)[0]
        session = talking({stem: "Question(about=Divide(30, 5))"})
        said = reply(session, item)
        from soup.bench import extract_choice

        self.assertEqual(
            extract_choice(
                said,
                [
                    "Multiply 5 by 5 to find 25 teams.",
                    "Divide 30 by 5 to find 6 teams.",
                ],
            ),
            "B",
            said,
        )
        self.assertNotIn("unparseable", said.lower())
        self.assertNotIn("unscripted", said.lower())

    def test_a_described_act_is_taught_not_identified(self):
        stem = "as what is ensuring that one person does not carry a whole task referred to"
        item = stem + "\n\nA. Work delegation\nC. Work distribution"
        nested = (
            "Question(about=Identity(subject=Ensure(that=Not("
            "Carry(subject=Person(), object=Task(quality=Whole()))))))"
        )
        script = Script(
            {stem: nested},
            answers={"Identity": "workload distribution"},
            definitions={"Ensure": "that", "Carry": "that"},
        )
        session = fresh(llm=script)
        reply(session, item)
        taught = [d.split("(", 1)[0] for d in script.defined]
        self.assertTrue(
            "Ensure" in taught or "Carry" in taught,
            script.defined,
        )
        self.assertFalse(
            any(a.startswith("Identity") for a in script.asked),
            script.asked,
        )

    def test_a_failed_hear_does_not_speak_the_parse_error(self):
        item = "What color is the sky?\n\nA. Green\nB. Blue"
        session = talking({})
        said = reply(session, item)
        self.assertNotIn("unparseable", said.lower(), said)
        self.assertNotIn("unscripted", said.lower(), said)


class TestSelfTeaching(unittest.TestCase):
    """Ask what a concept means, keep the rule, then answer the original."""

    def test_an_unknown_operation_is_defined_then_answered(self):
        op, factor = _op()
        x = _n()
        english = "what is %s %sd" % (x, op.lower())
        session = talking(
            {english: "Question(about=%s(%s))" % (op, x)},
            definitions={op: "Multiply(x, %s)" % factor},
        )
        said = reply(session, english)
        self.assertIn(str(x * factor), said, (op, x, factor, said))
        self.assertIn(op, said)
        self.assertIn("worked out", said)
        self.assertTrue(session.knowledge.rules_for(op))

    def test_python_is_kept_as_the_realization(self):
        op = _pick(("HomeDir", "UserHome", "MyHome"))
        session = talking({}, definitions={op: 'Python("os.path.expanduser(\'~\')")'})
        said = reply(session, "Question(about=%s(of=User()))" % op)
        home = os.path.expanduser("~")
        self.assertIn(home, said)
        rule = session.knowledge.rules_for(op)[0]
        self.assertEqual(rule.method, "python")
        self.assertIn("Python(", session.knowledge.concept(op).gloss)
        dumped = session.knowledge.to_json()
        self.assertTrue(any(op in r["rule"] and "Python(" in r["rule"] for r in dumped["rules"]))
        self.assertTrue(any(c["name"] == op and "Python(" in c.get("gloss", "") for c in dumped["concepts"]))

    def test_the_rule_is_kept_and_the_next_question_is_free(self):
        op, factor = _op()
        x, y = _n(), _n()
        first = "what is %s %sd" % (x, op.lower())
        second = "what is the %s of %s" % (op.lower(), y)
        script = Script(
            {
                first: "Question(about=%s(%s))" % (op, x),
                second: "Question(about=%s(%s))" % (op, y),
            },
            definitions={op: "Multiply(x, %s)" % factor},
        )
        session = fresh(llm=script)
        reply(session, first)
        self.assertIn(str(y * factor), reply(session, second), (op, y, factor))
        self.assertEqual(script.defined, ["%s(x)" % op])
        self.assertEqual(script.asked, [])

    def test_learning_beats_guessing_the_answer(self):
        op, factor = _op()
        x = _n()
        english = "%s %s" % (op.lower(), x)
        script = Script(
            {english: "Question(about=%s(%s))" % (op, x)},
            answers={op: "about thirty"},
            definitions={op: "Multiply(x, %s)" % factor},
        )
        session = fresh(llm=script)
        self.assertIn(str(x * factor), reply(session, english), (op, x, factor))
        self.assertEqual(script.asked, [])

    def test_an_ungrounded_definition_is_not_kept(self):
        op = _pick(("Vibe", "Aura", "Mood"))
        xs = _items(2)
        english = "what is the %s of this" % op.lower()
        session = talking(
            {english: "Question(about=%s(collection=%s))" % (op, _list(xs))},
            definitions={op: "%s(x)" % _verb()},
        )
        reply(session, english)
        self.assertEqual(session.knowledge.rules_for(op), [])

    def test_a_request_is_not_answered_by_the_model(self):
        thing = _thing()
        english = "make me a %s" % thing.lower()
        script = Script(
            {english: "Request(action=Make(target=%s()))" % thing},
            answers={"Make": "a spooky picture"},
        )
        session = fresh(llm=script)
        said = reply(session, english)
        self.assertEqual(script.asked, [])
        self.assertNotIn("teach me", said)
        self.assertTrue(script.defined, "the teacher in the chair should have been asked")

    def test_an_assertion_is_remembered_not_fact_checked(self):
        animal, place = _pick(("Cat", "Fox", "Hen")), _pick(("Mat", "Log", "Rug"))
        english = "the %s sat on the %s" % (animal.lower(), place.lower())
        script = Script(
            {english: "Remember(proposition=Sat(subject=%s(), on=%s()))" % (animal, place)},
            answers={"Sat": "no it did not"},
        )
        session = fresh(llm=script)
        reply(session, english)
        self.assertEqual(script.asked, [])
        self.assertEqual(script.defined, [])

    def test_arithmetic_is_ours_not_the_model_s(self):
        a, b = _n(), _n()
        english = "what is %s times %s" % (a, b)
        script = Script(
            {english: "Question(about=Multiply(%s, %s))" % (a, b)},
            answers={"Multiply": "about thirty"},
        )
        session = fresh(llm=script)
        self.assertIn(str(a * b), reply(session, english), (a, b))
        self.assertEqual(script.asked, [])
        self.assertEqual(script.defined, [])

    def test_a_fact_we_lack_is_asked_rather_than_taught(self):
        first, last = _pick(("Marie", "Niels", "Ada")), _pick(("Curie", "Bohr", "Lovelace"))
        concept = "%s%s" % (first, last)
        english = "who is %s %s" % (first.lower(), last.lower())
        session = talking(
            {english: "Question(about=Identity(subject=%s()))" % concept},
            answers={"Identity": "a physicist"},
        )
        said = reply(session, english)
        self.assertIn("physicist", said)
        self.assertNotIn("teach me", said)

    def test_what_the_model_says_is_written_down_and_not_asked_twice(self):
        script = Script(
            {"what is the capital of france": "Question(about=Capital(subject=France()))"},
            answers={"Capital": "Paris"},
        )
        session = fresh(llm=script)
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertEqual(len(script.asked), 1)

    def test_a_silent_model_leaves_the_teacher_to_it(self):
        verb, noun = _verb(), _thing()
        english = "what is the %s of a %s" % (verb.lower(), noun.lower())
        session = talking({english: "Question(about=%s(subject=%s()))" % (verb, noun)})
        said = reply(session, english)
        self.assertNotIn("teach me", said)
        self.assertTrue(session.ears.seat.defined)
        self.assertTrue("know" in said or "no idea" in said, said)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.session = fresh(seed=11)

    def test_arithmetic_answers(self):
        a, b = _n(), _n()
        said = reply(self.session, "Question(about=Multiply(%s, %s))" % (a, b))
        self.assertIn(str(a * b), said, (a, b, said))

    def test_advice_is_not_a_lesson_about_choose(self):
        said = reply(
            fresh(),
            "Question(about=Advice(proposition=Choose(subject=Developer(), between=TypeScript(), Rust(), reason=Why())))",
        )
        self.assertNotIn("teach me", said)
        self.assertTrue("typescript" in said.lower() or "rust" in said.lower(), said)

    def test_an_unknown_choice_is_taught_not_guessed(self):
        script = Script(answers={"Choose": "TypeScript()"})
        session = fresh(llm=script)
        said = reply(
            session,
            "Question(about=Choose(between=TypeScript(), or=Rust()))",
        )
        self.assertNotIn("teach me", said)
        self.assertEqual(script.asked, [])

    def test_memory_across_turns(self):
        name = _pick(("Keal", "Ash", "Rin", "Noor"))
        reply(self.session, 'Remember(proposition=Name(subject=User(), value="%s"))' % name)
        self.assertIn(name, reply(self.session, "Question(about=Name(subject=User()))"))

    def test_facts_and_comparison(self):
        older, younger = _two_people()
        old_age, young_age = _n(40, 80), _n(10, 30)
        reply(self.session, "Remember(proposition=Age(subject=%s(), value=%s))" % (older, old_age))
        reply(self.session, "Remember(proposition=Age(subject=%s(), value=%s))" % (younger, young_age))
        told = self.session.respond(
            "Question(about=OlderThan(left=%s(), right=%s()))" % (older, younger)
        )
        self.assertEqual(
            told.answer.first("value", "truth"),
            Lit(True),
            (older, younger, old_age, young_age, told.said),
        )

    def test_unknown_person_is_admitted(self):
        who = _person()
        said = reply(self.session, "Question(about=Age(subject=%s()))" % who)
        self.assertTrue("know" in said or "no idea" in said, said)

    def test_an_unknown_verb_becomes_something_teachable(self):
        verb, thing = _verb(), _thing()
        said = reply(self.session, "Request(action=%s(target=%s()))" % (verb, thing))
        self.assertIn(verb, said)
        self.assertIn("teach me", said)

    def test_a_capability_told_is_a_capability_answered(self):
        can, cannot = _rng.sample(("Drive", "Fly", "Swim", "Cook", "Knit", "Sail"), 2)
        reply(self.session, "Remember(proposition=Can(subject=User(), action=%s()))" % can)
        # What Soup decided, not how it got worded: the wording is the seat's.
        told = self.session.respond(
            "Question(about=Can(subject=User(), action=%s()))" % can
        )
        self.assertEqual(told.answer.first("value", "truth"), Lit(True), told.said)
        never = self.session.respond(
            "Question(about=Can(subject=User(), action=%s()))" % cannot
        )
        self.assertTrue(_is_unknown_answer(never.answer), never.said)

    def test_a_denied_capability_stays_denied(self):
        skill = _pick(("Swim", "Juggle", "Yodel", "Whistle"))
        reply(self.session, "Remember(proposition=Can(subject=User(), action=%s()), truth=False)" % skill)
        told = self.session.respond(
            "Question(about=Can(subject=User(), action=%s()))" % skill
        )
        self.assertEqual(told.answer.first("value", "truth"), Lit(False), told.said)

    def test_what_we_do_know_about_a_name_still_comes_back(self):
        who = _person()
        reply(self.session, "Remember(proposition=IsA(subject=%s(), kind=Person()))" % who)
        # A bare name is recalled; Identity(subject=...) is a different question
        # and looks for an Identity fact, which we do not have.
        self.assertIn("person", reply(self.session, "Question(about=%s())" % who).lower())

    def test_teaching_loop_learns_then_answers(self):
        op = _pick(("Vibe", "Aura", "Mood", "Zing"))
        xs, ys = _items(3), _items(2)
        asked = reply(self.session, "Question(about=%s(collection=%s))" % (op, _list(xs)))
        self.assertIn(op, asked)
        learned = reply(self.session, "%s(x) := Count(x)" % op)
        self.assertIn(str(len(xs)), learned, (op, xs, learned))
        self.assertIn(
            str(len(ys)),
            reply(self.session, "Question(about=%s(collection=%s))" % (op, _list(ys))),
        )

    def test_an_unrelated_question_is_not_the_pending_lesson(self):
        op = _pick(("Vibe", "Aura", "Mood", "Zing"))
        reply(self.session, "Question(about=%s(collection=%s))" % (op, _list(_items(3))))
        self.assertEqual(self.session.lesson.concept, op)
        said = reply(self.session, "Question(about=Add(5, 5))")
        self.assertIn("10", said)
        self.assertNotIn("i know %s" % op.lower(), said.lower())
        self.assertEqual(self.session.knowledge.rules_for(op), [])
        self.assertIsNotNone(self.session.lesson)
        self.assertEqual(self.session.lesson.concept, op)

    def test_a_sequence_threads_it(self):
        said = reply(
            fresh(),
            'Request(action=Seq(Add(5, 5), Add(Ref("it"), 100), Divide(Ref("it"), 2)))',
        )
        self.assertIn("10", said)
        self.assertIn("110", said)
        self.assertIn("55", said)
        self.assertNotIn("teach me", said.lower())

    def test_sequence_is_a_paraphrase_of_seq(self):
        said = reply(
            fresh(),
            'Request(action=Sequence(Add(2, 2), Multiply(Ref("it"), 3)))',
        )
        self.assertIn("4", said)
        self.assertIn("12", said)
        self.assertNotIn("teach me", said.lower())

    def test_teaching_by_definition_syntax_without_a_gap(self):
        op, factor = _op()
        x = _n()
        reply(self.session, "%s(x) := Multiply(x, %s)" % (op, factor))
        said = reply(self.session, "Request(action=%s(%s))" % (op, x))
        self.assertIn(str(x * factor), said, (op, x, factor, said))

    def test_learned_concept_survives_a_save(self):
        op = _pick(("Zing", "Woz", "Narp", "Plox"))
        factor, x = _n(3, 9), _n()
        who, age = _person(), _n(18, 90)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            first = Session(memory_path=path, seed=1)
            reply(first, "%s(x) := Multiply(x, %s)" % (op, factor))
            reply(first, "Remember(proposition=Age(subject=%s(), value=%s))" % (who, age))
            first.save()
            second = Session(memory_path=path, seed=1)
            self.assertIn(str(x * factor), reply(second, "Request(action=%s(%s))" % (op, x)), (op, x, factor))
            self.assertIn(str(age), reply(second, "Question(about=Age(subject=%s()))" % who), (who, age))

    def test_nonsense_does_not_become_a_definition(self):
        op = _pick(("Vibe", "Aura", "Mood"))
        reply(self.session, "Question(about=%s(collection=%s))" % (op, _list(_items(3))))
        reply(self.session, _pick(("asdkjh qwe zzz", "bleep blorp nork", "zzzz qqq")))
        self.assertEqual(self.session.knowledge.rules_for(op), [])

    def test_explaining_shows_the_derivation(self):
        a, b = _n(), _n()
        reply(self.session, "Question(about=Multiply(%s, %s))" % (a, b))
        said = reply(self.session, 'Explain(about=Ref("that"))')
        self.assertIn(str(a * b), said, (a, b, said))

    def test_laughing_goes_through_the_seat(self):
        jokes = _rng.sample(("lol", "haha", "lmao", "heh", "lmaoooo", "rofl"), 2)
        session = talking({jokes[0]: "Laugh()", jokes[1]: "Laugh()"}, seed=7)
        for joke in jokes:
            said = reply(session, joke)
            self.assertNotIn("teach me", said)
            self.assertNotEqual(said, joke.title())

    def test_a_greeting_the_regex_list_never_contained(self):
        hi = _pick(("heya soup", "yo soup", "howdy soup"))
        session = talking({hi: "Greeting()"}, seed=3)
        self.assertTrue(reply(session, hi))


class TestTheWorld(unittest.TestCase):
    """Read, write, fetch, json. Natives for the wire; Create/Change are rules."""

    def test_inline_python_and_shell_run(self):
        a, b = _n(), _n()
        r = Realizer(fresh_knowledge())
        self.assertEqual(r.realize(parse('Python("%s + %s")' % (a, b))).value, Lit(a + b), (a, b))
        word = _pick(("hi", "yo", "hey"))
        self.assertEqual(r.realize(parse('Shell("echo %s")' % word)).value, Lit(word), word)

    def test_write_then_read_round_trips(self):
        text = _pick(("hi", "yo", "hey"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "%s.txt" % _pick(("a", "b", "c")))
            session = fresh()
            reply(session, "Request(action=Write(path=%s, contents=%s))" % (_q(path), _q(text)))
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), text)
            said = reply(session, "Question(about=Read(path=%s))" % _q(path))
            self.assertIn(text, said)

    def test_json_get_and_replace_compose(self):
        n = _n()
        word, other = _pick(("cat", "dog")), _pick(("hat", "log"))
        r = Realizer(fresh_knowledge())
        got = r.realize(parse("GetProperty(Json('{\"n\": %s}'), \"n\")" % n))
        self.assertEqual(got.value, Lit(n), n)
        swapped = r.realize(parse("Replace(%s, %s, %s)" % (_q(word), _q(word), _q(other))))
        self.assertEqual(swapped.value, Lit(other), (word, other))
        url = r.realize(
            parse(
                'Url(scheme="https", host="example.com", path="/w/api.php", '
                'query=Object(action="wbsearchentities", search="france"))'
            )
        )
        self.assertEqual(
            url.value,
            Lit("https://example.com/w/api.php?action=wbsearchentities&search=france"),
        )

    def test_create_a_two_file_app_from_scratch(self):
        fn = _pick(("hello", "greet", "wave"))
        word = _pick(("hi", "yo", "hey"))
        mod = _pick(("greet", "voice", "sayhi"))
        greet_src = 'def %s():\n    return "%s"\n' % (fn, word)
        main_src = "from %s import %s\nprint(%s())\n" % (mod, fn, fn)
        english = (
            "create a tiny python app with two files in %s: %s.py with a %s() "
            "function that returns %s, and main.py that prints %s()"
        )
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, "scratch")
            heard = (
                "Request(action=Create(target=Directory(path=%s), files=["
                "File(path=%s, contents=%s), File(path=%s, contents=%s)]))"
                % (_q(folder), _q(mod + ".py"), _q(greet_src), _q("main.py"), _q(main_src))
            )
            session = talking({english % (folder, mod, fn, word, fn): heard})
            reply(session, english % (folder, mod, fn, word, fn))
            greet_path = os.path.join(folder, mod + ".py")
            main_path = os.path.join(folder, "main.py")
            self.assertTrue(os.path.isfile(greet_path), greet_path)
            self.assertTrue(os.path.isfile(main_path), main_path)
            with open(greet_path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), greet_src)
            with open(main_path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), main_src)
            ran = subprocess.run(
                ["python3", "main.py"], cwd=folder, capture_output=True, text=True, timeout=5
            )
            self.assertEqual(ran.returncode, 0, ran.stderr)
            self.assertEqual(ran.stdout.strip(), word)

    def test_change_an_existing_program(self):
        old, new = _pick(("hello", "howdy")), _pick(("goodbye", "later"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hello.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('print("%s")\n' % old)
            english = "change %s so it prints %s instead of %s" % (path, new, old)
            heard = (
                "Request(action=Change(target=File(path=%s), from=%s, to=%s))"
                % (_q(path), _q(old), _q(new))
            )
            session = talking({english: heard})
            reply(session, english)
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), 'print("%s")\n' % new)


# Real payloads, trimmed. Recorded from wikidata.org so the shapes are not invented.
_WIKI = {
    ("wbsearchentities", "virginia", "item"): {
        "search": [{"id": "Q1370", "label": "Virginia", "description": "state of the United States",
                    "match": {"type": "label", "text": "Virginia"}}]
    },
    ("wbsearchentities", "france", "item"): {
        "search": [{"id": "Q142", "label": "France", "description": "country",
                    "match": {"type": "label", "text": "France"}}]
    },
    ("wbsearchentities", "capital", "property"): {
        "search": [{"id": "P36", "label": "capital", "description": "seat of government",
                    "match": {"type": "label", "text": "capital"}}]
    },
    ("wbsearchentities", "height", "property"): {
        "search": [
            {"id": "P2044", "label": "elevation above sea level",
             "match": {"type": "alias", "text": "height"}},
            {"id": "P2048", "label": "height", "description": "vertical length",
             "match": {"type": "label", "text": "height"}},
        ]
    },
    ("wbsearchentities", "tower", "property"): {
        "search": [{"id": "P14837", "label": "Tower Records Online artist ID",
                    "match": {"type": "label", "text": "Tower Records Online artist ID"}}]
    },
    ("wbgetclaims", "Q142", "P36"): {
        "claims": {"P36": [
            {"rank": "preferred",
             "mainsnak": {"snaktype": "value", "datavalue": {
                 "type": "wikibase-entityid", "value": {"id": "Q90"}}}},
            {"rank": "normal", "qualifiers": {"P582": []},
             "mainsnak": {"snaktype": "value", "datavalue": {
                 "type": "wikibase-entityid", "value": {"id": "Q84"}}}},
        ]}
    },
    ("wbgetclaims", "Q1370", "P36"): {
        "claims": {"P36": [
            {"rank": "preferred",
             "mainsnak": {"snaktype": "value", "datavalue": {
                 "type": "wikibase-entityid", "value": {"id": "Q49233"}}}}
        ]}
    },
    ("wbgetentities", "Q49233", None): {
        "entities": {"Q49233": {"labels": {"en": {"value": "Richmond"}}}}
    },
    ("wbgetentities", "Q90", None): {
        "entities": {"Q90": {"labels": {"en": {"value": "Paris"}}}}
    },
    # Nobody asks "what is the minimum number of players of chess", and
    # wikidata has no property called "players" at all.
    ("wbsearchentities", "players", "property"): {"search": []},
    ("wbsearchentities", "minimum number of players", "property"): {
        "search": [{"id": "P1872", "label": "minimum number of players",
                    "description": "minimum numbers of players of a game",
                    "match": {"type": "label", "text": "minimum number of players"}}]
    },
    ("wbsearchentities", "chess", "item"): {
        "search": [{"id": "Q718", "label": "chess", "description": "strategy board game",
                    "match": {"type": "label", "text": "chess"}}]
    },
    ("wbgetclaims", "Q718", "P1872"): {
        "claims": {"P1872": [
            {"rank": "normal",
             "mainsnak": {"snaktype": "value", "datavalue": {
                 "type": "quantity", "value": {"amount": "+2"}}}},
        ]}
    },
}


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


class _WikidataAPI:
    """Serve recorded Wikidata payloads through the real Fetch concept."""

    def __init__(self):
        self.urls = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.urls.append(url)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        action = params.get("action", [""])[0]
        if action == "wbsearchentities":
            key = (
                action,
                params.get("search", [""])[0],
                params.get("type", [""])[0],
            )
            payload = _WIKI.get(key, {"search": []})
        elif action == "wbgetclaims":
            key = (
                action,
                params.get("entity", [""])[0],
                params.get("property", [""])[0],
            )
            payload = _WIKI.get(key, {"claims": {}})
        elif action == "wbgetentities":
            entities = {}
            for qid in params.get("ids", [""])[0].split("|"):
                body = _WIKI.get((action, qid, None), {}).get("entities", {}).get(qid)
                if body is not None:
                    entities[qid] = body
            payload = {"entities": entities}
        else:
            payload = {}
        return _Response(payload)


class TestLookingItUp(unittest.TestCase):
    """Facts out of wikidata, and the vocabulary that comes with them."""

    def setUp(self):
        self.knowledge = fresh_knowledge()
        self.api = _WikidataAPI()
        self.fetch = patch("soup.builtins.urllib.request.urlopen", side_effect=self.api.open)
        self.fetch.start()
        self.addCleanup(self.fetch.stop)
        self.wiki = Wikidata(self.knowledge)

    def test_a_normal_session_uses_wikidata_for_an_ordinary_question(self):
        text = "what is the capital of virgina"
        session = Session(
            memory_path=None,
            # The ears normalize the typo to the entity's proper name.
            llm=Script({text: "Question(about=Capital(subject=Virginia()))"}),
        )
        self.assertEqual(len(session.realizer.sources), 1)
        self.assertIsInstance(session.realizer.sources[0], Wikidata)
        self.assertIn("Richmond", reply(session, text))
        self.assertTrue(any("search=virginia" in url for url in self.api.urls))
        self.assertEqual(Session(memory_path=None, sources=[]).realizer.sources, [])

    def test_the_min_of_an_attribute_is_looked_up_not_spoken_as_a_list_op(self):
        # No naming model: "minimum number of players" is Wikidata's own
        # label, so peeling Minimum off Players is enough.
        out = Realizer(self.knowledge, sources=[self.wiki]).realize(
            parse("Minimum(collection=Players(), in=Chess())")
        )
        self.assertEqual(out.value, Lit(2), out.trace)
        self.assertEqual(
            self.knowledge.query(
                parse('WikidataProperty(name=MinimumPlayers(), value=v)')
            )[0][1]["v"],
            Lit("P1872"),
        )

    def test_a_fact_is_looked_up_before_it_is_taught(self):
        naming = Script({}, definitions={"Players": "Count(x)"})
        naming.property_names = lambda concept, subject=None, specifically=None: [
            "minimum number of players"
        ]
        wiki = Wikidata(self.knowledge, seat=naming)
        out = Realizer(self.knowledge, seat=naming, sources=[wiki]).realize(
            parse("Players(subject=Chess())")
        )
        self.assertEqual(out.value, Lit(2), out.trace)
        self.assertEqual(naming.defined, [])

    def test_a_property_nobody_calls_that_is_found_by_paraphrase(self):
        naming = Script({})
        naming.names = ["minimum number of players"]
        naming.property_names = lambda concept, subject=None, specifically=None: naming.names
        wiki = Wikidata(self.knowledge, seat=naming)
        self.assertEqual(wiki.answer(parse("Players(inGame=Chess())")), Lit(2))
        # The property it settled on is written down, so you can see which
        # one answered you and say it was the wrong one.
        self.assertEqual(
            self.knowledge.query(parse('WikidataProperty(name=Players(), value=v)'))[0][1]["v"],
            Lit("P1872"),
        )

    def test_a_paraphrase_that_names_nothing_real_answers_nothing(self):
        naming = Script({})
        naming.property_names = lambda concept, subject=None, specifically=None: ["vibe of the thing"]
        wiki = Wikidata(self.knowledge, seat=naming)
        # A bad suggestion has to fail to match. It must never become a
        # fact with a citation stapled to it.
        self.assertIsNone(wiki.answer(parse("Players(inGame=Chess())")))
        self.assertEqual(self.knowledge.facts_mentioning("Players"), [])

    def test_without_a_seat_the_lookup_is_exactly_as_strict_as_before(self):
        wiki = Wikidata(self.knowledge)
        self.assertIsNone(wiki.answer(parse("Players(inGame=Chess())")))

    def test_a_property_of_a_thing_is_found(self):
        answer = self.wiki.answer(parse("Capital(subject=France())"))
        self.assertEqual(render(answer, multiline=False), '"Paris"')

    def test_the_capital_that_is_current_wins(self):
        answer = self.wiki.answer(parse("Capital(subject=France())"))
        self.assertEqual(render(answer, multiline=False), '"Paris"')

    def test_the_identifiers_are_learned_once(self):
        self.wiki.answer(parse("Capital(subject=France())"))
        spent = self.wiki.calls
        self.wiki.answer(parse("Capital(subject=France())"))
        self.assertLess(self.wiki.calls - spent, spent)
        self.assertEqual(self.wiki._recall("WikidataId", "France"), "Q142")
        self.assertEqual(self.wiki._recall("WikidataProperty", "Capital"), "P36")

    def test_every_returned_statement_is_kept_as_a_concept_fact(self):
        self.wiki.answer(parse("Capital(subject=France())"))
        statements = self.knowledge.query(
            parse(
                'WikidataStatement(subject=France(), attribute=Capital(), '
                'entityId="Q142", propertyId="P36", rank=r, snaktype=s, '
                'valueType=t, value=v, qualifiers=q, data=raw)'
            )
        )
        self.assertEqual(len(statements), 2)
        self.assertEqual(
            {binding["r"] for _, binding in statements},
            {Lit("preferred"), Lit("normal")},
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "memory.json")
            self.knowledge.save(path)
            restored = fresh_knowledge()
            self.assertTrue(restored.load(path))
            self.assertEqual(
                len(
                    restored.query(
                        parse(
                            'WikidataStatement(subject=France(), attribute=Capital(), '
                            'entityId="Q142", propertyId="P36", rank=r, snaktype=s, '
                            'valueType=t, value=v, qualifiers=q, data=raw)'
                        )
                    )
                ),
                2,
            )

    def test_a_concept_it_had_to_look_up_becomes_one_it_knows(self):
        self.assertFalse(self.knowledge.knows_concept("Capital"))
        self.wiki.answer(parse("Capital(subject=France())"))
        self.assertTrue(self.knowledge.knows_concept("Capital"))
        self.assertIn("government", self.knowledge.concept("Capital").gloss)

    def test_a_label_beats_an_alias(self):
        self.assertEqual(self.wiki._property("Height"), "P2048")
        results = self.knowledge.query(
            parse(
                'WikidataSearchResult(query="height", kind="property", id=i, '
                'label=l, description=d, match=m, data=raw)'
            )
        )
        self.assertEqual(len(results), 2)

    def test_wikidata_search_is_a_rule_over_fetch_and_structured_url(self):
        source = self.knowledge.rules_for("WikidataSearch")[0].source_text()
        self.assertIn("Json(Fetch(Url(", source)
        self.assertIn('host="www.wikidata.org"', source)
        self.assertIn('path="/w/api.php"', source)
        self.assertTrue(self.knowledge.rules_for("WikidataSearchHit"))
        self.assertTrue(self.knowledge.rules_for("WikidataBestStatement"))
        self.assertFalse(hasattr(Realizer(self.knowledge), "lookup"))

    def test_live_statements_are_filtered_before_preferred_rank(self):
        statements = parse(
            '[Object(rank="preferred", qualifiers=Object(P582=[1])), '
            'Object(rank="normal", qualifiers=Object())]'
        )
        chosen = self.wiki._realize(
            Call("WikidataBestStatement", (Arg("statements", statements),))
        )
        self.assertEqual(chosen.get("rank"), Lit("normal"))

    def test_multiple_entity_ids_are_encoded_as_one_structured_query_value(self):
        self.assertEqual(self.wiki._labels(["Q90", "Q142"]).get("Q90"), "Paris")
        self.assertTrue(any("ids=Q90%7CQ142" in url for url in self.api.urls))

    def test_a_near_miss_is_refused_rather_than_answered(self):
        self.assertIsNone(self.wiki._property("Tower"))

    def test_a_multiword_noun_phrase_is_searched_as_words(self):
        self.assertEqual(_subject_text(parse("Tower(quality=Eiffel())")), "eiffel tower")
        self.assertEqual(_subject_text(parse("NewZealand()")), "new zealand")

    def test_a_question_is_not_a_name(self):
        self.assertIsNone(_subject_text(parse("Maximum(collection=Files())")))

    def test_a_record_is_preferred_to_a_guess(self):
        session = talking(
            {"what is the capital of france": "Question(about=Capital(subject=France()))"},
            answers={"Capital": "Marseille"},
        )
        session.realizer.sources.append(Wikidata(session.knowledge))
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertEqual(session.ears.seat.asked, [])

    def test_the_model_still_gets_what_wikidata_lacks(self):
        op = _pick(("Vibe", "Aura", "Mood"))
        english = "what is the %s of france" % op.lower()
        session = talking(
            {english: "Question(about=%s(subject=France()))" % op},
            answers={op: "unmatched"},
        )
        session.realizer.sources.append(Wikidata(session.knowledge))
        self.assertIn("unmatched", reply(session, english))

    def test_no_network_is_not_an_error(self):
        self.fetch.stop()
        self.fetch = patch(
            "soup.builtins.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        )
        self.fetch.start()
        self.addCleanup(self.fetch.stop)
        wiki = Wikidata(fresh_knowledge())
        session = fresh()
        session.realizer.sources.append(wiki)
        said = reply(session, "Question(about=Capital(subject=France()))")
        self.assertFalse(wiki.available)
        self.assertNotIn("Traceback", said)
        a, b = _n(), _n()
        self.assertIn(str(a * b), reply(session, "Question(about=Multiply(%s, %s))" % (a, b)), (a, b))

    def test_a_dead_server_makes_soup_deaf_to_english_not_to_its_own_notation(self):
        self.fetch.stop()
        seat = Seat(url="http://localhost:9/api/chat", timeout=0.4)
        session = fresh(seed=2, llm=seat)
        a, b = _n(), _n()
        self.assertIn(str(a * b), reply(session, "Question(about=Multiply(%s, %s))" % (a, b)), (a, b))
        said = reply(session, "hello")
        self.assertFalse(seat.available)
        self.assertEqual(said, seat.last_error)


class TestDiscourse(unittest.TestCase):
    def test_it_points_at_the_last_answer(self):
        d = Discourse()
        n = _n(20, 99)
        d.note_answer(Lit(n))
        self.assertEqual(d.resolve_pronoun("it"), Lit(n))
        self.assertEqual(d.resolve_pronoun("that"), Lit(n))

    def test_pending_teach_steals_it(self):
        d = Discourse()
        d.note_answer(Lit(_n(20, 99)))
        d.pending_param = "x"
        self.assertEqual(d.resolve_pronoun("it"), Var("x"))


class TestHearMode(unittest.TestCase):
    """--hear stops at the ears. Multiply(2, 3) must not become 6."""

    def test_notation_is_not_realized(self):
        from soup.cli import _ears_only, hear_once

        out = hear_once(_ears_only(None), "Question(about=Multiply(2, 3))")
        self.assertIn("Multiply(2, 3)", out)
        self.assertNotIn("6", out)

    def test_scripted_english_stops_at_ears(self):
        from soup.cli import hear_once
        from soup.discourse import Discourse
        from soup.ears import Ears

        english = "where is myFunction?"
        ears = Ears(
            fresh_knowledge(),
            Discourse(),
            seat=Script({english: "Question(about=Location(subject=MyFunction()))"}),
        )
        out = hear_once(ears, english)
        self.assertIn("MyFunction", out)
        self.assertIn("Location", out)


class TestHearTeachSpeak(unittest.TestCase):
    """The pipeline stages as concepts: Hear, Teach, Speak."""

    def test_hear_then_realize_as_if_it_was_said(self):
        out = Realizer(fresh_knowledge()).realize(
            parse('Hear(text="Question(about=Multiply(2, 3))")')
        )
        # Utterance mood: hear, then realize. Question bottoms out as Answer.
        self.assertEqual(getattr(out.value, "concept", None), "Answer", out.value)
        self.assertEqual(out.value.first("value"), Lit(6))

    def test_hear_a_file_as_a_program_does_not_execute_it(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "nope.txt")
            src = os.path.join(d, "said.soup")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write("Request(action=Write(path=%s, contents=%s))" % (_q(target), _q("x")))
            out = Realizer(fresh_knowledge()).realize(
                parse("Hear(Read(path=%s), as=Program())" % _q(src))
            )
            self.assertFalse(os.path.exists(target), "program mood executed a write")
            self.assertEqual(getattr(out.value, "concept", None), "Meaning")

    def test_speak_says_what_mouth_would(self):
        out = Realizer(fresh_knowledge()).realize(parse("Speak(Add(2, 2))"))
        self.assertIsInstance(out.value, Lit)
        self.assertIn("4", str(out.value.value))

    def test_teach_without_a_body_asks_the_seat_to_define(self):
        op, factor = _op()
        x = _n()
        script = Script(definitions={op: "Multiply(x, %s)" % factor})
        realizer = Realizer(fresh_knowledge(), seat=script)
        taught = realizer.realize(parse("Teach(concept=%s(x))" % op))
        self.assertEqual(getattr(taught.value, "concept", None), "Taught", taught.value)
        self.assertIn(str(x * factor), str(realizer.realize(parse("%s(%s)" % (op, x))).value), (op, x, factor))


class TestTurnThinkConversation(unittest.TestCase):
    """The loop itself as concepts. Thinking is Conversation you are not in."""

    def test_turn_hears_then_speaks(self):
        out = Realizer(fresh_knowledge()).realize(
            parse('Turn(text="Question(about=Add(2, 2))")')
        )
        self.assertEqual(getattr(out.value, "concept", None), "Answer", out.value)
        self.assertEqual(out.value.first("value"), Lit(4), out.value)
        self.assertIsNotNone(out.said)
        self.assertIn("4", out.said)

    def test_think_returns_meaning_not_english(self):
        out = Realizer(fresh_knowledge()).realize(parse("Think(about=Add(2, 2))"))
        self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)
        inner = out.value.first("of", "value")
        self.assertEqual(inner, Lit(4), out.value)

    def test_think_does_not_do_the_write(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "nope.txt")
            out = Realizer(fresh_knowledge()).realize(
                parse("Think(about=Write(path=%s, contents=%s))" % (_q(target), _q("x")))
            )
            self.assertFalse(os.path.exists(target), "thinking executed a write")
            self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)

    def test_think_does_not_file_a_memory(self):
        knowledge = fresh_knowledge()
        Realizer(knowledge).realize(
            parse("Think(about=Remember(proposition=Age(subject=Ada(), value=30)))")
        )
        self.assertEqual(knowledge.facts_mentioning("Ada"), [])

    def test_think_budget_is_visible_and_finite(self):
        realizer = Realizer(fresh_knowledge(), think_budget=0)
        self.assertEqual(realizer.think_budget, 0)
        out = realizer.realize(parse("Think(about=Add(2, 2))"))
        self.assertEqual(getattr(out.value, "concept", None), "Unknown", out.value)

    def test_conversation_with_self_is_think(self):
        out = Realizer(fresh_knowledge()).realize(
            parse("Conversation(about=Add(2, 2), with=Self(), as=Thought())")
        )
        self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)
        self.assertEqual(out.value.first("of", "value"), Lit(4), out.value)

    def test_consult_is_a_seat_turn_that_stays_thought(self):
        english = "what is two plus two"
        script = Script({english: "Question(about=Add(2, 2))"})
        out = Realizer(fresh_knowledge(), seat=script).realize(
            parse("Consult(about=%s)" % _q(english))
        )
        self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)
        inner = out.value.first("of", "value")
        self.assertEqual(getattr(inner, "concept", None), "Answer", inner)
        self.assertEqual(inner.first("value"), Lit(4), inner)

    def test_subagent_is_a_nested_conversation_with_its_own_seat(self):
        english = "what is two plus two"
        helper = Script({english: "Question(about=Add(2, 2))"})
        out = Realizer(fresh_knowledge(), agents={"helper": helper}).realize(
            parse("Subagent(about=%s, with=Agent(name=%s))" % (_q(english), _q("helper")))
        )
        self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)
        inner = out.value.first("of", "value")
        self.assertEqual(getattr(inner, "concept", None), "Answer", inner)
        self.assertEqual(inner.first("value"), Lit(4), inner)

    def test_subagent_does_not_steal_the_outer_it(self):
        session = fresh()
        reply(session, "Question(about=Add(2, 2))")
        before = session.discourse.last_value
        session.realizer.realize(parse("Subagent(about=Add(9, 9))"))
        self.assertEqual(session.discourse.last_value, before)

    def test_hear_as_thought_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "nope.txt")
            src = os.path.join(d, "said.soup")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write("Request(action=Write(path=%s, contents=%s))" % (_q(target), _q("x")))
            out = Realizer(fresh_knowledge()).realize(
                parse("Hear(Read(path=%s), as=Thought())" % _q(src))
            )
            self.assertFalse(os.path.exists(target), "thought mood executed a write")
            self.assertEqual(getattr(out.value, "concept", None), "Meaning", out.value)


class TestThinkingItThrough(unittest.TestCase):
    """Solve: rounds of asking yourself, then realizing again with more."""

    def test_a_name_it_knows_nothing_about_becomes_a_kind(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        n = _n(2, 6)
        taught = teacher.classify(parse("Chess()"), parse("Game(players=%s)" % n))
        self.assertIsNotNone(taught)
        self.assertTrue(knowledge.is_a("Chess", "Game"))
        self.assertEqual(
            knowledge.query(parse("Players(subject=Chess(), value=v)"))[0][1]["v"],
            Lit(n),
        )
        # A name is described, never rewritten. Kate plays chess, not "the
        # game of two players".
        self.assertEqual(knowledge.rules_for("Chess"), [])

    def test_a_kind_made_of_nothing_we_know_is_declined(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        self.assertIsNone(teacher.classify(parse("Blorp()"), parse("Flumph()")))
        self.assertEqual(knowledge.facts_mentioning("Blorp"), [])

    def test_solving_classifies_a_name_then_answers_with_it(self):
        n = _n(2, 6)
        session = talking({}, definitions={"Chess": "Game(players=%s)" % n})
        said = reply(session, "Question(about=Solve(about=Players(subject=Chess())))")
        self.assertIn(str(n), said, said)
        self.assertTrue(session.knowledge.is_a("Chess", "Game"))

    def test_a_missing_verb_is_worked_out_without_asking_the_person(self):
        xs = _items(3)
        session = talking({}, definitions={"Fifth": "Last(collection=collection)"})
        said = reply(
            session,
            "Question(about=Solve(about=Fifth(collection=%s)))" % _list(xs),
        )
        self.assertIn(str(xs[-1]), said, (xs, said))
        self.assertNotIn("teach me", said)

    def test_listing_the_scene_does_not_ask_you_what_play_means(self):
        session = fresh()
        said = reply(
            session,
            "Request(action=List(subject=Activities(), given=["
            "Read(subject=Ann(), object=Book()), "
            "Cook(subject=Margaret()), "
            "Play(subject=Kate(), object=Chess()), "
            "Wash(subject=Marie()), "
            "Do(subject=Unknown(), value=Unknown())]))",
        )
        self.assertIn("Ann", said)
        self.assertIn("Kate", said)
        self.assertNotIn("teach me", said.lower())

    def test_asking_what_the_group_is_doing_reads_the_scene_back(self):
        session = fresh()
        said = reply(
            session,
            "Question(about=Activities(subject=AllOf(Sisters(count=5))), given=["
            "Read(subject=Ann(), object=Book()), "
            "Cook(subject=Margaret()), "
            "Play(subject=Kate(), object=Chess())])",
        )
        self.assertIn("Ann", said)
        self.assertIn("Margaret", said)
        self.assertIn("Kate", said)
        self.assertNotIn("no idea", said.lower())
        self.assertEqual(session.knowledge.facts_mentioning("Ann"), [])

    def test_asking_about_one_of_them_does_not_dump_the_scene(self):
        session = fresh()
        said = reply(
            session,
            "Question(about=Doing(subject=Fifth(of=Sisters(count=5))), given=["
            "Read(subject=Ann(), object=Book()), "
            "Play(subject=Kate(), object=Chess())])",
        )
        self.assertFalse(
            "Ann" in said and "Kate" in said,
            said,
        )

    def test_a_scene_is_believed_for_the_question_and_not_after(self):
        who = _person()
        age = _n(18, 80)
        session = fresh()
        said = reply(
            session,
            "Question(about=Solve(about=Age(subject=%s()), given=[Age(subject=%s(), value=%s)]))"
            % (who, who, age),
        )
        self.assertIn(str(age), said, (who, age, said))
        self.assertEqual(session.knowledge.facts_mentioning(who), [])

    def test_a_question_carrying_a_scene_is_solved_not_looked_up(self):
        who = _person()
        age = _n(18, 80)
        script = Script({}, answers={"Age": "about a hundred"})
        session = fresh(llm=script)
        said = reply(
            session,
            "Question(about=Age(subject=%s()), given=[Age(subject=%s(), value=%s)])"
            % (who, who, age),
        )
        self.assertIn(str(age), said, (who, age, said))
        self.assertEqual(script.asked, [])

    def test_a_definition_that_wraps_the_concept_is_thrown_away(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = teacher.ask(Gap("Doing", parse("Doing(subject=Kate())")))
        self.assertIsNone(teacher.learn(lesson, parse("RealizedBy(Doing(subject=Kate()))")))
        self.assertEqual(knowledge.rules_for("Doing"), [])

    def test_a_definition_that_mentions_itself_at_all_is_thrown_away(self):
        # `Sisters(count=count) := Count(Sisters())` got through a check
        # that only looked for the head verbatim. It is still not a
        # definition of anything.
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = teacher.ask(Gap("Sisters", parse("Sisters(count=5)")))
        self.assertIsNone(teacher.learn(lesson, parse("Count(Sisters())")))
        self.assertEqual(knowledge.rules_for("Sisters"), [])

    def test_bookkeeping_relations_are_not_definitions(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        lesson = teacher.ask(Gap("Doing", parse("Doing(subject=Kate())")))
        self.assertIsNone(teacher.learn(lesson, parse("RealizedBy(other=Kate())")))
        self.assertEqual(knowledge.rules_for("Doing"), [])

    def test_an_ordinal_picks_the_one_at_that_position(self):
        xs = _items(5)
        out = Realizer(fresh_knowledge()).realize(
            parse("Nth(collection=%s, index=4)" % _list(xs))
        )
        self.assertEqual(out.value, Lit(xs[3]), xs)

    def test_the_word_is_defined_where_it_turned_up(self):
        xs = _items(5)
        session = talking({}, definitions={"Fifth": "Nth(collection=of, index=5)"})
        said = reply(session, "Question(about=Fifth(of=%s))" % _list(xs))
        self.assertIn(str(xs[4]), said, (xs, said))
        # The signature alone reads "a fifth of"; the occurrence is what
        # makes it an ordinal.
        self.assertIn("Fifth(of=", session.realizer.seat.context[0])

    def test_an_answer_with_a_hole_in_it_is_not_an_answer(self):
        from soup.builtins import _collapse_unknown

        n = _n(2, 9)
        asked = parse("Doing(subject=Fifth(of=Sisters(count=%s)))" % n)
        half = parse(
            "Answer(value=Doing(subject=Unknown(about=Nth(collection=Sisters(count=%s),"
            " index=5))), to=%s)" % (n, render(asked, False))
        )
        # Not "it's what nth of 5 sisters, index 5 is's doing".
        self.assertEqual(
            _collapse_unknown(half).get("value"), Call("Unknown", (Arg("about", asked),))
        )
        # An answer with no hole in it is left exactly as it was.
        whole = parse("Answer(value=%s, to=%s)" % (n, render(asked, False)))
        self.assertEqual(_collapse_unknown(whole), whole)

    def test_counting_a_quantity_is_the_quantity(self):
        n = _n(2, 9)
        out = Realizer(fresh_knowledge()).realize(parse("Count(collection=%s)" % n))
        self.assertEqual(out.value, Lit(n))

    def test_an_invented_slot_name_still_finds_the_fact(self):
        n = _n(2, 9)
        knowledge = fresh_knowledge()
        knowledge.assert_fact(parse("Players(subject=Chess(), value=%s)" % n))
        # The ears invent a preposition per sentence. Same question.
        out = Realizer(knowledge).realize(parse("Players(inGame=Chess())"))
        self.assertEqual(out.value, Lit(n))

    def test_the_smallest_of_a_list_is_still_the_smallest(self):
        xs = _items(5)
        out = Realizer(fresh_knowledge()).realize(
            parse("Minimum(collection=%s)" % _list(xs))
        )
        self.assertEqual(out.value, Lit(min(xs)), xs)

    def test_a_scene_question_asks_each_hole_once(self):
        n = _n(2, 6)
        script = Script({}, definitions={"Chess": "Game(players=%s)" % n})
        session = fresh(llm=script)
        said = reply(
            session,
            "Question(about=Players(subject=Chess()), given=["
            "Read(subject=Ann(), object=Book()), Play(subject=Kate(), object=Chess())])",
        )
        self.assertIn(str(n), said, said)
        # The verb, once, on the way through; then the noun underneath it.
        self.assertEqual(script.defined, ["Players(subject=subject)", "Chess()"])
        # The scene is gone. What it taught us is not.
        self.assertEqual(session.knowledge.facts_mentioning("Kate"), [])
        self.assertTrue(session.knowledge.is_a("Chess", "Game"))

    def test_thinking_gives_up_rather_than_spinning(self):
        verb, thing = _verb(), _thing()
        script = Script({})
        session = fresh(llm=script)
        said = reply(
            session, "Question(about=Solve(about=%s(target=%s())))" % (verb, thing)
        )
        self.assertTrue(said.strip(), said)
        self.assertLessEqual(len(script.defined), 4, script.defined)

    def test_a_hole_rolls_into_thinking_before_it_asks(self):
        op, factor = _op()
        x = _n()
        while x == factor:
            x = _n()
        session = talking({}, definitions={op: "Multiply(x, %s)" % factor})
        said = reply(session, "Question(about=%s(%s))" % (op, x))
        self.assertIn(str(x * factor), said, (op, x, factor, said))
        self.assertNotIn("teach me", said)

    def test_only_they_know_where_they_live_so_thinking_is_skipped(self):
        script = Script({}, definitions={"LivesIn": "Location(subject=subject)"})
        session = fresh(llm=script)
        told = session.respond("Question(about=Location(subject=User()))")
        self.assertEqual(told.answer.concept, "Ask", told.said)
        self.assertEqual(script.defined, [])


class TestSessionIsTurn(unittest.TestCase):
    """The host loop is realizing Turn, not a private pipeline."""

    def test_respond_is_a_turn(self):
        session = fresh()
        out = session.respond("Question(about=Add(2, 2))")
        self.assertIn("4", out.said)
        self.assertIsNotNone(out.heard)
        self.assertIn("heard", out.result.effects)
        self.assertIn("spoke", out.result.effects)


if __name__ == "__main__":
    unittest.main()
