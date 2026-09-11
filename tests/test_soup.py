"""Tests for soup. Plain unittest, no dependencies.

    python3 -m unittest discover -s tests -v

English goes through a scripted seat. The tests never touch a network. Soup's
own notation (`Multiply(6, 4)`, `Vibe(x) := Count(x)`) is parsed directly,
because that is this system's language, not natural language understanding.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Seat, Session, fresh_knowledge, parse, parse_definition, render
from soup.discourse import Discourse
from soup.ears import Ears
from soup.expr import Call, Lit, Seq, Var, is_name, match, substitute
from soup.knowledge import SYMMETRIC, TAXONOMIC, TRANSITIVE, Knowledge
from soup.lookup import Wikidata, _subject_text
from soup.local import find_weights, mlx_id
from soup.mouth import Mouth
from soup.realize import Gap, Realizer
from soup.seat import vocabulary_brief
from soup.teacher import Lesson, Teacher


def reply(session: Session, text: str) -> str:
    return session.respond(text).said


def meaning(session: Session, text: str) -> str:
    heard = session.respond(text).heard
    assert heard is not None
    return render(heard.expr, multiline=False)


def fresh(seed: int = 1, llm=None, lookup=None) -> Session:
    return Session(memory_path=None, seed=seed, llm=llm, lookup=lookup)


def talking(heard, answers=None, definitions=None, seed: int = 1, lookup=None) -> Session:
    """A session whose ears recites a script instead of calling a model."""
    return fresh(seed=seed, llm=Script(heard, answers, definitions), lookup=lookup)


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

    def define(self, signature, vocabulary):
        self.defined.append(signature)
        head = signature.split("(", 1)[0]
        body = self.definitions.get(head)
        return parse(body) if body else None


def _key(utterance: str) -> str:
    return utterance.strip().rstrip("?!.").lower()


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

    def test_recursion_is_bounded(self):
        head, body = parse_definition("Loop(x) := Loop(x)")
        self.knowledge.add_rule(head, body)
        result = self.realize("Loop(1)")
        self.assertTrue(any("depth" in line for line in result.trace))


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
        s._ask = lambda system, utterance: reply_text
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


class TestFaithfulEars(unittest.TestCase):
    """A model may name a relation you did not say. It may not add things."""

    def seat(self, reply_text: str) -> Seat:
        s = Seat()
        s._ask = lambda system, utterance: reply_text
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

    def test_a_pronoun_may_be_tidied_up(self):
        heard = self.seat("Remember(proposition=Give(subject=She(), to=He()))").hear(
            "she gave him a book"
        )
        self.assertIsNotNone(heard)


class TestMouth(unittest.TestCase):
    def setUp(self):
        self.mouth = Mouth(fresh_knowledge(), Discourse(), seed=1)

    def test_an_adjective_sits_in_front_of_the_noun(self):
        noun, adj = _pick(_NOUNS), _pick(_ADJECTIVES)
        self.assertEqual(
            self.mouth.describe(parse("%s(quality=%s())" % (noun, adj))),
            "%s %s" % (adj.lower(), noun.lower()),
        )

    def test_laughing_is_not_the_word_laugh(self):
        said = self.mouth.say(parse("Laugh()"))
        self.assertNotEqual(said, "Laugh")
        self.assertNotIn("teach me", said)

    def test_modals_read_as_english(self):
        self.assertEqual(
            self.mouth.clause(parse("Can(subject=User(), action=Drive())")),
            "you can drive",
        )


class TestTheClock(unittest.TestCase):
    """Two concepts, not a list of the words people use for them."""

    def test_the_plain_time_is_a_twelve_hour_clock(self):
        said = reply(fresh(), "Question(about=Time())")
        self.assertRegex(said, r"\d?\d:\d\d [ap]m")

    def test_the_other_clock_face_is_its_own_concept(self):
        said = reply(fresh(), "Question(about=TwentyFourHourTime())")
        self.assertRegex(said, r"[012]\d:\d\d")
        self.assertNotIn("m", said.split(":")[-1])

    def test_time_with_an_unknown_argument_is_not_quietly_the_wrong_clock(self):
        said = reply(fresh(), "Question(about=Time(in=Swahili()))")
        self.assertNotRegex(said, r"\d\d:\d\d")

    def test_the_date_answers(self):
        self.assertRegex(reply(fresh(), "Question(about=Date())"), r"\d{4}")


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

    def test_a_grounded_definition_becomes_a_rule(self):
        knowledge = fresh_knowledge()
        teacher = Teacher(knowledge)
        op, factor = _op()
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
        self.assertIn("teach me", said)

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
        self.assertIn("teach me", said)
        self.assertIn(verb, said)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.session = fresh(seed=11)

    def test_arithmetic_answers(self):
        a, b = _n(), _n()
        said = reply(self.session, "Question(about=Multiply(%s, %s))" % (a, b))
        self.assertIn(str(a * b), said, (a, b, said))

    def test_memory_across_turns(self):
        name = _pick(("Keal", "Ash", "Rin", "Noor"))
        reply(self.session, 'Remember(proposition=Name(subject=User(), value="%s"))' % name)
        self.assertIn(name, reply(self.session, "Question(about=Name(subject=User()))"))

    def test_facts_and_comparison(self):
        older, younger = _two_people()
        old_age, young_age = _n(40, 80), _n(10, 30)
        reply(self.session, "Remember(proposition=Age(subject=%s(), value=%s))" % (older, old_age))
        reply(self.session, "Remember(proposition=Age(subject=%s(), value=%s))" % (younger, young_age))
        said = reply(self.session, "Question(about=OlderThan(left=%s(), right=%s()))" % (older, younger))
        self.assertIn(said[:3], ("yep", "yea", "yes", "cor"), (older, younger, old_age, young_age, said))

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
        self.assertIn(
            reply(self.session, "Question(about=Can(subject=User(), action=%s()))" % can)[:3],
            ("yep", "yea", "yes", "cor"),
        )
        said = reply(self.session, "Question(about=Can(subject=User(), action=%s()))" % cannot)
        self.assertTrue("know" in said or "no idea" in said or "never told" in said, said)

    def test_a_denied_capability_stays_denied(self):
        skill = _pick(("Swim", "Juggle", "Yodel", "Whistle"))
        reply(self.session, "Remember(proposition=Can(subject=User(), action=%s()), truth=False)" % skill)
        said = reply(self.session, "Question(about=Can(subject=User(), action=%s()))" % skill)
        self.assertIn(said[:2], ("no", "na", "nu"), said)

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
    ("wbgetentities", "Q90", None): {
        "entities": {"Q90": {"labels": {"en": {"value": "Paris"}}}}
    },
}


class _OfflineWikidata(Wikidata):
    """Wikidata with the wire pulled out, answering from recorded payloads."""

    def _get(self, **params):
        self.calls += 1
        action = params.get("action")
        if action == "wbsearchentities":
            key = (action, params["search"], params["type"])
        elif action == "wbgetclaims":
            key = (action, params["entity"], params["property"])
        else:
            key = (action, params.get("ids"), None)
        return _WIKI.get(key)


class TestLookingItUp(unittest.TestCase):
    """Facts out of wikidata, and the vocabulary that comes with them."""

    def setUp(self):
        self.knowledge = fresh_knowledge()
        self.wiki = _OfflineWikidata(self.knowledge)

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

    def test_a_concept_it_had_to_look_up_becomes_one_it_knows(self):
        self.assertFalse(self.knowledge.knows_concept("Capital"))
        self.wiki.answer(parse("Capital(subject=France())"))
        self.assertTrue(self.knowledge.knows_concept("Capital"))
        self.assertIn("government", self.knowledge.concept("Capital").gloss)

    def test_a_label_beats_an_alias(self):
        self.assertEqual(self.wiki._property("Height"), "P2048")

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
            lookup=_OfflineWikidata(fresh_knowledge()),
        )
        # The lookup on the session needs the session's own store so learned
        # ids land where realization will look next time.
        session.realizer.lookup = _OfflineWikidata(session.knowledge)
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertEqual(session.ears.seat.asked, [])

    def test_the_model_still_gets_what_wikidata_lacks(self):
        op = _pick(("Vibe", "Aura", "Mood"))
        english = "what is the %s of france" % op.lower()
        session = talking(
            {english: "Question(about=%s(subject=France()))" % op},
            answers={op: "unmatched"},
        )
        session.realizer.lookup = _OfflineWikidata(session.knowledge)
        self.assertIn("unmatched", reply(session, english))

    def test_no_network_is_not_an_error(self):
        wiki = Wikidata(fresh_knowledge(), timeout=0.4, endpoint="http://localhost:9/w/api.php")
        session = fresh()
        session.realizer.lookup = wiki
        said = reply(session, "Question(about=Capital(subject=France()))")
        self.assertFalse(wiki.available)
        self.assertNotIn("Traceback", said)
        a, b = _n(), _n()
        self.assertIn(str(a * b), reply(session, "Question(about=Multiply(%s, %s))" % (a, b)), (a, b))

    def test_a_dead_server_makes_soup_deaf_to_english_not_to_its_own_notation(self):
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


if __name__ == "__main__":
    unittest.main()
