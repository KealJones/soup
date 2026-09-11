"""Tests for soup. Plain unittest, no dependencies.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Seat, Session, fresh_knowledge, parse, parse_definition, render
from soup.expr import Arg, Call, Lit, Seq, Var, call, match, substitute
from soup.knowledge import SYMMETRIC, TAXONOMIC, TRANSITIVE, Evidence, Knowledge
from soup.lookup import Wikidata, _subject_text
from soup.realize import Realizer
from soup.seat import vocabulary_brief


def reply(session: Session, text: str) -> str:
    return session.respond(text).said


def meaning(session: Session, text: str) -> str:
    return render(session.respond(text).heard.expr, multiline=False)


def fresh(seed: int = 1) -> Session:
    return Session(memory_path=None, seed=seed)


class TestExpressions(unittest.TestCase):
    def test_round_trip(self):
        for source in [
            "Double(value=6)",
            "Greg()",
            "Map(collection=[1, 2, 3], transformation=Double())",
            'Name(subject=User(), value="Keal")',
            "Let(file=Maximum(collection=Files(), by=FileSize()), then=Delete(target=file))",
        ]:
            self.assertEqual(render(parse(source), multiline=False), source)

    def test_case_decides_concept_versus_binding(self):
        self.assertIsInstance(parse("Greg"), Call)
        self.assertIsInstance(parse("greg"), Var)

    def test_definition_parses(self):
        head, body = parse_definition("MostAdorable(items) := Maximum(collection=items)")
        self.assertEqual(head.concept, "MostAdorable")
        self.assertEqual(body.concept, "Maximum")

    def test_match_and_substitute(self):
        pattern = parse("Double(x)")
        bindings = match(pattern, parse("Double(6)"))
        self.assertIsNotNone(bindings)
        self.assertEqual(bindings["x"], Lit(6))
        self.assertEqual(
            render(substitute(parse("Multiply(x, 2)"), bindings), multiline=False),
            "Multiply(6, 2)",
        )

    def test_match_rejects_different_concepts(self):
        self.assertIsNone(match(parse("Double(x)"), parse("Triple(6)")))

    def test_multiline_render_is_reparseable(self):
        source = (
            "Let(file=Maximum(collection=Files(folder=Downloads()), by=FileSize()), "
            "then=Delete(target=file))"
        )
        pretty = render(parse(source), multiline=True)
        self.assertIn("\n", pretty)
        self.assertEqual(render(parse(pretty), multiline=False), source)


class TestRealization(unittest.TestCase):
    def setUp(self):
        self.knowledge = fresh_knowledge()
        self.realizer = Realizer(self.knowledge)

    def realize(self, source: str):
        return self.realizer.realize(parse(source))

    def test_arithmetic(self):
        self.assertEqual(self.realize("Multiply(6, 4)").value, Lit(24))
        self.assertEqual(self.realize("Add(2, Multiply(3, 4))").value, Lit(14))

    def test_taught_rules_are_realizations(self):
        self.assertEqual(self.realize("Double(21)").value, Lit(42))

    def test_rules_bind_by_position_when_names_differ(self):
        self.assertEqual(self.realize("Double(value=21)").value, Lit(42))
        self.assertEqual(self.realize("Half(subject=10)").value, Lit(5))

    def test_binding_survives_composition(self):
        result = self.realize("Let(n=Add(2, 3), then=Multiply(n, n))")
        self.assertEqual(result.value, Lit(25))

    def test_map_applies_a_concept_to_each_item(self):
        result = self.realize("Map(collection=[1, 2, 3], transformation=Double())")
        self.assertEqual(result.value, Seq((Lit(2), Lit(4), Lit(6))))

    def test_partial_application_in_map(self):
        result = self.realize("Map(collection=[1, 2], transformation=Multiply(10))")
        self.assertEqual(result.value, Seq((Lit(10), Lit(20))))

    def test_operations_distribute_over_collections(self):
        result = self.realize("Add([1, 2, 3], 10)")
        self.assertEqual(result.value, Seq((Lit(11), Lit(12), Lit(13))))

    def test_distribution_leaves_single_argument_operations_alone(self):
        self.assertEqual(self.realize("Add(collection=[1, 2, 3])").value, Lit(6))

    def test_unknown_concept_becomes_a_gap(self):
        result = self.realize("MostAdorable(collection=[1, 2])")
        self.assertEqual([g.concept for g in result.gaps], ["MostAdorable"])

    def test_names_are_never_gaps(self):
        result = self.realize("Greg()")
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.value, parse("Greg()"))

    def test_missing_data_is_not_a_gap(self):
        result = self.realize("Age(subject=Carl())")
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.value.concept, "Unknown")

    def test_ignorance_propagates_to_the_thing_actually_missing(self):
        result = self.realize("Add(1, Age(subject=Carl()))")
        self.assertEqual(render(result.value.get("about"), False), "Age(subject=Carl())")

    def test_facts_answer_questions(self):
        self.knowledge.assert_fact(parse("Age(subject=Alice(), value=30)"))
        self.assertEqual(self.realize("Age(subject=Alice())").value, Lit(30))

    def test_composed_rule_over_facts(self):
        self.knowledge.assert_fact(parse("Age(subject=Alice(), value=30)"))
        self.knowledge.assert_fact(parse("Age(subject=Bob(), value=24)"))
        self.assertEqual(self.realize("OlderThan(left=Alice(), right=Bob())").value, Lit(True))
        self.assertEqual(self.realize("OlderThan(left=Bob(), right=Alice())").value, Lit(False))

    def test_query_fills_a_hole_from_memory(self):
        self.knowledge.assert_fact(parse("Has(subject=Greg(), object=Car())"))
        result = self.realize("Query(pattern=Has(subject=Who(), object=Car()))")
        self.assertEqual(render(result.value, False), "Greg()")

    def test_learned_concepts_compose_with_builtins(self):
        head, body = parse_definition("Quadruple(x) := Multiply(Double(x), 2)")
        self.knowledge.add_rule(head, body)
        self.assertEqual(self.realize("Quadruple(5)").value, Lit(20))

    def test_recursion_is_bounded(self):
        head, body = parse_definition("Loop(x) := Loop(x)")
        self.knowledge.add_rule(head, body)
        result = self.realize("Loop(1)")
        self.assertTrue(any("depth" in line for line in result.trace))


class TestKnowledge(unittest.TestCase):
    def test_round_trip_through_disk(self):
        k = fresh_knowledge()
        k.assert_fact(parse("Age(subject=Alice(), value=30)"))
        head, body = parse_definition("Vibe(x) := Count(x)")
        k.add_rule(head, body)
        k.note_noun("Car", common=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            k.save(path)
            loaded = Knowledge()
            self.assertTrue(loaded.load(path))
        self.assertIn("Car", loaded.common_nouns)
        self.assertEqual(len(loaded.query(parse("Age(subject=Alice(), value=v)"))), 1)
        self.assertEqual(loaded.rules_for("Vibe")[0].source_text(), "Vibe(x) := Count(x)")

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
        k.relate("Ann", "ParentOf", "Bo")
        self.assertEqual(k.related("Bo", "ChildOf"), ["Ann"])

    def test_seeded_relations_keep_their_old_behaviour(self):
        k = fresh_knowledge()
        self.assertEqual(k.related("Farewell", "OppositeOf"), ["Greeting"])
        self.assertEqual(k.related("Add", "SimilarTo"), ["Sum"])
        self.assertIn(TRANSITIVE, k.properties_of("IsA"))


class TestEars(unittest.TestCase):
    def setUp(self):
        self.session = fresh()

    def test_arithmetic_questions(self):
        self.assertEqual(meaning(self.session, "what is 6 times 4"), "Question(about=Multiply(6, 4))")
        self.assertEqual(
            meaning(self.session, "what's 2 plus 3 times 4"),
            "Question(about=Add(2, Multiply(3, 4)))",
        )

    def test_statements_become_memory(self):
        self.assertEqual(
            meaning(self.session, "Alice is 30 years old"),
            "Remember(proposition=Age(subject=Alice(), value=30))",
        )
        self.assertEqual(
            meaning(self.session, "Greg has the car"),
            "Remember(proposition=Has(subject=Greg(), object=Car()))",
        )

    def test_yes_no_questions_undo_inversion(self):
        self.assertEqual(
            meaning(self.session, "is Alice older than Bob"),
            "Ask(proposition=OlderThan(left=Alice(), right=Bob()))",
        )

    def test_wh_questions_become_queries(self):
        self.assertEqual(
            meaning(self.session, "who has the car"),
            "Query(pattern=Has(subject=Who(), object=Car()))",
        )

    def test_pronouns_resolve_against_the_conversation(self):
        reply(self.session, "what is 6 times 7")
        self.assertEqual(meaning(self.session, "double it"), "Request(action=Double(42))")

    def test_collections_and_mapping(self):
        self.assertEqual(
            meaning(self.session, "double every number in [1,2,3]"),
            "Request(action=Map(collection=[1, 2, 3], transformation=Double()))",
        )

    def test_raw_concept_syntax_is_accepted(self):
        self.assertEqual(meaning(self.session, "Multiply(3, 4)"), "Multiply(3, 4)")

    def test_decimals_survive_tokenizing(self):
        self.assertEqual(
            meaning(self.session, "what is 3.5 plus 1.5"), "Question(about=Add(3.5, 1.5))"
        )

    def test_bare_collections_and_operators(self):
        self.assertEqual(meaning(self.session, "[1,2]"), "Question(about=[1, 2])")
        self.assertEqual(meaning(self.session, "what is 2 * 3"), "Question(about=Multiply(2, 3))")

    def test_brackets_are_never_verbs(self):
        self.assertEqual(meaning(self.session, "sort [3,1]"), "Request(action=Sort([3, 1]))")

    def test_unknown_words_still_get_a_shape(self):
        """Word order is readable even when not one word is."""
        self.assertEqual(
            meaning(self.session, "asdkjh qwe zzz"),
            "Remember(proposition=Qwe(subject=Asdkjh(), object=Zzz()))",
        )

    def test_word_order_survives_an_unknown_vocabulary(self):
        cases = {
            "the cat sat on the mat": "Remember(proposition=Sat(subject=Cat(), on=Mat()))",
            "she gave him a book": (
                "Remember(proposition=Gave(subject=Her(), recipient=Him(), object=Book()))"
            ),
            "please send an email to Dave about the meeting": (
                "Request(action=Send(object=Email(), to=Dave(), about=Meeting()))"
            ),
        }
        for utterance, expected in cases.items():
            self.assertEqual(meaning(fresh(), utterance), expected, utterance)

    def test_prepositions_become_argument_names(self):
        said = meaning(self.session, "the fox jumps over the lazy dog")
        self.assertIn("over=", said)
        self.assertIn("subject=Fox()", said)


class TestSeat(unittest.TestCase):
    """The LLM seat, exercised without a model anywhere near it."""

    def seat(self, reply: str) -> Seat:
        s = Seat()
        s._ask = lambda system, utterance: reply
        return s

    def test_a_reply_becomes_a_concept_expression(self):
        heard = self.seat("Remember(proposition=Sat(subject=Cat(), on=Mat()))").hear(
            "the cat sat on the mat"
        )
        self.assertEqual(render(heard, multiline=False), "Remember(proposition=Sat(subject=Cat(), on=Mat()))")

    def test_fences_and_chatter_are_stripped(self):
        for wrapper in [
            "```\nQuestion(about=Multiply(6, 4))\n```",
            "```python\nQuestion(about=Multiply(6, 4))\n```",
            "Sure! Here you go:\nQuestion(about=Multiply(6, 4))",
            "<think>hmm, a product</think>Question(about=Multiply(6, 4))",
        ]:
            heard = self.seat(wrapper).hear("what is 6 times 4", ["Multiply"])
            self.assertEqual(render(heard, multiline=False), "Question(about=Multiply(6, 4))", wrapper)

    def test_a_dropped_bracket_is_repaired(self):
        heard = self.seat("Remember(proposition=Meaning(value=AllOf(Creepy(), Dark())))").hear(
            "spooky means creepy and dark", ["Meaning"]
        )
        self.assertIsNotNone(heard)
        heard = self.seat("Question(about=Multiply(6, 4)").hear("what is 6 times 4", ["Multiply"])
        self.assertEqual(render(heard, multiline=False), "Question(about=Multiply(6, 4))")

    def test_nonsense_is_declined_rather_than_invented(self):
        self.assertIsNone(self.seat("I'm sorry, I can't help with that.").hear("x"))

    def test_the_ears_may_be_baffled_but_may_not_make_things_up(self):
        # A real failure: qwen answered "who is albert einstein" with Alice.
        seat = self.seat("Question(about=Who(subject=Alice(), name=Einstein()))")
        self.assertIsNone(seat.hear("who is albert einstein?"))
        self.assertIn("Alice", seat.last_error)

    def test_a_name_that_was_actually_said_is_allowed_through(self):
        heard = self.seat("Question(about=Identity(subject=AlbertEinstein()))").hear(
            "who is albert einstein?"
        )
        self.assertEqual(render(heard, multiline=False), "Question(about=Identity(subject=AlbertEinstein()))")

    def test_an_answer_is_taken_but_a_restatement_is_not(self):
        seat = self.seat('Capital(subject=France(), value="Paris")')
        self.assertEqual(render(seat.answer(parse("Capital(subject=France())")), False), '"Paris"')
        seat = self.seat("Capital(subject=France())")
        self.assertIsNone(seat.answer(parse("Capital(subject=France())")))

    def test_a_model_that_does_not_know_says_so(self):
        self.assertIsNone(self.seat("Unknown()").answer(parse("Age(subject=User())")))

    def test_an_unquoted_answer_is_still_an_answer(self):
        seat = self.seat("100 degrees celsius")
        self.assertEqual(render(seat.answer(parse("BoilingPoint(subject=Water())")), False), '"100 degrees celsius"')

    def test_a_question_restated_before_the_answer_is_seen_through(self):
        seat = self.seat('Capital(subject=France())\n"Paris"')
        self.assertEqual(render(seat.answer(parse("Capital(subject=France())")), False), '"Paris"')


class TestVocabularyBrief(unittest.TestCase):
    """What the model is told about the concepts soup already has."""

    def test_the_brief_groups_concepts_by_what_they_are(self):
        brief = vocabulary_brief(fresh_knowledge())
        self.assertIn("operation", brief)
        self.assertIn("Multiply", brief)
        self.assertIn("relation", brief)
        self.assertIn("IsA", brief)

    def test_the_brief_says_inventing_is_allowed(self):
        self.assertIn("invent", vocabulary_brief(fresh_knowledge()).lower())


class TestFaithfulEars(unittest.TestCase):
    """A model may name a relation you did not say. It may not add things."""

    def seat(self, reply: str) -> Seat:
        s = Seat()
        s._ask = lambda system, utterance: reply
        return s

    def test_an_invented_relation_is_allowed(self):
        # "turn down" really does mean Decrease, and the word is not there.
        heard = self.seat("Request(action=Decrease(target=Volume()))").hear(
            "turn the volume down"
        )
        self.assertIsNotNone(heard)

    def test_an_invented_person_is_not(self):
        seat = self.seat("Question(about=Who(subject=Alice(), name=Einstein()))")
        self.assertIsNone(seat.hear("who is albert einstein?"))
        self.assertIn("Alice", seat.last_error)

    def test_a_pronoun_may_be_tidied_up(self):
        heard = self.seat("Remember(proposition=Give(subject=She(), to=He()))").hear(
            "she gave him a book"
        )
        self.assertIsNotNone(heard)


class TestNounPhrasesAreThings(unittest.TestCase):
    def test_an_adjective_on_a_noun_is_not_a_gap(self):
        session = fresh()
        session.respond("the fox jumps over the lazy dog")
        result = session.last_result
        assert result is not None
        self.assertEqual([g.concept for g in result.gaps], [])


class _Oracle:
    """A stand-in for a model, so the tests never touch a network."""

    available = True

    def __init__(self, **answers: str) -> None:
        self.answers = answers
        self.asked = []

    def hear(self, utterance, vocabulary=None):
        return None

    def answer(self, expr):
        self.asked.append(render(expr, multiline=False))
        text = self.answers.get(expr.concept)
        return Lit(text) if text is not None else None


class TestAskingTheModel(unittest.TestCase):
    """When soup runs out of what it knows, the model gets the question."""

    def test_a_fact_we_lack_is_asked_rather_than_taught(self):
        oracle = _Oracle(Identity="a physicist")
        session = Session(memory_path=None, seed=3, llm=oracle)
        said = reply(session, "who is albert einstein?")
        self.assertIn("physicist", said)
        self.assertNotIn("teach me", said)

    def test_what_the_model_says_is_written_down_and_not_asked_twice(self):
        oracle = _Oracle(Capital="Paris")
        session = Session(memory_path=None, seed=3, llm=oracle)
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertEqual(len(oracle.asked), 1)

    def test_a_request_is_ours_to_carry_out_not_the_model_s(self):
        oracle = _Oracle(Make="a spooky picture")
        session = Session(memory_path=None, seed=3, llm=oracle)
        said = reply(session, "make me a sandwich")
        self.assertEqual(oracle.asked, [])
        self.assertIn("teach me", said)

    def test_an_assertion_is_remembered_not_fact_checked(self):
        oracle = _Oracle(Sat="no it did not")
        session = Session(memory_path=None, seed=3, llm=oracle)
        reply(session, "the cat sat on the mat")
        self.assertEqual(oracle.asked, [])

    def test_the_model_is_a_last_resort_not_a_first_one(self):
        oracle = _Oracle(Multiply="about thirty")
        session = Session(memory_path=None, seed=3, llm=oracle)
        self.assertIn("24", reply(session, "what is 6 times 4"))
        self.assertEqual(oracle.asked, [])

    def test_a_silent_model_leaves_the_teacher_to_it(self):
        oracle = _Oracle()
        session = Session(memory_path=None, seed=3, llm=oracle)
        said = reply(session, "what is the flumph of a grobble")
        self.assertTrue(oracle.asked)
        self.assertIn("teach me", said)


# Real payloads, trimmed. Recorded from wikidata.org so the shapes are not
# invented: see scripts/compare_ears.py for how they were obtained.
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
        # France has ten capitals on record; nine of them ended.
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
        # "height" is P2048's label and merely an alias of "elevation above
        # sea level", and they are not the same question.
        self.assertEqual(self.wiki._property("Height"), "P2048")

    def test_a_near_miss_is_refused_rather_than_answered(self):
        # The best property match for "tower" is a Tower Records artist ID.
        self.assertIsNone(self.wiki._property("Tower"))

    def test_a_multiword_noun_phrase_is_searched_as_words(self):
        self.assertEqual(_subject_text(parse("Tower(quality=Eiffel())")), "eiffel tower")
        self.assertEqual(_subject_text(parse("NewZealand()")), "new zealand")

    def test_a_question_is_not_a_name(self):
        self.assertIsNone(_subject_text(parse("Maximum(collection=Files())")))

    def test_a_record_is_preferred_to_a_guess(self):
        oracle = _Oracle(Capital="Marseille")
        session = Session(memory_path=None, seed=3, llm=oracle)
        session.realizer.lookup = _OfflineWikidata(session.knowledge)
        self.assertIn("Paris", reply(session, "what is the capital of france"))
        self.assertEqual(oracle.asked, [])

    def test_the_model_still_gets_what_wikidata_lacks(self):
        oracle = _Oracle(Vibe="unmatched")
        session = Session(memory_path=None, seed=3, llm=oracle)
        session.realizer.lookup = _OfflineWikidata(session.knowledge)
        self.assertIn("unmatched", reply(session, "what is the vibe of france"))

    def test_no_network_is_not_an_error(self):
        wiki = Wikidata(fresh_knowledge(), timeout=0.4, endpoint="http://localhost:9/w/api.php")
        session = Session(memory_path=None, seed=3)
        session.realizer.lookup = wiki
        # Nothing local resolves this, so the lookup is really attempted.
        said = reply(session, "what is the capital of france")
        self.assertFalse(wiki.available)
        self.assertNotIn("Traceback", said)
        # And the next question does not wait on the same refused socket.
        self.assertIn("24", reply(session, "what is 6 times 4"))

    def test_a_dead_server_disables_the_seat_but_not_the_ears(self):
        seat = Seat(url="http://localhost:9/api/chat", timeout=0.4)
        session = Session(memory_path=None, seed=2, llm=seat)
        self.assertIn("24", reply(session, "what is 6 times 4"))
        said = reply(session, "the cat sat on the mat")
        self.assertFalse(seat.available)
        self.assertNotIn("catch", said)

    def test_endpoint_shape_follows_the_url(self):
        self.assertTrue(Seat(url="http://h/api/chat").native)
        self.assertFalse(Seat(url="http://h/v1/chat/completions").native)
        payload = Seat(url="http://h/api/chat")._payload("s", "u")
        self.assertIs(payload["think"], False)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.session = fresh(seed=11)

    def test_greeting(self):
        self.assertTrue(reply(self.session, "hey"))

    def test_arithmetic_answers(self):
        self.assertIn("24", reply(self.session, "what is 6 times 4"))

    def test_memory_across_turns(self):
        reply(self.session, "my name is Keal")
        self.assertIn("Keal", reply(self.session, "what is my name"))

    def test_facts_and_comparison(self):
        reply(self.session, "Alice is 30 years old")
        reply(self.session, "Bob is 24 years old")
        self.assertIn(reply(self.session, "is Alice older than Bob")[:3], ("yep", "yea", "yes", "cor"))

    def test_reference_chain(self):
        reply(self.session, "what is 6 times 7")
        reply(self.session, "double it")
        self.assertIn("89", reply(self.session, "add 5 to that"))

    def test_unknown_person_is_admitted(self):
        said = reply(self.session, "how old is Carl")
        self.assertTrue("know" in said or "no idea" in said, said)

    def test_a_heard_sentence_is_never_called_unintelligible(self):
        """Not knowing the word is our problem, not a failure to hear."""
        said = reply(self.session, "what is the flumph?")
        self.assertNotIn("catch", said)
        self.assertNotIn("past me", said)
        self.assertTrue("know" in said or "no idea" in said, said)

    def test_an_unknown_verb_becomes_something_teachable(self):
        """A word we have never met is the Teacher's job, not a dead end."""
        session = fresh(seed=4)
        said = reply(session, "flumph the widget")
        self.assertNotIn("catch", said)
        self.assertNotIn("past me", said)
        self.assertIn("Flumph", said)

    def test_the_clock_answers(self):
        self.assertRegex(reply(self.session, "what time is it?"), r"\d:\d\d")
        self.assertRegex(reply(self.session, "what is the date?"), r"\d{4}")

    def test_a_capability_told_is_a_capability_answered(self):
        reply(self.session, "i can drive")
        self.assertIn(reply(self.session, "can i drive?")[:3], ("yep", "yea", "yes", "cor"))
        said = reply(self.session, "can i fly?")
        self.assertTrue("know" in said or "no idea" in said or "never told" in said, said)

    def test_a_denied_capability_stays_denied(self):
        reply(self.session, "i can not swim")
        said = reply(self.session, "can i swim?")
        self.assertIn(said[:2], ("no", "na", "nu"))

    def test_modals_read_as_english(self):
        clause = self.session.mouth.clause(parse("Can(subject=User(), action=Drive())"))
        self.assertEqual(clause, "you can drive")

    def test_how_are_you_has_more_than_one_spelling(self):
        for greeting in ["howsit?", "sup", "how is it going", "how you doing"]:
            said = reply(fresh(seed=5), greeting)
            self.assertNotIn("catch", said)
            self.assertNotIn("past me", said)

    def test_teaching_loop(self):
        asked = reply(self.session, "what is the vibe of [1,2,3]")
        self.assertIn("Vibe", asked)
        learned = reply(self.session, "vibe means the count of it")
        self.assertIn("3", learned)
        self.assertIn("2", reply(self.session, "what is the vibe of [7,8]"))

    def test_teaching_by_definition_syntax(self):
        reply(self.session, "Quadruple(x) := Multiply(x, 4)")
        self.assertIn("24", reply(self.session, "quadruple 6"))

    def test_teaching_from_english(self):
        reply(self.session, "to quintuple something means to multiply it by 5")
        self.assertIn("20", reply(self.session, "quintuple 4"))

    def test_learned_concept_survives_a_save(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "memory.json")
            first = Session(memory_path=path, seed=1)
            reply(first, "Zing(x) := Multiply(x, 7)")
            reply(first, "Alice is 30 years old")
            first.save()
            second = Session(memory_path=path, seed=1)
            self.assertIn("21", reply(second, "zing 3"))
            self.assertIn("30", reply(second, "how old is Alice"))

    def test_nonsense_does_not_become_a_definition(self):
        reply(self.session, "what is the vibe of [1,2,3]")
        reply(self.session, "asdkjh qwe zzz")
        self.assertEqual(self.session.knowledge.rules_for("Vibe"), [])

    def test_explaining_shows_the_derivation(self):
        reply(self.session, "what is 6 times 4")
        self.assertIn("24", reply(self.session, "why"))


if __name__ == "__main__":
    unittest.main()
