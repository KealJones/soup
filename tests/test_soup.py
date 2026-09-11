"""Tests for soup. Plain unittest, no dependencies.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Session, fresh_knowledge, parse, parse_definition, render
from soup.expr import Arg, Call, Lit, Seq, Var, call, match, substitute
from soup.knowledge import Evidence, Knowledge, Relation
from soup.realize import Realizer


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
        k.relate("Sprint", Relation.IsA, "Run")
        self.assertTrue(k.is_a("Sprint", "Run"))
        self.assertFalse(k.is_a("Run", "Sprint"))


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

    def test_gibberish_is_admitted_not_guessed(self):
        self.assertIn("catch", reply(self.session, "asdkjh qwe zzz").lower() + "x")


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
