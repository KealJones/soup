"""Scoring Soup (or any mouth) on MMLU-Pro letters. No model."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup.bench import extract_choice, format_item, grade


class TestMmluProScoring(unittest.TestCase):
    def test_a_letter_answer_is_taken(self):
        self.assertEqual(extract_choice("the answer is (B).", ["no", "yes"]), "B")
        self.assertEqual(extract_choice("Answer: H", ["x"] * 10), "H")

    def test_matching_option_text_counts(self):
        options = ["Multiply 5 by 5 to find 25 teams.", "Divide 30 by 5 to find 6 teams."]
        self.assertEqual(extract_choice("divide 30 by 5 to find 6 teams.", options), "B")

    def test_a_short_option_is_not_hunted_in_noise(self):
        options = ["0", "30", "3", "10"]
        self.assertIsNone(
            extract_choice(
                "i don't have them (worked out Ring(target=target) := Range(start=0, end=100))",
                options,
            )
        )
        self.assertIsNone(extract_choice("i don't know Characteristic. teach me: Characteristic := ...", ["0", "2"]))
        self.assertIsNone(extract_choice("that went past me", ["0", "2"]))

    def test_leading_letter_with_option_text_counts(self):
        options = ["Multiply 5 by 5 to find 25 teams.", "Divide 30 by 5 to find 6 teams."]
        self.assertEqual(
            extract_choice("B. Divide 30 by 5 to find 6 teams.", options),
            "B",
        )

    def test_grade_is_strict(self):
        item = {"answer": "B", "options": ["wrong", "right"]}
        self.assertTrue(grade(item, "B"))
        self.assertFalse(grade(item, "A"))
        self.assertFalse(grade(item, None))

    def test_the_prompt_lists_every_option(self):
        text = format_item(
            {
                "question": "How many teams?",
                "options": ["25", "6"],
            }
        )
        self.assertIn("How many teams?", text)
        self.assertIn("A. 25", text)
        self.assertIn("B. 6", text)


if __name__ == "__main__":
    unittest.main()
