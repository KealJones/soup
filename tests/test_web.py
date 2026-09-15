"""Offline checks for Soup's keyless web and transcription concepts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from email.message import Message
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup.builtins import fresh_knowledge
from soup.expr import Call, Lit, Seq
from soup.parse import parse
from soup.realize import Realizer


class _Response:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


class TestWebConcepts(unittest.TestCase):
    def realize(self, source: str):
        return Realizer(fresh_knowledge()).realize(parse(source)).value

    def test_web_search_returns_concept_records_from_keyless_results(self):
        class FakeDDGS:
            def text(self, query, max_results):
                self.query = query
                self.max_results = max_results
                return [
                    {
                        "title": "Soup docs",
                        "href": "https://example.test/soup",
                        "body": "A concept soup.",
                    }
                ]

        fake = FakeDDGS()
        with patch("soup.web._ddgs_class", return_value=lambda: fake):
            out = self.realize('WebSearch(query="concept soup", max_results=3)')

        self.assertEqual(fake.query, "concept soup")
        self.assertEqual(fake.max_results, 3)
        self.assertEqual(out.concept, "SearchResults")
        self.assertEqual(out.get("query"), Lit("concept soup"))
        results = out.get("results")
        self.assertIsInstance(results, Seq)
        self.assertEqual(len(results.items), 1)
        result = results.items[0]
        self.assertEqual(result.concept, "SearchResult")
        self.assertEqual(result.get("title"), Lit("Soup docs"))
        self.assertEqual(result.get("url"), Lit("https://example.test/soup"))
        self.assertEqual(result.get("snippet"), Lit("A concept soup."))

    def test_visit_webpage_returns_sections_and_links_and_skips_script(self):
        response = _Response(
            b"<html><body><h1>Hello</h1><p>Read <a href='https://example.test'>the page</a>.</p>"
            b"<script>do not include this</script><ul><li>one</li><li>two</li></ul></body></html>"
        )
        with patch("soup.web.urllib.request.urlopen", return_value=response):
            out = self.realize('VisitWebpage(url="https://example.test")')

        self.assertEqual(out.concept, "WebPage")
        self.assertEqual(out.get("title"), Lit("Hello"))
        sections = out.get("sections")
        self.assertIsInstance(sections, Seq)
        self.assertEqual(sections.items[0].concept, "WebSection")
        paragraphs = sections.items[0].get("paragraphs")
        self.assertEqual(paragraphs.items[0], Lit("Read the page."))
        self.assertEqual(paragraphs.items[1], Lit("one"))
        self.assertEqual(paragraphs.items[2], Lit("two"))
        links = out.get("links")
        self.assertIsInstance(links, Seq)
        self.assertEqual(links.items[0].concept, "WebLink")
        self.assertEqual(links.items[0].get("text"), Lit("the page"))
        self.assertEqual(links.items[0].get("url"), Lit("https://example.test"))
        self.assertNotIn("do not include this", repr(out))

    def test_wikipedia_search_returns_all_candidates_as_concepts(self):
        responses = [
            _Response(
                json.dumps(
                    {
                        "query": {
                            "search": [
                                {
                                    "pageid": 42,
                                    "title": "Soup (software)",
                                    "snippet": "software project",
                                },
                                {"pageid": 43, "title": "Soup", "snippet": "food"},
                            ]
                        }
                    }
                ).encode(),
                "application/json; charset=utf-8",
            ),
            _Response(
                json.dumps(
                    {
                        "query": {
                            "pages": {
                                "42": {
                                    "title": "Soup (software)",
                                    "extract": "A fictional software project.",
                                }
                            }
                        }
                    }
                ).encode(),
                "application/json; charset=utf-8",
            ),
        ]
        with patch("soup.web.urllib.request.urlopen", side_effect=responses):
            out = self.realize('WikipediaSearch(query="Soup software")')

        self.assertEqual(out.concept, "SearchResults")
        results = out.get("results")
        self.assertIsInstance(results, Seq)
        self.assertEqual(len(results.items), 2)
        first = results.items[0]
        self.assertIsInstance(first, Call)
        self.assertEqual(first.get("title"), Lit("Soup (software)"))
        self.assertEqual(first.get("snippet"), Lit("A fictional software project."))
        self.assertEqual(
            first.get("url"), Lit("https://en.wikipedia.org/wiki/Soup_(software)")
        )
        self.assertEqual(results.items[1].get("title"), Lit("Soup"))
        self.assertEqual(results.items[1].get("snippet"), Lit("food"))

    def test_audio_transcription_uses_local_mlx_whisper_when_available(self):
        calls = []

        def transcribe(path, **kwargs):
            calls.append((path, kwargs))
            return {"text": "hello from audio"}

        fake_whisper = types.SimpleNamespace(transcribe=transcribe)
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with patch.dict(sys.modules, {"mlx_whisper": fake_whisper}):
                out = self.realize('TranscribeAudio(path="%s")' % audio.name)

        self.assertEqual(out.value, "hello from audio")
        self.assertEqual(calls[0][0], audio.name)
        self.assertEqual(calls[0][1]["path_or_hf_repo"], "mlx-community/whisper-large-v3-turbo")


if __name__ == "__main__":
    unittest.main()
