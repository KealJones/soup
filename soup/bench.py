"""Run Soup against a public multiple-choice bench. Scoring is letters, not vibes.

MMLU-Pro is the dataset. This module is the exam conditions: how we show a
question, how we read a letter out of English, how we record a row. Frontier
numbers from other labs used different conditions; ours are whatever Soup
does when you just ask it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, List, Optional

from .expr import render

__all__ = [
    "LETTERS",
    "extract_choice",
    "format_item",
    "grade",
    "load_split",
    "fetch_validation",
    "fetch_split",
    "RecordingSeat",
    "ReplaySeat",
]

LETTERS = "ABCDEFGHIJ"

_ABSTAIN = (
    "i don't know",
    "i do not know",
    "teach me",
    "didn't catch",
    "did not catch",
    "went past me",
    "couldn't turn",
    "could not turn",
    "no idea",
    "never told me",
)

_LETTER_PATTERNS = (
    re.compile(r"(?:the\s+)?answer\s+is\s*\(?([A-J])\)?", re.I),
    re.compile(r"answer\s*:\s*\(?([A-J])\)?", re.I),
    # Case-insensitive: "a. 0" is a pick, not a shrug. Every system being
    # scored here gets read the same way, generously.
    re.compile(r"^\s*\(?([A-Ja-j])\)?[.)]\s+\S"),
    re.compile(r"\(([A-Ja-j])\)\s*$"),
    re.compile(r"^\s*([A-Ja-j])[.)]?\s*$"),
)


def format_item(item: dict) -> str:
    """English Soup can hear: the question, then A. ... J."""
    lines = [item["question"].strip(), ""]
    for i, option in enumerate(item.get("options") or ()):
        if not option or option == "N/A":
            continue
        lines.append("%s. %s" % (LETTERS[i], option))
    return "\n".join(lines).strip()


def extract_choice(said: str, options: Iterable[str]) -> Optional[str]:
    """A letter, or nothing. Abstaining is not a guess."""
    text = (said or "").strip()
    if not text:
        return None
    lower = text.lower()
    if any(marker in lower for marker in _ABSTAIN):
        return None
    for pattern in _LETTER_PATTERNS:
        found = pattern.search(text)
        if found:
            return found.group(1).upper()
    opts = list(options)
    for i, option in enumerate(opts):
        if not option or option == "N/A":
            continue
        needle = option.strip()
        if len(needle) < 4:
            # "0" matches every "start=0". Too short to hunt for in prose.
            continue
        if needle.lower() in lower:
            return LETTERS[i]
    # A lone letter somewhere near the end, last resort.
    tail = re.findall(r"\b([A-J])\b", text[-80:])
    if len(tail) == 1:
        return tail[0].upper()
    return None


def grade(item: dict, choice: Optional[str]) -> bool:
    gold = (item.get("answer") or "").strip().upper()
    return choice is not None and choice == gold


def fetch_validation(path: str) -> List[dict]:
    """The 70-item official validation split, cached on disk."""
    return fetch_split(path, "validation", expected=70)


def fetch_split(path: str, split: str, expected: int = 0) -> List[dict]:
    """A TIGER-Lab/MMLU-Pro split, cached. Pages of 100; the test set is 12032.

    Writes the cache after every page so a 429 is a resume, not a restart.
    """
    items: List[dict] = []
    if os.path.isfile(path):
        with open(path) as fh:
            items = json.load(fh)
        if expected and len(items) >= expected:
            return items
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    offset = len(items)
    page = 100
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=TIGER-Lab/MMLU-Pro&config=default&split=%s"
            "&offset=%d&length=%d" % (split, offset, page)
        )
        payload = _get_json(url)
        rows = payload.get("rows") or []
        if not rows:
            break
        for row in rows:
            items.append(dict(row["row"]))
        offset = len(items)
        total = payload.get("num_rows_total") or expected
        print("  cached %s %d/%s" % (split, len(items), total), file=sys.stderr)
        with open(path, "w") as fh:
            json.dump(items, fh)
        if expected and len(items) >= expected:
            break
        if total and len(items) >= total:
            break
        if len(rows) < page:
            break
        time.sleep(0.4)
    return items


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "soup-bench/0.1"})
    delay = 5.0
    last = None
    for _ in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            print("  hf %s, retry in %.0fs" % (exc.code, delay), file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise last  # type: ignore[misc]


def load_split(path: str) -> List[dict]:
    with open(path) as fh:
        return json.load(fh)


class RecordingSeat:
    """A real seat that writes down every exchange it had.

    A test that hands Soup a concept expression somebody typed by hand is
    only testing Soup against a sentence the ears might never produce. The
    useful fixture is what the models actually said, so this sits in front
    of a real seat, passes everything through, and keeps the transcript.

    Everything it does not wrap is delegated, so it is a seat wherever a
    seat is expected.
    """

    def __init__(self, seat) -> None:
        self.seat = seat
        self.transcript: List[dict] = []

    def __getattr__(self, name):
        return getattr(self.seat, name)

    def _log(self, channel: str, asked: str, replied) -> None:
        self.transcript.append(
            {
                "channel": channel,
                "asked": asked,
                "replied": None if replied is None else render(replied, False),
            }
        )

    def hear(self, utterance, vocabulary=None):
        out = self.seat.hear(utterance, vocabulary)
        self._log("hear", utterance, out)
        return out

    def define(self, signature, vocabulary, context=None):
        out = self.seat.define(signature, vocabulary, context=context)
        self._log("define", _key(signature, context), out)
        return out

    def answer(self, expr):
        out = self.seat.answer(expr)
        self._log("answer", render(expr, False), out)
        return out

    def property_names(self, concept, subject=None, specifically=None):
        out = self.seat.property_names(concept, subject, specifically)
        self._log("property_names", concept, None)
        return out


class ReplaySeat:
    """The recorded transcript, served back. No network, no model.

    Same three questions, same answers the real models gave, so a test
    exercises the whole pipe over real hearings. Anything the recording does
    not cover comes back as a refusal, which is what a model that does not
    know looks like anyway.
    """

    available = True

    def __init__(self, transcript: Iterable[dict]) -> None:
        self.replies: dict = {}
        for row in transcript:
            self.replies.setdefault((row["channel"], row["asked"]), []).append(row["replied"])
        self.misses: List[tuple] = []
        self.last_error = ""

    def _take(self, channel: str, asked: str):
        from .parse import ParseError, parse

        queue = self.replies.get((channel, asked))
        if not queue:
            self.misses.append((channel, asked))
            self.last_error = "nothing recorded for %s %s" % (channel, asked)
            return None
        text = queue[0] if len(queue) == 1 else queue.pop(0)
        if text is None:
            return None
        try:
            return parse(text)
        except (ParseError, ValueError, IndexError):
            return None

    def hear(self, utterance, vocabulary=None):
        return self._take("hear", utterance)

    def define(self, signature, vocabulary, context=None):
        return self._take("define", _key(signature, context))

    def answer(self, expr):
        return self._take("answer", render(expr, False))

    def property_names(self, concept, subject=None, specifically=None):
        return []


def _key(signature: str, context: Optional[str]) -> str:
    return signature if not context or context == signature else "%s  in  %s" % (signature, context)
