"""An optional LLM seat for the ears.

Soup's constructions are fast, free and never confidently wrong, but they
only cover the sentences somebody thought to write a pattern for. A seat is
somewhere a model can sit and do the one job it is unambiguously better at:
reading a sentence nobody anticipated and saying what shape it is.

The seat is held to the same contract as the rest of the ears. It returns a
concept expression and nothing else. It does not get to decide how anything
is accomplished, it does not answer the question, and a concept it invents is
not a failure: an unknown concept is exactly what the Teacher exists for.

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
import urllib.error
import urllib.request
from typing import List, Optional

from .expr import Expr
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
  Greeting() Farewell() Thanks() Affirm() Deny()

RULES
1. Say what the sentence MEANS, never how to accomplish it. "delete the
   biggest file" is Delete(target=Maximum(collection=Files(), by=FileSize())),
   never a loop or a sequence of steps.
2. If you do not know a concept name, invent a clear one. Inventing
   MostAdorable() is correct and useful. Do not substitute something you do
   know but that means something else.
3. Keep the structure of the original sentence. Prepositions make good
   argument names: "the cat sat on the mat" is
   Remember(proposition=Sat(subject=Cat(), on=Mat())).
4. Use Ref("it") / Ref("that") for pronouns pointing at earlier turns.

EXAMPLES
what is 6 times 4
Question(about=Multiply(6, 4))

make this picture look spooky but still cute
Request(action=Make(target=Ref("this"), look=AllOf(Spooky(), Cute())))

Alice is 30 years old
Remember(proposition=Age(subject=Alice(), value=30))

who has the car
Question(about=Query(pattern=Has(subject=Who(), object=Car())))

delete the biggest file in my downloads folder
Request(action=Delete(target=Maximum(collection=AllOf(kind=File(), within=Downloads()), by=FileSize())))

why is the sky blue
Question(about=Why(proposition=Is(subject=Sky(), value=Blue())))

spooky means creepy and dark
Teach(concept=Spooky(), meaning=AllOf(Creepy(), Dark()))

double it
Request(action=Double(Ref("it")))'''


class Seat:
    """A model sitting in the ears, used only for what constructions miss."""

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

    def hear(self, utterance: str, vocabulary: Optional[List[str]] = None) -> Optional[Expr]:
        if not self.available:
            return None
        prompt = _SYSTEM
        if vocabulary:
            prompt += "\n\nCONCEPTS ALREADY KNOWN (prefer these when they fit):\n"
            prompt += ", ".join(sorted(vocabulary)[:220])
        reply = self._ask(prompt, utterance)
        if reply is None:
            return None
        try:
            return parse(_clean(reply))
        except (ParseError, ValueError, IndexError):
            self.last_error = "unparseable: %s" % reply[:120]
            return None

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


def _clean(reply: str) -> str:
    """Models like to wrap things in fences and add a sentence of pride."""
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
