"""The chat loop."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .expr import Call, render
from .lookup import Wikidata
from .seat import Seat, seat_from_env
from .session import DEFAULT_MEMORY, Reply, Session

__all__ = ["main", "run_repl"]

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else text


DIM = lambda s: _c(s, "2")
BOLD = lambda s: _c(s, "1")
CYAN = lambda s: _c(s, "36")
GREEN = lambda s: _c(s, "32")
YELLOW = lambda s: _c(s, "33")

BANNER = """%s
  messy english -> concepts -> realization -> english
  type %s for commands, %s to leave
""" % (
    BOLD("soup"),
    CYAN(":help"),
    CYAN(":q"),
)

HELP = """
  :ears              the concept expression from your last message
  :answer            the concept structure soup answered with
  :why               how it got there
  :know              what soup knows, in numbers
  :concepts [word]   list concepts (optionally filtered)
  :facts [word]      list remembered facts
  :rules             list realizations soup has been taught
  :show              show ears + answer for every turn from now on
  :quiet             stop doing that
  :save              write memory to disk
  :wipe              forget everything learned this session and reseed
  :help  :q
"""


def run_repl(session: Session, show: bool = False) -> int:
    print(BANNER)
    while True:
        try:
            line = input(BOLD("you ") + DIM("> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            command, _, rest = line[1:].partition(" ")
            if command in ("q", "quit", "exit"):
                break
            if command in ("show", "debug"):
                show = True
                print(DIM("  showing concept structures"))
                continue
            if command == "quiet":
                show = False
                continue
            handled = _command(session, command, rest.strip())
            if not handled:
                print(DIM("  unknown command, try :help"))
            continue

        reply = session.respond(line)
        if show:
            _dump(reply)
        print(CYAN("soup") + DIM(" > ") + reply.said)
        print()

    path = session.save()
    if path:
        print(DIM("  memory saved to %s" % path))
    return 0


def _dump(reply: Reply) -> None:
    if reply.heard is not None:
        print(DIM("  ears  "), GREEN(render(reply.heard.expr, multiline=False)))
    if reply.result is not None:
        print(DIM("  soup  "), YELLOW(render(reply.result.value, multiline=False)))
        for gap in reply.result.gaps:
            print(DIM("  gap   "), gap.signature())


def _command(session: Session, command: str, argument: str) -> bool:
    k = session.knowledge
    turns = session.discourse.turns

    if command == "help":
        print(HELP)
        return True

    if command == "ears":
        last = _last_meaning(session)
        print(render(last, multiline=True) if last is not None else DIM("  nothing yet"))
        return True

    if command == "answer":
        if session.last_result is None:
            print(DIM("  nothing yet"))
            return True
        print(render(session.last_result.value, multiline=True))
        return True

    if command in ("why", "trace"):
        if session.last_result is None or not session.last_result.trace:
            print(DIM("  no steps to show"))
            return True
        for step in session.last_result.trace:
            print("  " + step)
        for effect in session.last_result.effects:
            print(DIM("  (%s)" % effect))
        return True

    if command == "know":
        print("  " + k.summary())
        return True

    if command == "concepts":
        names = sorted(k.concepts)
        if argument:
            names = [n for n in names if argument.lower() in n.lower()]
        _columns(names)
        return True

    if command == "facts":
        shown = 0
        for fact in k.facts:
            line = render(fact.proposition, multiline=False)
            if argument and argument.lower() not in line.lower():
                continue
            mark = "" if fact.truth else DIM(" (false)")
            print("  %s%s %s" % (line, mark, DIM("[%s]" % fact.evidence.source)))
            shown += 1
        if not shown:
            print(DIM("  nothing remembered yet"))
        return True

    if command == "rules":
        any_shown = False
        for bucket in k.rules.values():
            for rule in bucket:
                print("  %s %s" % (rule.source_text(), DIM("[%s]" % rule.evidence.source)))
                any_shown = True
        if not any_shown:
            print(DIM("  no realizations yet"))
        return True

    if command == "save":
        path = session.save()
        print(DIM("  saved to %s" % path if path else "  memory is off"))
        return True

    if command == "wipe":
        from .builtins import fresh_knowledge

        session.knowledge = fresh_knowledge()
        session.realizer.knowledge = session.knowledge
        session.ears.knowledge = session.knowledge
        session.teacher.knowledge = session.knowledge
        session.mouth.knowledge = session.knowledge
        del turns[:]
        print(DIM("  wiped"))
        return True

    return False


def _last_meaning(session: Session):
    for turn in reversed(session.discourse.turns):
        if turn.meaning is not None:
            return turn.meaning
    return None


def _columns(names: List[str], width: int = 4) -> None:
    if not names:
        print(DIM("  none"))
        return
    size = max(len(n) for n in names) + 2
    row: List[str] = []
    for name in names:
        row.append(name.ljust(size))
        if len(row) == width:
            print("  " + "".join(row).rstrip())
            row = []
    if row:
        print("  " + "".join(row).rstrip())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="soup", description="talk to soup")
    parser.add_argument("message", nargs="*", help="say one thing and exit")
    parser.add_argument(
        "--memory",
        default=DEFAULT_MEMORY,
        help="where to keep what soup learns (default: %s)" % DEFAULT_MEMORY,
    )
    parser.add_argument("--no-memory", action="store_true", help="do not read or write memory")
    parser.add_argument("--show", action="store_true", help="print concept structures as you go")
    parser.add_argument(
        "--llm",
        nargs="?",
        const="",
        metavar="MODEL",
        help="hand sentences the constructions miss to a local model "
        "(OpenAI-compatible endpoint, $SOUP_LLM_URL, default ollama)",
    )
    parser.add_argument(
        "--wikidata",
        action="store_true",
        help="look facts up in wikidata, learning the concepts as it goes",
    )
    args = parser.parse_args(argv)

    seat = seat_from_env()
    if args.llm is not None:
        seat = seat or Seat()
        if args.llm:
            seat.model = args.llm

    session = Session(memory_path=None if args.no_memory else args.memory, llm=seat)
    if args.wikidata:
        session.realizer.lookup = Wikidata(session.knowledge)

    if args.message:
        reply = session.respond(" ".join(args.message))
        if args.show:
            _dump(reply)
        print(reply.said)
        session.save()
        return 0

    return run_repl(session, show=args.show)
