"""Say a lot of things to Soup at once and look at what comes back.

    python3 scripts/sweep.py [--quiet]

Not a test suite. This is the "does it feel like talking to something" check.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Session
from soup.expr import render

SCRIPT = [
    "hey",
    "what's up",
    "who are you",
    "what can you do",
    "my name is Keal",
    "what is my name",
    "i am 34 years old",
    "how old am i",
    "what is 6 times 4",
    "what's 100 divided by 8",
    "what is 2 plus 3 times 4",
    "double 21",
    "double it",
    "add 5 to that",
    "what is half of that",
    "square 7",
    "what is the sum of [1,2,3,4]",
    "what is the average of [2,4,9]",
    "how many are in [1,2,3]",
    "sort [5,1,4]",
    "reverse those",
    "double every number in [1,2,3]",
    "what is the biggest of [3,9,4]",
    "Alice is 30 years old",
    "Bob is 24 years old",
    "is Alice older than Bob",
    "is Bob older than Alice",
    "who is Alice",
    "how old is Alice",
    "how old is Carl",
    "Greg has the car",
    "who has the car",
    "does Greg have the car",
    "Alice likes Bob",
    "does Bob like Alice",
    "Alice lives in Paris",
    "where is Alice",
    "what do you know about Alice",
    "remember that Bob has a bike",
    "who has a bike",
    "to quadruple something means to multiply it by 4",
    "quadruple 6",
    "what is the vibe of [1,2,3]",
    "vibe means the count of it",
    "what is the vibe of [7,8]",
    "TallerThan(a, b) := GreaterThan(left=Height(subject=a), right=Height(subject=b))",
    "Alice is 170 tall",
    "Bob is 180 tall",
    "is Bob taller than Alice",
    "what does double mean",
    "explain",
    "thanks",
    "asdkjh qwe zzz",
    "bye",
]


def main() -> int:
    quiet = "--quiet" in sys.argv
    session = Session(memory_path=None, seed=7)
    odd = 0
    for line in SCRIPT:
        reply = session.respond(line)
        flag = ""
        said = reply.said.lower()
        if "didn't catch" in said or "went past me" in said or "couldn't turn" in said:
            flag = "  <-- not understood"
            odd += 1
        elif "i don't know" in said or "no idea" in said or "never told me" in said:
            flag = "  <-- unresolved"
        print("you  > %s" % line)
        if not quiet and reply.heard is not None:
            print("      %s" % render(reply.heard.expr, multiline=False))
        print("soup > %s%s" % (reply.said, flag))
        print()
    print("%d of %d utterances were not understood at all" % (odd, len(SCRIPT)))
    print(session.knowledge.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
