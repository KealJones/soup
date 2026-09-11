"""Constructions against the model, on the same sentences.

Prints only the disagreements, because the agreements tell you nothing. The
question this is here to answer is not "does the model parse" but "when the
two of them differ, which one would you rather have shipped".

    python3 scripts/compare_ears.py            # the sweep's sentences
    python3 scripts/compare_ears.py --all      # agreements too
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import Session, fresh_knowledge  # noqa: E402
from soup.expr import render  # noqa: E402
from soup.seat import Seat  # noqa: E402

SENTENCES = [
    # the ones that needed a hand-written patch to work
    "who is albert einstein",
    "how tall is the eiffel tower",
    "what is the capital of new zealand",
    "what is the population of new york city",
    "how tall is mount everest",
    "the fox jumps over the lazy dog",
    # ordinary traffic the constructions are supposed to be good at
    "what is 6 times 4",
    "the cat sat on the mat",
    "alice is 30 years old",
    "how old is alice",
    "double it",
    "spooky means creepy and dark",
    "hello",
    "thanks",
    "what time is it",
    "i can drive",
    "can i drive",
    # things nobody wrote a pattern for
    "make this picture look spooky but still cute",
    "turn the volume down a bit",
    "she gave him a book yesterday",
    "what is the airspeed velocity of an unladen swallow",
    "remind me to call my mum when i get home",
    "is the sky blue",
    "why is the sky blue",
    "delete the biggest file in my downloads folder",
    "sort these by size and show me the top three",
]


def main() -> int:
    show_all = "--all" in sys.argv

    plain = Session(memory_path=None, seed=1)

    seat = Seat()
    seat.learn_vocabulary(fresh_knowledge())
    if not seat.available:
        print("no model reachable at %s" % seat.url)
        return 1

    print("model: %s at %s" % (seat.model, seat.url))
    print("brief: %d characters of vocabulary\n" % len(seat.brief))

    agreed = differed = refused = 0
    spent = 0.0

    for text in SENTENCES:
        heard = plain.ears.listen(text)
        theirs = render(heard.expr, multiline=False)
        pattern = heard.construction or "fallback"

        start = time.time()
        guess = seat.hear(text, list(plain.knowledge.concepts))
        spent += time.time() - start

        if guess is None:
            refused += 1
            print("%s\n  ears  %s\n  model refused: %s\n" % (text, theirs, seat.last_error))
            continue

        mine = render(guess, multiline=False)
        if mine == theirs:
            agreed += 1
            if show_all:
                print("%s\n  both  %s\n" % (text, mine))
            continue

        differed += 1
        print("%s\n  ears  %-70s [%s]\n  model %s\n" % (text, theirs, pattern[:40], mine))

    total = len(SENTENCES)
    print("%d sentences: %d agreed, %d differed, %d refused" % (total, agreed, differed, refused))
    print("%.1fs of model time, %.2fs per sentence" % (spent, spent / max(total, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
