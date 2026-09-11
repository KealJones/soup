# soup

A small conversational system that thinks in concepts instead of tokens. No
model, no API key, no network, no dependencies. Just Python.

```
you  > Alice is 30 years old
soup > noted
you  > Bob is 24 years old
soup > got it
you  > is Alice older than Bob
soup > yeah, Alice is older than Bob
you  > what is the vibe of [1,2,3]
soup > i don't know Vibe. teach me: Vibe(collection) := ...
you  > vibe means the count of it
soup > learned vibe. so: 3
you  > what is the vibe of [7,8]
soup > it's 2
```

## Running it

Needs Python 3.9 or newer. Nothing else.

```sh
./chat                          # talk to it
./chat --show                   # talk to it, printing the concept structures
./chat "what is 6 times 7"      # say one thing and quit
./chat --no-memory              # do not read or write data/memory.json

python3 -m unittest discover -s tests
python3 scripts/sweep.py        # talk at it with 50-odd utterances at once
```

In the chat, `:help` lists the commands. The useful ones are `:ears` (the
concept expression your last message became), `:why` (how it got the answer),
`:rules` (realizations it has been taught) and `:facts`.

## The idea

Language is one representation of meaning, not the medium of thought. So the
English goes away as early as possible and comes back as late as possible.

```
                 "if greg has the car he probably left"
                                  |
                                EARS
                                  |
                                  v
          Conditional(condition=Has(subject=Greg, object=Car),
                      then=Probably(Left(subject=Greg)))
                                  |
                             REALIZATION            concepts in,
                    code | taught rules | memory    concepts out
                                  |
                    +-------------+-------------+
                    |                           |
             everything resolved          something didn't
                    |                           |
                    v                           v
             concept structure               TEACHER
                    |                  "teach me MostAdorable"
                  MOUTH                        |
                    |                    new realization
                    v                          |
              "yeah, probably" <---------------+
```

Three claims hold the thing together.

**The ears stop at meaning.** They never decide how Soup accomplishes
anything. `"double every number in this list"` becomes a structure mentioning
`Map`, `Double`, `2` and the referenced collection. Whether any of that is
executable is somebody else's problem.

**Concept expressions look like Python calls** because that buys binding,
arguments, nesting and composition for nothing. `Double(value=Last(Number(...)))`
preserves that the thing found by one expression is the argument to another,
which two extracted "intents" would have thrown away. But `Spooky()` is an
expression of meaning first and a function only if something realizes it.

**A realization is concepts in, concepts out.** Sometimes that is Python
(`Multiply`). Sometimes it is a rewrite you were taught
(`Double(x) := Multiply(x, 2)`). Sometimes it is a lookup in what you have been
told (`Age(subject=Alice)`). They compose, so a learned concept immediately
becomes material for the next one. Tools terminate in effects; these can
terminate in more thought.

The payoff is the failure mode. Because the ears already said exactly which
concept failed and the shape it was used in, not knowing something is
surgical:

```
you  > find the most adorable pokemon
soup > i don't know MostAdorable.
       teach me: MostAdorable(collection) := ...
```

That is a much better question than "what does this whole sentence mean".

## How it is put together

| module | job |
| --- | --- |
| `expr.py` | the concept expression type, plus rendering, matching, substitution |
| `parse.py` | reads the Python-ish syntax, including `Head(x) := Body` |
| `knowledge.py` | concepts, relations, remembered facts, taught rules, persistence |
| `realize.py` | resolves an expression as far as it can and records what it could not |
| `builtins.py` | the starting vocabulary and the realizations that bottom out in code |
| `ears.py` | English to concepts: a construction grammar with typed, backtracking slots |
| `mouth.py` | concepts to casual English |
| `teacher.py` | turns a gap into a question, and an answer into a realization |
| `discourse.py` | what "it" and "that" currently point at |
| `session.py` | the loop that wires all of the above together |

### Not knowing, in three flavours

Soup distinguishes them, because they call for different replies.

- **A name it has not heard.** `Carl()` resolves to itself. Names cannot fail
  to evaluate.
- **A thing it was never told.** `Age(subject=Carl)` becomes
  `Unknown(about=...)`, which propagates up so the answer names the thing that
  was actually missing rather than the outermost operation. Mouth says "i
  don't have Carl's age".
- **A word it does not have a meaning for.** `MostAdorable(collection=...)`
  records a `Gap`, and the Teacher asks about exactly that concept with
  exactly that signature.

An utterance where *every* content word is a mystery is none of the above. That
is not a concept to be taught, it is a sentence Soup failed to hear, and it
says so.

### Relations are concepts too

There is no enum of relation kinds. `IsA`, `PartOf` and `OppositeOf` are
concepts like any other, and how far a walk over them travels is decided by
properties asserted about them rather than by code:

```
you  > Remember(proposition=IsA(subject=RhymesWith(), kind=Symmetric()))
soup > okay, rhymes with is symmetric
```

After that, `related("Fig", "RhymesWith")` finds `Big`, where a moment earlier
it found nothing. Marking a relation `Taxonomic` likewise puts it to work in
`is_a` and in realization inheritance, so a relation nobody anticipated can
carry a taxonomy.

`knowledge.py` knows only three property names, `Symmetric`, `Transitive` and
`Taxonomic`, plus the two edges that attach them. Everything else about how
relations behave is asserted knowledge, editable at runtime and saved with the
rest of memory.

### Teaching it things

Three ways, all equivalent once they land:

```
you  > MostAdorable(items) := Maximum(collection=items, by=Adorableness())
you  > to quadruple something means to multiply it by 4
you  > vibe means the count of it
```

Taught rules go in `data/memory.json` next to the facts, so they survive
restarts. `:rules` shows them.

## Limits, honestly

The grammar is a construction grammar, not a parser with a linguistics
department behind it. It handles arithmetic, collections, attributes of
people and things, comparatives, wh-questions, yes/no questions with
subject-aux inversion, imperatives, modals, pronouns across turns, and
definitions. It will not handle relative clauses, coordination inside
arguments, tense, or most of the ways a real sentence can go. When it misses,
it says so rather than guessing, which is the tradeoff: a model-backed ears
module would understand far more and be confidently wrong far more often.

Saying so accurately matters more than it sounds. A sentence Soup parsed but
cannot answer gets "i don't know what the time is"; only a sentence it truly
could not parse gets "i didn't catch that". Blaming the speaker for a hole in
your own vocabulary is the most annoying thing a program can do.

The clock is the one thing Soup reads from outside its own memory, because
"i don't know what time it is" is a silly answer.

Memory is closed-world and flat. "does Bob like Alice" answers "i don't know"
rather than "no", which is correct but chattier than people expect.
