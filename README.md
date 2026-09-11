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
| `ears.py` | English to concepts: a construction grammar with typed, backtracking slots, then word order |
| `seat.py` | optional local model, tried only for the sentences constructions miss |
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

### Nothing is unintelligible

The ears always return structure. If no construction matches, word order is
still there to be read, so a noun phrase, a verb and its prepositions come
back as a concept expression whatever the words happen to be:

```
the cat sat on the mat   ->  Remember(proposition=Sat(subject=Cat(), on=Mat()))
she gave him a book      ->  Remember(proposition=Gave(subject=Her(), recipient=Him(), object=Book()))
asdkjh qwe zzz           ->  Remember(proposition=Qwe(subject=Asdkjh(), object=Zzz()))
```

Prepositions make the argument names, which keeps the shape of the original
sentence. A verb nobody has ever heard of is not a failure, it is a concept
with a signature, and that is precisely what the Teacher asks about.

### The seat

Constructions are fast, free and never confidently wrong, but they only cover
sentences somebody wrote a pattern for. A *seat* is somewhere a model can sit
and do the one job it is plainly better at: reading a sentence nobody
anticipated.

```
./chat --llm
./chat --llm qwen3.5:4b
SOUP_LLM_URL=http://localhost:1234/v1/chat/completions ./chat --llm
```

It is tried only for what the constructions miss, so the common path never
waits on a model, and it is held to the same contract as the rest of the
ears: return a concept expression, nothing else. It does not answer the
question and it does not decide how anything gets done.

```
you  > make this picture look spooky but still cute
       Request(action=Make(target=Ref("this"), look=AllOf(Spooky(), Cute())))
soup > i don't know Make. teach me: Make(target=target, look=look) := ...

you  > what is the airspeed velocity of an unladen swallow
       Question(about=Query(pattern=AirspeedVelocity(subject=UnladenSwallow())))
soup > i don't have unladen swallow's airspeed velocity
```

Ollama's native `/api/chat` is the default because it is the only endpoint of
the two that can actually turn thinking off; through the OpenAI-compatible
one a reasoning model spends fifty seconds on a single line of syntax instead
of two. Any OpenAI-shaped server works, it is all stdlib `urllib`, and no
server running just means no seat.

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
department behind it. Constructions cover arithmetic, collections, attributes
of people and things, comparatives, wh-questions, yes/no questions with
subject-aux inversion, imperatives, modals, pronouns across turns, and
definitions. Past that it falls back to word order, which reads a plain
transitive sentence well and gets steadily vaguer as the grammar gets harder:
relative clauses, tense and coordination inside an argument all come back
flatter than they went in. Turning the seat on is the answer to that, and the
honest reason it exists.

Nothing is reported as misheard any more, which cuts both ways: Soup will now
hand you a structure for a sentence it understood only loosely. Confidence on
`Heard` is the thing to watch, not the presence of a parse.

Distinguishing the kinds of not-knowing still matters. A sentence Soup parsed
but cannot answer gets "i don't know what the time is"; blaming the speaker
for a hole in your own vocabulary is the most annoying thing a program can
do.

The clock is the one thing Soup reads from outside its own memory, because
"i don't know what time it is" is a silly answer.

Memory is closed-world and flat. "does Bob like Alice" answers "i don't know"
rather than "no", which is correct but chattier than people expect.
