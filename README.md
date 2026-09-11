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

It is held to the same contract as the rest of the ears: return a concept
expression, nothing else. It does not decide how anything gets done.

Which reading wins is settled by measurement, not taste. `scripts/compare_ears.py`
runs both over the same sentences and prints only the disagreements:

- Where a construction recognised **actual words**, it wins. It is instant,
  free, and the model does not improve on it. `is the sky blue` really is
  better as `Ask(proposition=...)` than the model's `Question(about=...)`.
- Where nothing matched, or where the winner was a **catch-all that
  recognised no words at all** and merely imposed a shape, the model wins,
  and not narrowly. The constructions turn `spooky means creepy and dark`
  into `Dark(quality=[Spooky(), Means(), Creepy(), And()])`; the model gets
  `Teach(concept=Spooky(), meaning=AllOf(Creepy(), Dark()))`.

So the catch-alls are no longer the last line of defence, and the answer to
a badly parsed sentence is no longer a new hand-written pattern.

The prompt is given the concept vocabulary grouped by kind, which is what
stops a model inventing `Times` and `HowOld` next to the `Multiply` and
`Age` already sitting there.

The ears are allowed to be baffled. They are not allowed to make things up,
and a small model asked "who is albert einstein" will genuinely hand back a
concept named `Alice`. The line drawn is between naming and inventing: a
model may name a relation you never said, because reading "turn the volume
down" as `Decrease(target=Volume())` is the paraphrase we want from it, but
it may not introduce a *thing* that was never mentioned. Heads are free;
leaves are checked against what was actually spoken.

The ears are allowed to be baffled. They are not allowed to make things up,
and a small model asked "who is albert einstein" will genuinely hand back a
concept named `Alice`. So every name the model produces has to be one we
already know, one of our own structural wrappers, or a word the speaker
actually said. Anything else is thrown out and the sentence falls through to
the ordinary structural parse.

```
you  > make this picture look spooky but still cute
       Request(action=Make(target=Ref("this"), look=AllOf(Spooky(), Cute())))
soup > i don't know Make. teach me: Make(target=target, look=look) := ...
```

### Asking, as opposed to guessing

The same model gets a second, separate job. There are two ways to not know
something, and only one of them is a language problem:

- *I don't know that word.* A hole in the vocabulary. Only you can fill it,
  so the Teacher asks you.
- *I don't know that fact.* A hole in the world. Anyone could fill it, and a
  model is right there.

Soup used to send both to you, which is how "who is albert einstein" turned
into a request to be taught the verb *to be*. Now a question that nothing
known can resolve goes outside, and the answer is written into the store as
an ordinary fact with its source recorded:

```
you  > who is albert einstein?
       Question(about=Identity(subject=AlbertEinstein()))
soup > a german-born physicist who came up with relativity

you  > :facts
       Identity(subject=AlbertEinstein(), value="a german-born ...") [llm]
```

Because it lands as a fact, it is asked once, it shows up in `:facts` marked
`[llm]` rather than passing itself off as something you told us, and you can
contradict it.

Only questions go. A request is ours to carry out or to be taught, and an
assertion is yours to make, so neither is ever put outside. Nor is arithmetic
we can do ourselves: outside is the last thing tried, never the first.

### Looking it up, and keeping what comes back

A model asked for a fact will produce something fact-shaped. Wikidata asked
for a fact produces the fact, or nothing, which is a far more useful pair of
outcomes. So sources are tried in order of how much they deserve to be
believed, and a record beats a guess:

```
./chat --wikidata            # no model needed
./chat --llm --wikidata      # model reads the sentence, wikidata answers it
```

```
you  > when was albert einstein born
       Question(about=DateOfBirth(subject=AlbertEinstein()))
soup > 14 march 1879
```

The model, asked the same thing from memory, said April. That is the whole
argument for this being in front of it.

The lookup is also the one resolution strategy that **extends the vocabulary
as a side effect of being used**. Answering that question meant finding out
that `DateOfBirth` is Wikidata's P569 and that Einstein is Q937, and both are
worth writing down:

```
you  > :facts
       WikidataProperty(name=DateOfBirth(), value="P569")   [wikidata]
       WikidataId(name=AlbertEinstein(), value="Q937")      [wikidata]
       DateOfBirth(subject=AlbertEinstein(), value="14 march 1879") [wikidata]
```

The identifiers are ordinary facts, so the second question about Einstein is
cheaper than the first. And the concept itself gets *defined*, with
Wikidata's own description as its gloss, so a word that was a gap five
seconds ago is one soup can now tell you the meaning of. That is the sense in
which it teaches itself: not by being told, but by having gone and looked.

Search is a ranking rather than a lookup, so only an exact name counts, label
before alias. The top property match for "tower" is a Tower Records artist ID
and for "is" it is a library identifier; taking either would not be a near
miss, it would be a fabricated fact with a citation attached. A concept soup
can already act on is never looked up either, so this extends the vocabulary
without ever quietly reinterpreting it.

Statement rank and end dates are respected, because France has ten capitals
on record and nine of them stopped some time ago.

With no network the lookup marks itself unavailable and everything carries on
exactly as before.

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
