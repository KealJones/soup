# soup

A small conversational system that thinks in concepts instead of tokens. No
API key. A local model sits in-process and does the listening.

A model reads the sentence, and that is all it does. Everything after that is
Soup's: what the concepts mean, how they resolve, what gets learned when one
of them is missing, and what gets kept. When Soup meets an idea it does not
have, it works out a definition, files it as an ordinary rule you can read and
contradict, and answers the question it was actually asked. The next hundred
sentences built on that idea then need nobody's help at all.

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

Needs Python 3.9 or newer and a model. On this Mac that means MLX: soup
loads `qwen3.5:4b` for hearing and `qwen3.5:9b` for define, in-process, no
server. First run downloads the 4-bit weights into the Hugging Face cache.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./chat
```

Ollama is optional. If you already `ollama pull`'d something, `--llm` still
accepts those names; they map onto the matching MLX repo. A path to a `.gguf`
or `$SOUP_LLM_URL` still work if you would rather. `$SOUP_TEACHER_MODEL`
picks the define chair (default `qwen3.5:9b`).

```sh
./chat                          # talk to it
./chat --show                   # talk to it, printing the concept structures
./chat "what is 6 times 7"      # say one thing and quit
./chat --no-memory              # do not read or write data/memory.json
./chat --hear                   # print what the ears heard; do not realize

python3 -m unittest discover -s tests
python3 scripts/sweep.py        # talk at it with 50-odd utterances at once
```

Layers, terms, and which file is which: [GETTING_STARTED.md](GETTING_STARTED.md).

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
| `ears.py` | English to concepts. a hundred lines, because a model does the reading |
| `seat.py` | the model's seat, and the only three things it may be asked |
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

### The seat

`ears.py` used to be two thousand lines of construction grammar, and deleting
it was the single best change in this repo. What it really did was recognise
the phrasings somebody had thought of in advance. `(hi|hello|hey|yo|sup)` is
not understanding that a greeting has occurred; it is a list, and the list
does not contain `heya`. Every sentence it got wrong was fixed by extending a
list, which is not learning either, so it looked clever on its own examples
and fell over on the next thing anybody said.

Reading a sentence is the one part of this design that is solved better
elsewhere. So a model does it, from a seat with exactly three methods on it:

| | |
|---|---|
| `hear()` | what did this sentence mean? shape only, never an answer |
| `define()` | what does this concept mean, in terms of ones we have? |
| `answer()` | what is this worth, if nothing else could tell us? |

```sh
./chat                              # 4b hears, 9b defines, in-process
./chat --llm qwen3.5:4b
./chat --teacher qwen3.5:9b
./chat --llm /path/to/model.gguf
SOUP_LLM_URL=http://localhost:1234/v1/chat/completions ./chat   # still works
./chat --deaf                       # no model. soup will not understand a word
```

Nothing else in Soup talks to a model, and nothing that comes back is trusted
for having come from one. Concepts invented out of thin air are rejected,
restatements of the question are rejected, and a definition built from things
Soup does not have is thrown out by the Teacher. The prompt carries the real
concept vocabulary grouped by kind, which is what stops a model inventing
`Times` and `HowOld` alongside the `Multiply` and `Age` already present.

### Working it out instead of asking

The three seat methods are listed in order of what they are worth, and the
middle one is the interesting one. An answer helps once. A definition is kept.

So when a concept will not resolve, Soup asks what it *means* before it asks
what it is worth:

```
you  > what is 10 quintupled
soup > 50 (worked out Quintuple(x) := Multiply(x, 5) for myself)
you  > what is the quintuple of 3
soup > it's 15
```

The second answer cost nothing. `Quintuple` is a rule now, sitting in memory
next to `Double` and `Half`, and it survives restarts. Ask a 4B model to
quintuple ten and it guesses a number, often the wrong one. Ask it what
quintupling *is* and it is right, permanently.

Same story for vocabulary. "military time" is not a word list in `builtins.py`,
because the fifth spelling of it would break that. There is one concept,
`TwentyFourHourTime`, and mapping any phrase onto it is the ears' job:

```
you  > what time is it in military time
soup > 07:20
```

A definition Soup works out for itself goes through the same Teacher a human
answer would: rejected unless grounded in concepts already present,
parameterised the same way, filed as the same kind of ordinary rule. Only the
source differs. Asking you is still there, as the last resort it should always
have been.

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
./chat                       # wikidata is on
./chat --no-wikidata         # guess, or ask, instead of looking it up
```

`Session()` enables the same source by default for Python callers. Pass
`sources=[]` to keep a session offline or to provide only your own sources.

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
before alias. Those choices are `Filter` and `First` rules over the structured
JSON returned by `WikidataSearch`; statement rank and end dates use the same
concept operations. A concept soup can already act on is never looked up, so
this extends the vocabulary without quietly reinterpreting it.

With no network the lookup marks itself unavailable and everything carries on
exactly as before.

Soup keeps the complete search candidates and all statements returned for
the requested property as `WikidataSearchResult` and `WikidataStatement`
facts. The chosen answer is still filtered by exact label, rank, and end date;
the other returned records remain available in `:facts` and saved memory.

### Search and read the web

Soup also has explicit, keyless web actions. They return Markdown so the
model or a person can read the results:

```
search the web for concept-based AI assistants
Request(action=WebSearch(query="concept-based AI assistants"))

read https://example.org/guide
Request(action=VisitWebpage(url="https://example.org/guide"))

look up Ada Lovelace on Wikipedia
Request(action=WikipediaSearch(query="Ada Lovelace"))
```

`TranscribeAudio(path="/path/to/recording.wav")` runs Whisper locally. On
Apple Silicon, `mlx-whisper` is installed with the project requirements and
downloads its model on first use. No search, page-reading, or transcription
API key is needed. These are actions Soup takes when asked; they are separate
from the automatic Wikidata fact source.

The model is in-process. On Apple Silicon that is MLX running a 4-bit
Qwen3.5-4B; a Hugging Face id or a `.gguf` path also work. A server is still
accepted if `$SOUP_LLM_URL` is set.

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

Understanding is only as good as the model in the seat, and a 4B model is a
4B model. It is reliable at shape and much shakier at argument naming, so it
will hand back `BiggerThan(subject=, object=)` where the rule it needs to
match wants `left=` and `right=`, and the comparison then quietly fails to
fire. Bigger models do this less. This is a real cost of the design and not
one to wave away: the grammar it replaced was worse at almost everything, but
it was worse in ways that never changed between runs.

No model at all means no ears. Soup says "i have no model to hear you with"
rather than guessing, which is the honest answer but still a hard dependency
where there used to be none.

Self-teaching is bounded to two definitions per utterance and only fires on
concepts Soup does not already have, so it will not rescue a sentence that
was misread rather than unknown. A definition that turns out wrong is a rule
like any other: visible in `:rules`, and you can overwrite it by saying so.

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
