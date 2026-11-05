# soup

A small conversational system that thinks in concepts instead of tokens. No
API key. A local model sits in-process and does the listening.

A model reads the sentence, and that is its main job. Everything after that is
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

Status: an experiment, no longer under active development (last commit
2026-09-15). See [History](#history) for where it sits in the line of projects.

## Running it

Needs Python 3.9 or newer and a model. On Apple Silicon that means MLX: soup
loads `qwen3.5:4b` (`mlx-community/Qwen3.5-4B-MLX-4bit`) in-process, no
server, and the first run downloads the 4-bit weights into the Hugging Face
cache. That model hears and speaks. Defining concepts and answering facts go
to a larger teacher model, `qwen3.8:27b` by default; if it cannot be loaded,
the ears model fills in rather than Soup going deaf.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./chat
```

`requirements.txt` pulls in `mlx` and `mlx-lm`, `llama-cpp-python` (for
`.gguf` files), `ddgs` for keyless web search, and `mlx-whisper` on Apple
Silicon. `./chat` is a launcher that finds a new enough Python (or uses
`.venv`) and runs `python -m soup`.

Ollama is optional. Ollama-style names still work with `--llm` and
`--teacher`; the `qwen3.5` sizes 0.8b, 2b, 4b and 9b map onto their MLX
repos, and other names are looked for among Ollama's downloaded weights. A
path to a `.gguf` loads through llama.cpp. `$SOUP_LLM_URL` points the seat at
a server instead, and `$SOUP_LLM_MODEL`, `$SOUP_TEACHER_MODEL`,
`$SOUP_LLM_WEIGHTS`, `$SOUP_LLM_KEY` and `$SOUP_LLM_TIMEOUT` tune it.

```sh
./chat                          # talk to it
./chat --show                   # talk to it, printing the concept structures
./chat "what is 6 times 7"      # say one thing and quit
./chat --no-memory              # do not read or write data/memory.json
./chat --memory path.json       # keep memory somewhere else
./chat --hear                   # print what the ears heard; do not realize
./chat --llm qwen3.5:9b         # a different listening model
./chat --teacher qwen3.5:9b     # a different define/answer model
./chat --no-wikidata            # guess, or ask, instead of looking facts up
./chat --deaf                   # no model. soup will not understand a word

python3 -m unittest discover -s tests   # 236 tests, no model needed
python3 scripts/sweep.py                # talk at it with 50-odd utterances at once
```

Layers, terms, and which file is which, at length: [GETTING_STARTED.md](GETTING_STARTED.md).

In the chat, `:help` lists the commands:

| command | what it shows |
| --- | --- |
| `:ears` | the concept expression your last message became |
| `:answer` | the concept structure soup answered with |
| `:why` | how it got there |
| `:know` | what soup knows, in numbers |
| `:concepts [word]` | concepts, with glosses when filtered |
| `:facts [word]` | remembered facts |
| `:rules` | realizations it has been taught |
| `:show` / `:quiet` | print ears and answer every turn, or stop |
| `:save` | write memory to disk |
| `:wipe` | forget everything learned this session and reseed |
| `:q` | leave |

## The idea

Language is one representation of meaning, not the medium of thought. So the
English goes away as early as possible and comes back as late as possible.

```
                 "if greg has the car he probably left"
                                  |
                                HEAR
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
                  SPEAK                        |
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

Hearing, talking and learning are concepts too. A turn is realizing
`Turn(text)`, and `Hear`, `Speak` and `Teach` are ordinary entries in the
vocabulary, realized like any other.

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
| `builtins.py` | the starting vocabulary and the realizations that bottom out in code, including `Turn`, `Hear` and `Speak` |
| `ears.py` | English to concepts, and the checks on what the model hands back |
| `seat.py` | the models' seat, and the only things they may be asked |
| `local.py` | loads weights in-process (MLX, or llama.cpp for `.gguf`) |
| `teacher.py` | turns a gap into a question, and an answer into a realization |
| `lookup.py` | Wikidata: search, claims, property names, provenance |
| `web.py` | keyless web search, page reading, Wikipedia |
| `discourse.py` | what "it" and "that" currently point at |
| `session.py` | holds knowledge, discourse and the seat; one turn realizes `Turn(text)` |
| `cli.py` | the chat loop and its commands |
| `bench.py` | exam conditions for multiple-choice benches (MMLU-Pro) |

`scripts/sweep.py` is the "does it feel like talking to something" check,
`scripts/mmlu_pro.py` runs MMLU-Pro against Soup or a bare Qwen for
comparison, and `scripts/code_ir_spike.py` is a throwaway spike comparing
Python's `ast` with a 4B model at turning code into Soup expressions.
`tasks/later.md` holds ideas not done yet.

### Not knowing, in three flavours

Soup distinguishes them, because they call for different replies.

- **A name it has not heard.** `Carl()` resolves to itself. Names cannot fail
  to evaluate.
- **A thing it was never told.** `Age(subject=Carl)` becomes
  `Unknown(about=...)`, which propagates up so the answer names the thing that
  was actually missing rather than the outermost operation. The reply is "i
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
elsewhere. So a model does it, from a seat with a short, fixed list of
questions it may be asked. Each is narrow, and they are tried in order of
what the result is worth:

| | model | |
|---|---|---|
| `hear()` | ears | what did this sentence mean? shape only, never an answer |
| `define()` | teacher | what does this concept mean, in terms of ones we have? |
| `answer()` | teacher | what is this worth, if nothing else could tell us? |
| `property_names()` | teacher | what would a record keeper call this property? |
| `settle()` | teacher | which of these options, when everything else came up empty? |
| `speak()` | ears | say this finished answer in English, adding nothing |

Nothing outside the seat talks to a model, and nothing that comes back is
trusted for having come from one. Concepts invented out of thin air are
rejected, restatements of the question are rejected, and a definition built
from things Soup does not have is thrown out by the Teacher. The prompt
carries the real concept vocabulary grouped by kind, which is what stops a
model inventing `Times` and `HowOld` alongside the `Multiply` and `Age`
already present.

The ears are allowed to be baffled. They are not allowed to make things up,
and a small model asked "who is albert einstein" will genuinely hand back a
concept named `Alice`. The line drawn is between naming and inventing: a
model may name a relation you never said, because reading "turn the volume
down" as `Decrease(target=Volume())` is the paraphrase we want from it, but
it may not introduce a *thing* that was never mentioned. Heads are free;
leaves are checked against what was actually spoken.

```
you  > make this picture look spooky but still cute
       Request(action=Make(target=Ref("this"), look=AllOf(Spooky(), Cute())))
soup > i don't know Make. teach me: Make(target=target, look=look) := ...
```

### Speaking

Soup's answer is the concept expression. `Speak` only chooses words for it:
the seat is handed the finished expression and told to add nothing, because a
mouth that decides facts is a second brain disagreeing with the first. Text
that came back from a lookup is said as the source wrote it rather than
paraphrased. A style is a concept too (`Speak(of=X, style=Pirate())`). With no
model in the chair, the expression goes out as it is.

### Working it out instead of asking

`define()` is the interesting one. An answer helps once. A definition is kept.

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

Holes get thought about before they get asked about. `Solve` puts a hole to
itself as a question (`Kind(of=Chess())` lands as `IsA(Chess, Game)` plus
`Players(Chess)=2`), files what survives the Teacher, and realizes again,
within a budget. GETTING_STARTED.md walks through it.

### Asking, as opposed to guessing

There are two ways to not know something, and only one of them is a language
problem:

- *I don't know that word.* A hole in the vocabulary. Only you can fill it,
  so the Teacher asks you.
- *I don't know that fact.* A hole in the world. Anyone could fill it, and a
  record or a model is right there.

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
believed, and a record beats a guess. Wikidata is on by default, in `./chat`
and for Python callers of `Session()`; pass `sources=[]` to keep a session
offline or to provide only your own sources.

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

When nobody asks in Wikidata's words (`Players` matches no property label),
the teacher is asked what the property is *called*, never what the value is.
The suggestion has to match a real property name exactly, so a bad guess
finds nothing rather than a wrong fact with a citation stapled to it.

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

### A world to act in

Soup also has explicit, keyless actions. Their results stay in Soup's concept
language: `SearchResults` contains `SearchResult` values, and `WebPage`
contains sections and links. When answering, the source text is spoken, while
the full structured result remains available to Soup:

```
search the web for concept-based AI assistants
Request(action=WebSearch(query="concept-based AI assistants"))

read https://example.org/guide
Request(action=VisitWebpage(url="https://example.org/guide"))

look up Ada Lovelace on Wikipedia
Request(action=WikipediaSearch(query="Ada Lovelace"))
```

`TranscribeAudio(path="/path/to/recording.wav")` runs Whisper locally through
`mlx-whisper` on Apple Silicon, which downloads its model on first use. Files
are concepts as well (`File`, `Read`, `Write`, `Files`, `Delete`, `PathOf`,
`Join`), as are `Url`, `Fetch`, `Json` and `GetProperty`, so Wikidata itself
is reached through ordinary rules over them. `Python` runs a taught snippet
with a restricted set of builtins, and `Shell` runs a command. No search,
page-reading, or transcription API key is needed. These are actions Soup takes
when asked; they are separate from the automatic Wikidata fact source.

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

`Shell` runs whatever command reaches it, with an eight second timeout and no
confirmation step. Fine on your own machine, while you are watching; not
something to point at anything you care about.

The clock is the one thing Soup reads from outside its own memory unasked,
because "i don't know what time it is" is a silly answer.

Memory is closed-world and flat. "does Bob like Alice" answers "i don't know"
rather than "no", which is correct but chattier than people expect.

## History

Soup is the third of five related experiments in building an assistant whose
knowledge lives in a graph it can read and grow, rather than in a model's
weights:

1. [EKG AI](https://github.com/KealJones/ekg-ai) (Aug 15 to 20, 2026):
   typed executable programs, taught by an LLM Teacher.
2. [Spoon](https://github.com/KealJones/spoon) (Aug 22 to Sep 9): EKG
   renamed and rebuilt in Rust, ending as "everything is a Concept" with LLM
   ears, mouth and Teacher.
3. **Soup** (Sep 10 to 15): this repository.
4. [Napkin](https://github.com/KealJones/napkin) (Sep 16 to Oct 17).
5. [Noodle](https://github.com/KealJones/noodle) (Sep 29 to Oct 4).

Soup started the day after Spoon's last commit, carrying Spoon's concept
model into a much smaller Python codebase. Spoon's final experiment (Sep 9)
had found that a 4B model reading sentences into concepts did far better
writing Python-style calls, `add(2, 3)`, than angle brackets, `add<2, 3>`
(46% against 18% on the same cases). Soup's concept expressions are Python
calls from the first commit.

- **Sep 10.** First commit: ears, realization, mouth and Teacher, with a
  hand-written construction grammar doing the hearing and an optional local
  model tried only for what the constructions missed. Relations became
  concepts with properties instead of a closed enum. Fact questions went to
  the model instead of to you, then to Wikidata first.
- **Sep 11.** The construction grammar was deleted: `ears.py` went from 1,962
  lines to 107, and a model became the only reader, from a seat with
  `hear()`, `define()` and `answer()`. The model moved in-process so no server
  was needed. Soup got a world: `Read`, `Write`, `Fetch`, `Url`, `Json`,
  `GetProperty`.
- **Sep 15.** The concept model grew: `Turn`, `Hear` and `Speak` became
  concepts, `mouth.py` was deleted in favour of the seat choosing the words,
  defining and answering got their own larger teacher model, and `Solve` started asking itself
  before asking you. Web search, page reading, Wikipedia and transcription
  arrived, Wikidata answers got labelled provenance, and the MMLU-Pro bench
  harness was added. This was the last commit.

A separate TypeScript version of Soup also existed outside this repository.
That version is the one Napkin's architecture audit read on Sep 16; its
verdict was that concepts owned their rewrite rules well, but native and
effect code hid in closures that could not be inspected or saved, and the
session ran a second execution path beside the evaluator. This repository is
the Python one.

Napkin, which followed, kept the idea that everything is a concept and pushed
further on nothing being privileged, and it went the other way on the model:
within nine days no model heard or spoke at all.
