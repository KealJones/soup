# Getting started with soup

Soup thinks in concepts, not tokens. English comes in, becomes a concept
expression, gets realized against what Soup knows, and comes back out as
English. The model only does the first of those.

```
you     "is alice older than bob"
          |
        EARS          soup/ears.py      (a model in soup/seat.py)
          |
          v
        Question(about=OlderThan(left=Alice(), right=Bob()))
          |
        REALIZE       soup/realize.py
          |           natives in soup/builtins.py
          |           rules + facts in soup/knowledge.py
          v
        Answer(value=True, to=OlderThan(...))
          |
        MOUTH         soup/mouth.py
          |
you     "yeah, alice is older than bob"
```

If you already speak Soup's notation you can skip the model and type that
`Question(...)` yourself. `./chat --deaf` is useful for poking at the middle.

## Run it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./chat
```

Python 3.9+. On this Mac the ears load `qwen3.5:4b` in-process via MLX
(`soup/local.py`); define sits in `qwen3.5:9b`. First run downloads the
4-bit weights (about 2GB + 6GB).

```sh
./chat                          # talk
./chat --show                   # talk, and print the concept structures
./chat "what is 6 times 7"      # one shot
./chat --no-memory              # do not read or write data/memory.json
./chat --deaf                   # no model; only Soup's own notation works
./chat --hear                   # print what the ears heard; do not realize
./chat --no-wikidata            # do not look facts up (on by default)

python3 -m unittest discover -s tests
```

In the chat:

| command | what you get |
| --- | --- |
| `:ears` | the concept expression your last message became |
| `:answer` | the concept structure Soup resolved it to |
| `:why` | the realization trace |
| `:rules` | taught rewrites |
| `:facts` | remembered propositions |
| `:concepts [word]` | the vocabulary |
| `:wipe` | forget this session and reseed |

## The loop

`Session.respond` realizes `Turn(text)`. That is the whole host. Hear, the
Teacher detour, Speak, and Discourse live inside that concept.

`Session()` looks up missing facts in Wikidata by default. Pass `sources=[]`
to disable outside sources; the chat command's equivalent is `--no-wikidata`.

| concept | means |
| --- | --- |
| `Hear(text)` | ears. `as=Utterance()` stops at the speech act. `as=Program()` / `as=Thought()` understand and do not do |
| `Speak(x)` | mouth |
| `Teach(concept=..., meaning=...)` | file a rule. No body means ask the seat to define |
| `Turn(text)` | the public turn: Hear, realize, Speak. A gap becomes a lesson |
| `Think(about=x)` | realize `x` as thought. Computes. Does not Speak. Does not Write or Remember |
| `Consult(about=...)` | Hear through the seat, then Think |
| `Conversation(about=x, with=Self(), as=Thought())` | Think, named as a nested loop |
| `Subagent(about=..., with=Agent(name=...))` | nested Conversation: own Discourse, optional other seat |

Thinking is Conversation you are not addressed by. `Meaning` comes back.
Teach inside Think still files a rule (that is the good kind of side effect).
`Realizer.think_budget` is the depth cap (default 3).

## Layers

### 1. Concept expressions

The language the whole system shares. Looks like Python calls because that
buys binding, nesting, and named arguments for free. A call is meaning first,
and a function only if something realizes it.

| thing | type | example | file |
| --- | --- | --- | --- |
| expression | `Expr` | any of the rows below | `soup/expr.py` |
| literal | `Lit` | `6`, `"hi"`, `True` | |
| variable | `Var` | `x` in `Double(x)` | |
| argument | `Arg` | `left=Alice()` | |
| call | `Call` | `Multiply(3, 4)`, `Alice()` | |
| list | `Seq` | `[1, 2, 3]` | |

Syntax is parsed and printed in `soup/parse.py` / `render()`. Capitalized
names are concepts (`Alice()`), lowercase names are bindings (`x`).
`Head(x) := Body` is a definition.

The wrappers that say *what kind of move* an utterance is:

| speech act | means |
| --- | --- |
| `Question(about=...)` | they asked |
| `Request(action=...)` | they told you to do something |
| `Remember(proposition=...)` | they stated something |
| `Teach(concept=..., meaning=...)` | they defined a word |
| `Greeting()` `Laugh()` `Thanks()` | the obvious |

Ears pick one of those. Realization then works on the inside.

### 2. Ears

`soup/ears.py`. English in, one `Heard` (an `Expr` plus why) out.

If the text already looks like Soup notation, ears parse it directly. Otherwise
they ask the **seat** to `hear()`. The seat is the only place a model is
allowed to sit (`soup/seat.py`, weights in `soup/local.py`).

The seat has three methods, in order of how much they are worth:

| method | question | what comes back |
| --- | --- | --- |
| `hear()` | what did this sentence mean? | a shape, never an answer |
| `define()` | what does this concept mean in terms of ones we have? | a body, filed as a rule |
| `answer()` | what is this worth, if nothing else could tell us? | a fact, asked once |

`define` is the valuable one. Asked to quintuple ten, a 4B model guesses a
number. Asked what quintupling *is*, it says `Multiply(x, 5)`, and that is
kept.

Ears never decide how Soup accomplishes anything. `"delete the biggest file"`
is `Delete(target=Maximum(...))`, not a shell command. Unpacking that is
realization's job.

### 3. Knowledge

`soup/knowledge.py`. Plain serialisable data. Lives in `data/memory.json`.

| thing | what it is | example |
| --- | --- | --- |
| `ConceptDef` | a word Soup has heard, with a kind and a gloss | `Age` is an attribute |
| `Rule` | a rewrite: this expression becomes that one | `Double(x) := Multiply(x, 2)` |
| `Fact` | a proposition Soup believes, with truth and evidence | `Age(subject=Alice(), value=30)` |
| `Edge` | a relation between two concept names | `IsA` has property `Taxonomic` |
| `Evidence` | where a belief came from | `builtin`, `user`, `teacher`, `wikidata`, `llm` |

Concept kinds (seeded in `soup/builtins.py`): `entity`, `value`, `operation`,
`relation`, `property`, `attribute`, `quality`, `modality`, `speech`. Kind
matters. Entities stand for themselves (`Alice()`). Operations want to
compute. Attributes are facts you might be missing (`Age`, `Source`).

Three relation properties are wired into traversal, and that is all. There is
no enum of relation *kinds*. `IsA`, `PartOf`, `OppositeOf` are ordinary
concepts. Mark one `Symmetric` and it starts working both ways.

Natives (Python functions) are *not* in the JSON. They are registered at
import time with `@native` in `soup/builtins.py`. You cannot serialize a
closure.

### 4. Realization

`soup/realize.py`. Concepts in, concepts out. Execution is just the case
where a realization happens to touch the world.

**Realization** is the job. **Rule** is one way to do it (a rewrite, no
Python). The others:

| kind | where | example |
| --- | --- | --- |
| special form | `@native(..., special=True)` | `If`, `Question`, `Map` (do not evaluate args first) |
| native | `@native` in `builtins.py` | `Multiply`, `Write`, `Time`, `Fetch` |
| rule | `Knowledge.rules` | `Double(x) := Multiply(x, 2)` |
| fact | `Knowledge.facts` | `Age(subject=Alice)` -> `30` |
| inherited | walk `IsA` | if `Sprint` is a `Run`, try `Run`'s rule |
| learned | seat `define()`, then Teacher | `Quintuple(x) := Multiply(x, 5)` |
| outside | Wikidata, then seat `answer()` | `Capital(subject=France())` -> `"Paris"` |

Order is that list, top to bottom, in `_realize`. A name (`Greg()`, `lazy
dog`) is not a failure; it stands for itself. A known concept with a missing
particular becomes `Unknown(about=...)`. An unknown verb becomes a `Gap`.

A `Result` carries the value, the gaps, a trace (`:why`), effects (wrote a
file, read the clock), and any definitions it worked out this turn.

**Requests are not answered by the model.** Questions may go outside for a
missing fact. `"make me a sandwich"` has to be carried out or taught.

World natives (same family as the clock, not a coding agent):

| native | bottoms out in |
| --- | --- |
| `Time` `Date` `Day` | the clock |
| `Read` `Write` `Files` `Delete` | the filesystem |
| `Fetch` | HTTP |
| `WebSearch` | keyless DuckDuckGo search, returned as Markdown |
| `VisitWebpage` | fetch a page and return readable Markdown |
| `WikipediaSearch` | find an article and return its introduction and link |
| `TranscribeAudio` | local Whisper transcription of a file or audio URL |
| `Url` | structured URL, not string-gluing |
| `Json` `GetProperty` | parsing and picking fields |
| `Join` `Replace` `Concat` `PathOf` | paths and text |

`Create` and `Change` are *rules* over the filesystem natives:

```
Create(target=t, files=xs) := Map(collection=xs, transformation=Put(folder=PathOf(t)))
Change(target=t, from=a, to=b) := Write(path=PathOf(t), contents=Replace(Read(path=PathOf(t)), a, b))
```

Web search, page reading and Wikipedia lookup need no API key. `WebSearch`
uses the `ddgs` package (or its Python 3.9 predecessor). On Apple Silicon,
`TranscribeAudio` uses `mlx-whisper` and downloads its speech model on first
use.

`Get` already means "unwrap this" (`Find` / `Show`). JSON field access is
`GetProperty`. Do not merge them.

### 5. Not knowing

Three different holes, three different replies. `soup/teacher.py` handles
the language one.

| hole | example | what happens |
| --- | --- | --- |
| a name we have not heard | `Carl()` | it stands; names cannot fail |
| a fact we were never told | `Age(subject=Carl)` | `Unknown`, then maybe Wikidata or `answer()` |
| a word with no meaning | `Vibe(collection=[1,2])` | `Gap` -> Teacher: `teach me: Vibe(collection) := ...` |

A definition is rejected unless it is built from concepts Soup already has.
`Vibe(x) := Flumph(x)` teaches nothing if `Flumph` is also a mystery.

You can teach three equivalent ways:

```
Vibe(x) := Count(x)
vibe means the count of it
to quadruple something means to multiply it by 4
```

A realization is allowed to bottom out in code when nothing composable
will do: `HomeDir() := Python("os.path.expanduser('~')")`, `Shell("uname -s")`.
That is an ordinary rule, it persists, and `:rules` marks it `[python]` /
`[shell]` so you can see where a native probably belongs. Never for facts
about people.

### 5b. Thinking it through

A hole is a reason to think before it is a reason to ask. `Solve(about=x)`
realizes `x`, and when it will not resolve, puts the hole to itself as a
question, files whatever survives the Teacher, and realizes again. Capped by
`think_budget`.

| hole | the question it asks itself |
| --- | --- |
| a missing verb | `Teach(concept=Fifth(collection))` -> a rule |
| a name of unknown kind | `Kind(of=Chess())` -> `IsA` + property facts |

`Kind` is not `Teach`. A definition rewrites a concept away, and a name must
not be rewritten: chess is *a* game with two players, not interchangeable
with one. So it lands as `IsA(Chess, Game)` plus `Players(Chess)=2`, which
inheritance and `Query` already read.

A question can carry the situation it is about, and then it is solved rather
than looked up. The givens are believed for the length of the question and
dropped after, because a scene is not a belief:

```
Question(about=Players(subject=Chess()),
         given=[Play(subject=Kate(), object=Chess())])
-> asked myself Kind(of=Chess()) -> Taught(concept=Chess(), as=Game(players=2))
-> 2
```

`:why` shows the rounds. Nothing in here speaks, and nothing in here asks
you: a hole only you can fill comes back `Unknown` and the turn asks you
about it.

Asking you to define something is the last resort, not the first. Before
that, Soup tries to learn the word, tries to find out what the names in it
are, and asks the record. Getting "teach me Players" for something Wikidata
knows is a bug, not the design.

### 5c. Naming the property

Wikidata files how many people play chess under `minimum number of players`.
Nobody asks a question in those words, and `Players` matches no property
label at all, so the fact sits there unfound.

So the model gets asked what the property is *called*, and never what it is:

```
Players(inGame=Chess())
  -> "minimum number of players"     suggested
  -> P1872                           matched exactly, or nothing happens
  -> Q718 -> 2                       answered by the record
```

The suggestion still has to match a real property name exactly. A bad guess
finds nothing, which is the failure mode you want; it cannot turn into a
wrong fact with a citation stapled to it. The P-id is kept as a fact, so
`:facts` shows which property answered you and you can say it was the wrong
one.

### 6. Mouth

`soup/mouth.py`. Concept structure in, casual English out. This end is
supposed to be lossy. Soup's actual answer is the `Expr`; mouth is a renderer
with opinions.

### 7. Discourse

`soup/discourse.py`. What "it" and "that" point at, recent turns, a pending
teach. Pronoun resolution used to live more in ears; `Ref` currently realizes
to `Unknown` until discourse fills it.

### 8. Outside

Wikidata search and claim selection are rules over `Fetch`, `Json`,
`GetProperty`, `Filter`, and `First`. It is one fact source ahead of the model;
`--no-wikidata` skips it. Without a network connection, `Fetch` returns
`Unknown` and the rest of realization carries on. Search candidates and all
returned statements are kept as source-tagged concept facts in memory.

## File map

```
soup/
  __init__.py      public names
  __main__.py      python -m soup
  cli.py           REPL, flags, :commands
  session.py       the turn loop
  expr.py          Lit, Var, Arg, Call, Seq, match, substitute, render
  parse.py         the Python-ish syntax, including :=
  knowledge.py     concepts, rules, facts, edges, memory.json
  builtins.py      starting vocabulary + native Python realizations
  realize.py       the resolver, Gap, Result, @native
  ears.py          English -> Expr (or parse Soup notation)
  seat.py          hear / define / answer  (the model)
  local.py         in-process MLX / llama.cpp
  mouth.py         Expr -> English
  teacher.py       Gap -> Lesson -> Rule
  discourse.py     it / that / pending teach
  lookup.py        Wikidata fact-source glue over seeded concepts

chat               launcher
data/memory.json   what survived a restart
tests/test_soup.py the suite (scripted seat, no network)
scripts/sweep.py   throw a pile of utterances at it
tasks/later.md     unfinished: URL details, Run, workspace pin
GETTING_STARTED.md this file
README.md          the pitch
```

## A turn you can read

```
you  > what is 10 quintupled
ears > Question(about=Quintuple(10))          :ears
real > Quintuple is unknown
     > seat.define("Quintuple(x)")
     > Quintuple(x) := Multiply(x, 5)         kept in knowledge
     > Multiply(10, 5) -> 50                  native
mouth> 50 (worked out Quintuple(x) := Multiply(x, 5) for myself)

you  > what is the quintuple of 3
real > rule already there, 15                 no model this time
```

```
you  > create greet.py and main.py in /tmp/scratch
ears > Request(action=Create(target=Directory(path="/tmp/scratch"), files=[
         File(path="greet.py", contents="..."),
         File(path="main.py", contents="...")
       ]))
real > Create  (rule) -> Map of Put
     > Put     (rule) -> Write(Join(folder, PathOf(file)), GetProperty(file, "contents"))
     > Write   (native) touches disk
mouth> got it
```

## What to poke at

- Type Soup at it with `--deaf --show` and watch `:why`.
- Teach it a word, then `:rules`.
- Break a definition (`vibe means flumph`) and see the Teacher refuse it.
- `Read` / `Write` a file in `/tmp`, then `Change` a string in it.
- Do not add a new Python adapter because a website is shaped a certain way.
  If it is the world, it is a native (`Fetch`, `Write`). If it is a craft, it
  is a rule.
