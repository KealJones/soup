# Later

Ideas that showed up while giving soup a world, and that we did not do yet.

## URL remains a shitshow

`Url` is a native on purpose. Do not go back to concatenating query strings.
Still unfinished:

- fragments, ports, userinfo, `file:` and `data:`
- `Fetch` should take `Url(...)` without requiring it to have become a string
  first, and maybe a method (`GET` vs `POST`) and headers as an `Object`
- encoding of weird keys; repeated query keys

## Coding helper, next layer

Create and Change work when the ears (or a script) already put paths and
contents in the expression. Still missing:

- `Source(of=spec)` filled by the seat when a File has no contents, so a 4B
  model that says `Create(files=[Greet(function=Hello())])` still writes code
- `Run(program, args)` to execute what it just wrote, with a workspace root
- `Git` as concepts over `Run`, not a git library
- pinning `Write` / `Delete` to cwd (or a workspace) so it cannot spray `/`
- the 4B model mangling `/tmp/soup-spike/scratch` into `Tmp(SoupSpike())`
  and inventing `Tmp := PartOf`. Paths in the seat prompt have to stay
  string literals, and we should not learn a definition for a path segment.

## Reasoning past one hole at a time

`Solve` learns the verb and the kind, then realizes again. What it cannot do
yet is the step the sisters riddle actually turns on:

```
5 sisters. Ann reads, Margaret cooks, Kate plays chess, Marie does laundry.
what is the fifth doing?
```

Soup now hears the whole scene, works out `Fifth(of) := Nth(collection=of,
index=5)` for itself, and says plainly that it does not know the rest. What
stops it is that `Sisters(count=5)` is a number, not five sisters: there is
nothing for `Nth` to index into. The givens name four of them and the fifth
is never named at all.

The rest needs two things it does not have:

- **counting a scene.** Five sisters, four named, so there is exactly one
  unnamed. That is `Count` over the givens against `Sisters(count=5)`.
- **uniqueness.** A two-player game with one named player has one open seat,
  and if exactly one person is unaccounted for, she is in it.

Both are rules over `Query` / `Count` / `Not`, not new Python. Do not write a
`Riddle` native, and do not let `answer()` have it: the model knows this one
by heart and quoting the punchline is not solving it.

## Small things

- `Get` already means "unwrap", so json field access is `GetProperty` / `At`.
  Do not alias them.
- `Make` is "make me a sandwich", not `Create`. Do not merge them.
- Mouth used to describe `Acknowledged` as a missing fact, because Write
  returned it and realization tried to compute it. It is a finished speech
  act, same as `Taught`. Keep it in the self-describing natives.

## The loop as concepts

Done. `Session.respond` realizes `Turn(text)`. Subagent is a nested
Conversation with its own Discourse and an optional named seat
(`Realizer.agents`). Think quotes Write/Delete/Fetch/Remember. Teach inside
Think still files. `Realizer.think_budget` is public (default 3).

Do not wrap `session.respond` in a native called `Chat`. That is the host
calling itself.

