# Later

Ideas that showed up while giving soup a world, and that we did not do yet.

## Recast Wikidata as concepts

`lookup.py` is still a special adapter. It should be rules on top of
`Fetch`, `Json`, `GetProperty`, and `Url`.

```
WikidataSearch(text, kind) := Json(Fetch(Url(
  scheme="https",
  host="www.wikidata.org",
  path="/w/api.php",
  query=Object(action="wbsearchentities", search=text, type=kind, language="en")
)))
```

Exact label-before-alias, preferred rank, and skipping `P582` end dates are
`Filter` / `First`, not Python. Then `self.lookup` can leave the realizer.

## URL remains a shitshow

`Url` is a native on purpose. Do not go back to concatenating query strings.
Still unfinished:

- fragments, ports, userinfo, `file:` and `data:`
- `Fetch` should take `Url(...)` without requiring it to have become a string
  first, and maybe a method (`GET` vs `POST`) and headers as an `Object`
- encoding of weird keys; repeated query keys
- Wikidata's `|` in `ids=` once that recast happens

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

## Small things

- `Get` already means "unwrap", so json field access is `GetProperty` / `At`.
  Do not alias them.
- `Make` is "make me a sandwich", not `Create`. Do not merge them.
- Mouth used to describe `Acknowledged` as a missing fact, because Write
  returned it and realization tried to compute it. It is a finished speech
  act, same as `Taught`. Keep it in the self-describing natives.

