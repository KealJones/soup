"""ast vs 4B into Soup IR. Throwaway spike, not a product.

    python3.12 scripts/code_ir_spike.py
"""

from __future__ import annotations

import ast
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import fresh_knowledge
from soup.expr import Arg, Call, Expr, Lit, Seq, Var, concept_names, render
from soup.parse import ParseError, parse
from soup.seat import Seat, _clean

SAMPLES = [
    (
        "add",
        "python",
        "def add(a, b):\n    return a + b\n",
        {"name": "add", "params": {"a", "b"}, "ops": {"Add", "Return"}, "flow": set()},
    ),
    (
        "clamp",
        "python",
        "def clamp(x, lo, hi):\n"
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n",
        {"name": "clamp", "params": {"x", "lo", "hi"}, "ops": {"Return", "If"}, "flow": {"If"}},
    ),
    (
        "unique",
        "python",
        "def unique(items):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            out.append(x)\n"
        "    return out\n",
        {"name": "unique", "params": {"items"}, "ops": {"For", "If", "Return"}, "flow": {"For", "If"}},
    ),
    (
        "avg_positive",
        "python",
        "def avg_positive(xs):\n"
        "    ys = [x for x in xs if x > 0]\n"
        "    return sum(ys) / len(ys) if ys else 0\n",
        {"name": "avg_positive", "params": {"xs"}, "ops": {"Divide", "If"}, "flow": {"If"}},
    ),
    (
        "ts_add",
        "typescript",
        "export function add(a: number, b: number): number {\n"
        "  return a + b;\n"
        "}\n",
        {"name": "add", "params": {"a", "b"}, "ops": {"Add", "Return"}, "flow": set()},
    ),
    (
        "ts_unique",
        "typescript",
        "export function unique<T>(items: T[]): T[] {\n"
        "  const seen = new Set<T>();\n"
        "  const out: T[] = [];\n"
        "  for (const x of items) {\n"
        "    if (!seen.has(x)) {\n"
        "      seen.add(x);\n"
        "      out.push(x);\n"
        "    }\n"
        "  }\n"
        "  return out;\n"
        "}\n",
        {"name": "unique", "params": {"items"}, "ops": {"For", "If", "Return"}, "flow": {"For", "If"}},
    ),
]

_IR = '''You translate source code into Soup concept expressions. You are not
writing code. Reply with exactly one expression and nothing else: no prose,
no fences, no explanation.

SYNTAX
  Concept(name=Value, name=Value)
  Concept()
  "text"  42  [1, 2, 3]

Concept names are CapitalizedCamelCase. Argument names are lowercase.
A function is:
  Function(name="add", language=Python(), params=[a, b], body=Return(Add(a, b)))

Use these when they fit: Function, Return, Assign, If, For, While, Seq,
Add, Subtract, Multiply, Divide, Not, And, Or, GreaterThan, LessThan,
EqualTo, Contains, Filter, Map, Count, First, Last, Append, Call, Var.
Parameter names are bare: a, not Var("a") and not "a" unless it is text.
Do not invent a wrapper like Code() or Program() around the Function.
Do not drop a branch. Do not rewrite a loop into a word you cannot back.
'''

_BIN = {
    ast.Add: "Add",
    ast.Sub: "Subtract",
    ast.Mult: "Multiply",
    ast.Div: "Divide",
    ast.Mod: "Modulo",
    ast.Pow: "Power",
}
_CMP = {
    ast.Lt: "LessThan",
    ast.LtE: "AtMost",
    ast.Gt: "GreaterThan",
    ast.GtE: "AtLeast",
    ast.Eq: "EqualTo",
    ast.NotEq: "NotEqual",
    ast.In: "Contains",
    ast.NotIn: "Contains",
}


def _call(concept: str, **kwargs) -> Call:
    return Call(concept, tuple(Arg(k, v) for k, v in kwargs.items() if v is not None))


def _from_ast(node) -> Expr:
    if node is None:
        return Lit(None)
    if isinstance(node, ast.Module):
        items = [_from_ast(s) for s in node.body]
        return items[0] if len(items) == 1 else Seq(tuple(items))
    if isinstance(node, ast.FunctionDef):
        params = Seq(tuple(Var(a.arg) for a in node.args.args))
        body = [_from_ast(s) for s in node.body]
        return _call(
            "Function",
            name=Lit(node.name),
            language=_call("Python"),
            params=params,
            body=body[0] if len(body) == 1 else Seq(tuple(body)),
        )
    if isinstance(node, ast.Return):
        return _call("Return", value=_from_ast(node.value) if node.value else Lit(None))
    if isinstance(node, ast.Assign):
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else _from_ast(target)
        return _call("Assign", name=Var(name) if isinstance(name, str) else name, value=_from_ast(node.value))
    if isinstance(node, ast.If):
        return _call(
            "If",
            condition=_from_ast(node.test),
            then=_block(node.body),
            **({"otherwise": _block(node.orelse)} if node.orelse else {}),
        )
    if isinstance(node, ast.For):
        return _call(
            "For",
            item=_from_ast(node.target),
            collection=_from_ast(node.iter),
            body=_block(node.body),
        )
    if isinstance(node, ast.Expr):
        return _from_ast(node.value)
    if isinstance(node, ast.BoolOp):
        op = "And" if isinstance(node.op, ast.And) else "Or"
        return Call(op, tuple(Arg(None, _from_ast(v)) for v in node.values))
    if isinstance(node, ast.BinOp):
        return _call(_BIN.get(type(node.op), type(node.op).__name__), left=_from_ast(node.left), right=_from_ast(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _call("Not", value=_from_ast(node.operand))
    if isinstance(node, ast.Compare):
        right = _from_ast(node.comparators[0])
        op = node.ops[0]
        name = _CMP.get(type(op), type(op).__name__)
        expr = _call(name, left=_from_ast(node.left), right=right)
        if isinstance(op, ast.NotIn):
            return _call("Not", value=expr)
        if isinstance(op, ast.In):
            return _call("Contains", collection=right, item=_from_ast(node.left))
        return expr
    if isinstance(node, ast.Call):
        func = _from_ast(node.func)
        args = tuple(Arg(None, _from_ast(a)) for a in node.args)
        if isinstance(func, Var):
            return Call(func.name[:1].upper() + func.name[1:] if func.name else "Call", args)
        if isinstance(func, Call) and func.concept == "GetProperty":
            return Call("Call", (Arg("on", func),) + args)
        return Call("Call", (Arg("of", func),) + args)
    if isinstance(node, ast.Attribute):
        return _call("GetProperty", of=_from_ast(node.value), name=Lit(node.attr))
    if isinstance(node, ast.Name):
        return Var(node.id)
    if isinstance(node, ast.Constant):
        return Lit(node.value)
    if isinstance(node, ast.List):
        return Seq(tuple(_from_ast(x) for x in node.elts))
    if isinstance(node, ast.ListComp):
        return _call(
            "Filter" if node.generators[0].ifs else "Map",
            collection=_from_ast(node.generators[0].iter),
            item=_from_ast(node.generators[0].target),
            **(
                {"test": _from_ast(node.generators[0].ifs[0])}
                if node.generators[0].ifs
                else {"body": _from_ast(node.elt)}
            ),
        )
    if isinstance(node, ast.IfExp):
        return _call(
            "If",
            condition=_from_ast(node.test),
            then=_from_ast(node.body),
            otherwise=_from_ast(node.orelse),
        )
    return _call("Unknown", about=Lit(type(node).__name__))


def _block(stmts) -> Expr:
    items = [_from_ast(s) for s in stmts]
    return items[0] if len(items) == 1 else Seq(tuple(items))


def ast_ir(source: str) -> Expr:
    return _from_ast(ast.parse(source))


def score(tree: Expr, expect: dict, known: set) -> dict:
    names = set(concept_names(tree))
    text = render(tree, False)
    params_hit = sum(1 for p in expect["params"] if ("(%s)" % p) in text or p in text)
    flow_hit = expect["flow"] <= names or (
        "For" in expect["flow"] and "Filter" in names
    )
    return {
        "parses": True,
        "name": expect["name"] in text,
        "params": params_hit == len(expect["params"]),
        "flow": flow_hit if expect["flow"] else True,
        "unknowns": "Unknown(" in text,
        "invented": sorted(n for n in names if n not in known and n[0:1].isupper()),
        "concepts": len(names),
        "chars": len(text),
    }


def llm_ir(seat: Seat, source: str, language: str) -> tuple:
    raw = seat._ask(_IR, "%s\n\n%s" % (language, source))
    if not raw:
        return None, seat.last_error or "empty"
    try:
        return parse(_clean(raw)), raw
    except (ParseError, ValueError, IndexError) as exc:
        return None, "unparseable: %s | %s" % (exc, (raw or "")[:220])


def main() -> int:
    known = set(fresh_knowledge().concepts)
    known.update(
        {
            "Function",
            "Return",
            "Assign",
            "For",
            "While",
            "Body",
            "Contains",
            "Call",
            "GetProperty",
            "NotEqual",
            "Python",
            "TypeScript",
            "Filter",
            "Map",
            "Append",
        }
    )
    seat = Seat(embed=True)
    orig = seat._payload

    def _payload(system, utterance):
        body = orig(system, utterance)
        if "options" in body:
            body["options"]["num_predict"] = 700
        else:
            body["max_tokens"] = 700
        return body

    seat._payload = _payload
    if not seat.available:
        print("no model:", seat.last_error)
        return 1
    # HTTP fallback is fine; this is a read of the 4B, not a loader test.
    probe = seat._ask("reply with the word ok", "ok")
    print("model:", seat.model, "probe:", (probe or seat.last_error or "")[:60])
    if not probe:
        return 1

    rows = []
    for name, language, source, expect in SAMPLES:
        print("==", name, language)
        ast_tree = None
        ast_err = ""
        t0 = time.time()
        if language == "python":
            try:
                ast_tree = ast_ir(source)
            except SyntaxError as exc:
                ast_err = str(exc)
        ast_ms = int((time.time() - t0) * 1000)

        t0 = time.time()
        llm_tree, llm_note = llm_ir(seat, source, language)
        llm_ms = int((time.time() - t0) * 1000)

        if ast_tree is not None:
            ast_s = score(ast_tree, expect, known)
            print("  ast  %s" % render(ast_tree, False)[:220])
            print("       parse name params flow unk invented=%s  %dms" % (
                ast_s["invented"], ast_ms))
        else:
            ast_s = None
            print("  ast  SKIP", ast_err or "not python")

        if llm_tree is not None:
            llm_s = score(llm_tree, expect, known)
            print("  llm  %s" % render(llm_tree, False)[:220])
            print("       parse name params flow unk invented=%s  %dms" % (
                llm_s["invented"], llm_ms))
        else:
            llm_s = None
            print("  llm  FAIL %s  %dms" % (llm_note, llm_ms))
        print()
        rows.append((name, language, ast_s, llm_s, ast_ms, llm_ms))

    print("score  (name / params / flow must be true)")
    print("%-14s %-8s %-20s %-20s" % ("sample", "lang", "ast", "4b"))
    for name, language, ast_s, llm_s, ast_ms, llm_ms in rows:
        def fmt(s, ms):
            if s is None:
                return "n/a"
            flags = "".join(
                "Y" if s[k] else "n"
                for k in ("parses", "name", "params", "flow")
            )
            extra = " unk" if s["unknowns"] else ""
            extra += (" +" + ",".join(s["invented"][:3])) if s["invented"] else ""
            return "%s%s %dms" % (flags, extra, ms)

        print("%-14s %-8s %-20s %-20s" % (name, language, fmt(ast_s, ast_ms), fmt(llm_s, llm_ms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
