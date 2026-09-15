"""Ask Soup the MMLU-Pro test set and write a scoreboard as it goes.

    python3.12 -u scripts/mmlu_pro.py --system qwen4b --workers 4
    python3.12 -u scripts/mmlu_pro.py --system qwen27b --workers 2
    python3.12 -u scripts/mmlu_pro.py --system soup

Writes --results after every item so a kill is just a resume.
Does not touch data/memory.json.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup.bench import RecordingSeat, extract_choice, fetch_split, format_item, grade
from soup.seat import DEFAULT_MODEL, DEFAULT_TEACHER, Seat
from soup.session import Session

SPLITS = {
    "test": (os.path.join("data", "mmlu_pro", "test.json"), 12032),
    "validation": (os.path.join("data", "mmlu_pro", "validation.json"), 70),
}
RESULTS = os.path.join("data", "bench", "mmlu_pro_test.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run soup on MMLU-Pro")
    parser.add_argument("--system", choices=("soup", "qwen4b", "qwen9b", "qwen27b"), default="soup")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--limit", type=int, default=0, help="stop after N items (0 = whole split)")
    parser.add_argument("--workers", type=int, default=0, help="parallel items (0 = 1 for soup, 4 for raw)")
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument(
        "--record",
        default="",
        help="write what the models actually said to this file, for replay in tests",
    )
    parser.add_argument("--memory", default=os.path.join("data", "bench", "mmlu_test_memory.json"))
    args = parser.parse_args(argv)
    workers = args.workers or (1 if args.system == "soup" else 4)

    cache, expected = SPLITS[args.split]
    print("fetching TIGER-Lab/MMLU-Pro %s (%d)..." % (args.split, expected), flush=True)
    items = fetch_split(cache, args.split, expected=expected)
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    done = _load_done(args.results, args.system)
    remaining = [item for item in items if item["question_id"] not in done]
    total = len(items)

    runner, recorder, chair = _runner(args.system, args.memory, bool(args.record))
    print(
        "ears=%s teacher=%s n=%d remaining=%d workers=%d -> %s"
        % (
            getattr(chair, "model", "?"),
            getattr(chair, "teacher", "?"),
            total,
            len(remaining),
            workers,
            args.results,
        ),
        flush=True,
    )

    finished = len(done)
    lock = threading.Lock()

    def score(item: dict) -> dict:
        asked = format_item(item)
        started = time.time()
        said = runner(asked)
        elapsed = time.time() - started
        choice = extract_choice(said, item.get("options") or ())
        return {
            "system": args.system,
            "question_id": item["question_id"],
            "category": item.get("category"),
            "gold": item.get("answer"),
            "choice": choice,
            "correct": grade(item, choice),
            "said": (said or "")[:500],
            "seconds": round(elapsed, 2),
            "ears": getattr(chair, "model", None),
            "teacher": getattr(chair, "teacher", None),
        }

    def commit(row: dict) -> None:
        nonlocal finished
        with lock:
            done[row["question_id"]] = row
            _save(args.results, [row])
            if recorder is not None:
                _save_recording(args.record, recorder, done)
            finished += 1
            n = finished
        mark = "ok" if row["correct"] else ("abstain" if row["choice"] is None else "wrong")
        print(
            "[%d/%d] #%s %s %s  gold=%s got=%s  %.1fs"
            % (n, total, row["question_id"], row.get("category"), mark, row.get("gold"), row["choice"], row["seconds"]),
            flush=True,
        )
        print("      %s" % ((row.get("said") or "").replace("\n", " ")[:160],), flush=True)

    if workers <= 1:
        for item in remaining:
            commit(score(item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score, item) for item in remaining]
            for fut in as_completed(futures):
                commit(fut.result())

    scored = [r for r in done.values() if r["system"] == args.system]
    n = len(scored)
    hits = sum(1 for r in scored if r["correct"])
    abstain = sum(1 for r in scored if r["choice"] is None)
    print()
    print(
        "%s  %d/%d correct (%.1f%%)  %d abstain  %d wrong"
        % (args.system, hits, n, 100.0 * hits / n if n else 0, abstain, n - hits - abstain),
        flush=True,
    )
    return 0


def _runner(system: str, memory: str, record: bool):
    if system == "soup":
        seat = Seat(embed=True, model=DEFAULT_MODEL, teacher=DEFAULT_TEACHER, timeout=300)
        recorder = RecordingSeat(seat) if record else None
        chair = recorder if recorder is not None else seat
        session = Session(memory_path=memory, llm=chair)

        def ask(text: str) -> str:
            reply = session.respond(text)
            session.save()
            return reply.said

        return ask, recorder, seat

    names = {
        "qwen4b": "qwen3.5:4b",
        "qwen9b": "qwen3.5:9b",
        "qwen27b": "qwen3.8:27b",
    }
    model = names[system]
    seat = Seat(embed=True, model=model, teacher=model, timeout=300)
    prompt = (
        "Answer the following multiple choice question. "
        "The last line of your response should be in the form: Answer: A"
    )

    def ask(text: str) -> str:
        return seat._ask(prompt, text, model=model, max_tokens=2048) or ""

    return ask, None, seat


def _save_recording(path: str, recorder, rows: dict) -> None:
    """What the models said, plus the score it led to. Replayed by the tests."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "transcript": recorder.transcript,
                "items": [
                    {
                        "question_id": r["question_id"],
                        "gold": r["gold"],
                        "choice": r["choice"],
                        "correct": r["correct"],
                        "said": r["said"],
                    }
                    for r in rows.values()
                ],
            },
            fh,
            indent=2,
        )
        fh.write("\n")


def _load_done(path: str, system: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        raw = fh.read()
    if not raw.strip():
        return {}
    rows = json.loads(raw)
    return {r["question_id"]: r for r in rows if r.get("system") == system}


def _save(path: str, rows: list) -> None:
    """Merge rows into the scoreboard. Safe for several processes at once."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read()
        existing = json.loads(raw) if raw.strip() else []
        by_key = {(r.get("system"), r.get("question_id")): r for r in existing}
        for row in rows:
            by_key[(row["system"], row["question_id"])] = row
        fh.seek(0)
        fh.truncate()
        json.dump(list(by_key.values()), fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
