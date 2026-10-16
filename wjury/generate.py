"""Optional: let models WRITE the SQL, to add natural mistakes like the paper's.

    python -m wjury.generate --models codegemma:7b,phi4:14b

Each model answers every question once (temperature 0), as in the paper.
Each answer is labeled by execution on all ten databases and saved to
data/generated.jsonl. Judge these together with the planted bugs by passing
both files:  --candidates data/candidates.jsonl,data/generated.jsonl

Answers that do not run at all are kept out (a judge would reject them
trivially), but counted in the printout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .db import DEFAULT_SEEDS, SCHEMA_FOR_PROMPT
from .execution import ExecError, differing_seeds, lenient_same, run
from .judges import judge_one, load_jsonl, make_judge
from .mutate import normalize
from .tasks import ALL_TASKS

ROOT = Path(__file__).resolve().parent.parent

GEN_SYSTEM = "You are an expert SQL developer."
GEN_TEMPLATE = """Write one SQLite query that answers the question below for this database.
Return only the SQL query, with no explanation.

Database schema:
{schema}

Question:
{question}
"""


def extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = m[-1] if m else text
    m = re.search(r"\b(WITH|SELECT)\b", sql, flags=re.IGNORECASE)
    if m:
        sql = sql[m.start():]
    return sql.split(";")[0].strip()


class _Generator:
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, cand):
        msgs = [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": GEN_TEMPLATE.format(schema=SCHEMA_FOR_PROMPT,
                                                                question=cand["question"])}]
        return self.backend.chat(msgs, json_mode=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated.jsonl"))
    ap.add_argument("--raw-out", default=str(ROOT / "results" / "generated_raw.jsonl"))
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--think", default=None)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--json-mode-openai", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args(argv)

    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["model"], r["task_id"]): r for r in load_jsonl(raw_path) if not r.get("error")}

    # Safety: data/generated.jsonl is what the judges voted on. If it holds answers whose raw
    # text is not in raw_out (e.g. a fresh clone, where only the labeled file is committed),
    # regenerating would silently replace judged queries with new ones and make the votes stale.
    out_path = Path(args.out)
    if out_path.exists():
        missing = [c["cid"] for c in load_jsonl(out_path)
                   if (c["operator"], c["task_id"]) not in done]
        if missing:
            raise SystemExit(f"{out_path} already holds {len(missing)} judged answers that are not in "
                             f"{raw_path}. Regenerating would replace them and make the saved votes "
                             f"stale. To start over, delete {out_path} and the votes for those queries "
                             f"first, or pass a different --out.")

    with raw_path.open("a", encoding="utf-8") as f:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            gen = _Generator(make_judge(model, args))
            for t in ALL_TASKS:
                if (model, t.id) in done:
                    continue
                res = judge_one(gen, {"question": t.question}, keep_chars=20000)
                rec = {"model": model, "task_id": t.id, "text": res["raw"], "error": res["error"]}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(model, t.id)] = rec
                print(f"  {model} {t.id} {'ERROR ' + res['error'] if res['error'] else 'ok'}", flush=True)

    # label every stored answer by execution
    stats = Counter()
    out = []
    by_id = {t.id: t for t in ALL_TASKS}
    for (model, tid), rec in sorted(done.items()):
        t = by_id[tid]
        sql = extract_sql(rec["text"])
        try:
            for s in DEFAULT_SEEDS:
                run(sql, s)
        except ExecError:
            stats["does_not_run"] += 1
            continue
        try:
            shown = normalize(sql)
            if differing_seeds(shown, sql):
                shown = sql
        except Exception:
            shown = sql
        wrong = bool(differing_seeds(sql, t.gold))
        stats["wrong" if wrong else "correct"] += 1
        out.append({"cid": f"G-{model}-{tid}", "task_id": tid, "tier": t.tier, "question": t.question,
                    "sql": shown, "label": 0 if wrong else 1,
                    "label_lenient": 1 if (not wrong or lenient_same(sql, t.gold)) else 0,
                    "source": "generated",
                    "operator": model, "bug_type": "natural" if wrong else None, "differs_on": 0})
    Path(args.out).write_text("".join(json.dumps(c) + "\n" for c in out), encoding="utf-8")
    print(f"wrote {len(out)} generated candidates to {args.out}: {dict(stats)}")


if __name__ == "__main__":
    main()
