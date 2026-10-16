"""Ask AI judges whether each candidate query is correct.

    python -m wjury.judges --models codegemma:7b,phi4:14b --backend ollama

Like the paper, each judge sees the question, the schema and the
candidate SQL, and returns a binary verdict. Unlike the paper, it also
reports how confident it is, which the weighted juries can use.

Every answer is appended to results/votes.jsonl as soon as it arrives, so
the run can be stopped and restarted at any time without losing work.
Models are run one after another (all candidates for model A, then B) so a
local model stays loaded in memory.

Backends
  ollama  local models through Ollama (http://localhost:11434)
  openai  any OpenAI-compatible endpoint (OpenRouter, Groq, Together, a
          local vLLM or LM Studio server, ...); key read from an env variable
  sim     SIMULATED judges for testing the pipeline only (see simulate.py)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .db import SCHEMA_FOR_PROMPT

ROOT = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = "You are a careful database engineer who reviews SQL queries written by others."

# Constant text first, so local servers can reuse the cached prefix between calls.
USER_TEMPLATE = """You will review a SQL query written for the SQLite database below.

Decide whether the query correctly answers the question. Then say how confident you are in
your decision, from 0 (pure guess) to 100 (completely certain).

Reply with only this JSON object and nothing else:
{{"verdict": "CORRECT" or "INCORRECT", "confidence": <integer from 0 to 100>}}

Database schema:
{schema}

Question:
{question}

Candidate query:
{sql}
"""


def build_messages(question: str, sql: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=SCHEMA_FOR_PROMPT,
                                                         question=question, sql=sql)},
    ]


# ---------------------------------------------------------------- parsing

_VERDICT = re.compile(r'"?verdict"?\s*[:=]\s*"?\s*(INCORRECT|CORRECT|NOT\s+CORRECT|WRONG|RIGHT)',
                      re.IGNORECASE)
_CONF = re.compile(r'"?confidence"?\s*[:=]\s*"?\s*(\d{1,3}(?:\.\d+)?)', re.IGNORECASE)


def parse_reply(text: str) -> tuple[int | None, float | None]:
    """Return (verdict, confidence) with verdict 1=CORRECT, 0=INCORRECT, None if unreadable.

    Confidence is the judge's stated certainty in its own verdict, scaled to 0..1.
    """
    if not text:
        return None, None
    # Reasoning models may think out loud first: look at the last JSON-ish object.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    objs = re.findall(r"\{[^{}]*\}", text)
    for blob in reversed(objs):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        v = str(d.get("verdict", "")).strip().upper()
        if v in ("CORRECT", "INCORRECT"):
            c = d.get("confidence")
            try:
                c = min(max(float(c), 0.0), 100.0) / 100.0
            except (TypeError, ValueError):
                c = None
            return (1 if v == "CORRECT" else 0), c
    m = None
    for m in _VERDICT.finditer(text):
        pass
    if m:
        word = re.sub(r"\s+", " ", m.group(1).upper())
        verdict = 1 if word in ("CORRECT", "RIGHT") else 0
        mc = list(_CONF.finditer(text))
        conf = min(float(mc[-1].group(1)), 100.0) / 100.0 if mc else None
        return verdict, conf
    # Last resort: plain prose. Use the last verdict word mentioned.
    words = re.findall(r"\b(NOT\s+CORRECT|INCORRECT|CORRECT)\b", text.upper())
    if words:
        return (1 if words[-1] == "CORRECT" else 0), None
    return None, None


# ---------------------------------------------------------------- backends

# Never send requests for a local model server through a system proxy (common on
# work and university networks, where it makes local calls hang).
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    host = urllib.parse.urlparse(url).hostname or ""
    opener = _NO_PROXY if host in ("localhost", "127.0.0.1", "::1") else urllib.request.build_opener()
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class OllamaJudge:
    def __init__(self, model, host="http://localhost:11434", json_mode=True, think=None,
                 timeout=600, num_ctx=4096):
        self.model, self.host, self.json_mode, self.think = model, host.rstrip("/"), json_mode, think
        self.timeout, self.num_ctx = timeout, num_ctx

    def __call__(self, cand: dict) -> dict:
        return self.chat(build_messages(cand["question"], cand["sql"]))

    def chat(self, messages, json_mode=None) -> dict:
        json_mode = self.json_mode if json_mode is None else json_mode
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        if self.think is not None:
            payload["think"] = self.think
        d = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        msg = d.get("message", {})
        return {"text": msg.get("content", ""),
                "prompt_tokens": d.get("prompt_eval_count"),
                "completion_tokens": d.get("eval_count")}


class OpenAICompatJudge:
    def __init__(self, model, base_url, api_key_env="OPENAI_API_KEY", json_mode=False,
                 timeout=300, max_tokens=None, extra=None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        if not self.key:
            sys.exit(f"Set the {api_key_env} environment variable to your API key.")
        self.json_mode, self.timeout, self.max_tokens = json_mode, timeout, max_tokens
        self.extra = extra or {}

    def __call__(self, cand: dict) -> dict:
        return self.chat(build_messages(cand["question"], cand["sql"]))

    def chat(self, messages, json_mode=None) -> dict:
        json_mode = self.json_mode if json_mode is None else json_mode
        payload = {"model": self.model, "temperature": 0, "messages": messages, **self.extra}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        d = _post_json(f"{self.base_url}/chat/completions", payload,
                       {"Authorization": f"Bearer {self.key}"}, self.timeout)
        choice = d["choices"][0]["message"]
        usage = d.get("usage") or {}
        return {"text": choice.get("content") or "",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}


def make_judge(model: str, args):
    if args.backend == "ollama":
        return OllamaJudge(model, host=args.ollama_host, json_mode=not args.no_json_mode,
                           think=args.think)
    if args.backend == "openai":
        return OpenAICompatJudge(model, args.base_url, args.api_key_env,
                                 json_mode=args.json_mode_openai, max_tokens=args.max_tokens)
    if args.backend == "sim":
        from .simulate import SimulatedJudge
        return SimulatedJudge(model)
    raise ValueError(args.backend)


# ---------------------------------------------------------------- running

def load_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def judge_one(judge, cand, retries=4, keep_chars=500) -> dict:
    delay = 5.0
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            out = judge(cand)
            verdict, conf = parse_reply(out["text"])
            return {"verdict": verdict, "confidence": conf,
                    "raw": out["text"][-keep_chars:],
                    "latency_s": round(out.get("latency_s") or (time.monotonic() - t0), 3),
                    "prompt_tokens": out.get("prompt_tokens"),
                    "completion_tokens": out.get("completion_tokens"), "error": None}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = f"{type(e).__name__}: {e}"
        except Exception as e:  # malformed response etc.: record it, do not crash the run
            err = f"{type(e).__name__}: {e}"
            break
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return {"verdict": None, "confidence": None, "raw": "", "latency_s": None,
            "prompt_tokens": None, "completion_tokens": None, "error": err}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--backend", choices=["ollama", "openai", "sim"], default="ollama")
    ap.add_argument("--candidates", default=str(ROOT / "data" / "candidates.jsonl"),
                    help="one or more JSONL files, comma-separated")
    ap.add_argument("--out", default=str(ROOT / "results" / "votes.jsonl"))
    ap.add_argument("--tasks", default="", help="only these task ids, comma-separated (quick tests)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N candidates (quick tests)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (API backends)")
    ap.add_argument("--retry-unparsed", action="store_true",
                    help="re-ask judges whose earlier reply could not be read")
    # ollama
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--no-json-mode", action="store_true", help="do not force JSON output in Ollama")
    ap.add_argument("--think", default=None,
                    help="reasoning setting passed to Ollama, e.g. low / medium / high for gpt-oss")
    # openai-compatible
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--json-mode-openai", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args(argv)

    cands = [c for f in args.candidates.split(",") for c in load_jsonl(f)]
    if args.tasks:
        keep = set(args.tasks.split(","))
        cands = [c for c in cands if c["task_id"] in keep]
    if args.limit:
        cands = cands[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    for r in load_jsonl(out):
        if r.get("prompt_version") != PROMPT_VERSION:
            continue
        if r["verdict"] is not None or not args.retry_unparsed:
            if r.get("error") is None:
                done[(r["cid"], r["model"])] = r

    lock = Lock()
    with out.open("a", encoding="utf-8") as f:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            judge = make_judge(model, args)
            todo = [c for c in cands if (c["cid"], model) not in done]
            print(f"\n== {model}: {len(cands) - len(todo)} already done, {len(todo)} to go "
                  f"(the first answer can take a few minutes while the model loads)", flush=True)
            stats = {"n": 0, "unparsed": 0, "errors": 0, "t0": time.monotonic()}

            def work(c, model=model, judge=judge, stats=stats):
                res = judge_one(judge, c)
                rec = {"cid": c["cid"], "model": model, "prompt_version": PROMPT_VERSION,
                       "backend": args.backend, **res}
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    stats["n"] += 1
                    stats["errors"] += res["error"] is not None
                    stats["unparsed"] += res["error"] is None and res["verdict"] is None
                    if stats["n"] % 10 == 0 or stats["n"] == len(todo):
                        el = time.monotonic() - stats["t0"]
                        eta = el / stats["n"] * (len(todo) - stats["n"])
                        print(f"  {stats['n']}/{len(todo)}  unreadable={stats['unparsed']} "
                              f"errors={stats['errors']}  elapsed={el/60:.1f}m  eta={eta/60:.1f}m",
                              flush=True)

            if args.workers > 1:
                with ThreadPoolExecutor(args.workers) as ex:
                    list(ex.map(work, todo))
            else:
                for c in todo:
                    work(c)
            if stats["errors"]:
                print(f"  ! {stats['errors']} calls failed; run the same command again to retry them")


if __name__ == "__main__":
    main()
