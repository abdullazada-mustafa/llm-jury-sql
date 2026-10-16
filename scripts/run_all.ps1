# Full run on Windows. Safe to stop (Ctrl+C) and run again: finished answers are kept.
# With the committed votes in results/votes.jsonl, every judging step is skipped and only
# the analysis runs. To redo everything, delete results/votes.jsonl, results/generated_raw.jsonl
# and data/generated.jsonl.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$models = "codegemma:7b,phi4:14b,gemma3:12b,codellama:13b"

# 1. Judges review the 273 planted-bug queries (gpt-oss: no JSON mode, low reasoning effort)
python -m wjury.judges --models $models
python -m wjury.judges --models "gpt-oss:20b" --no-json-mode --think low

# 2. The same models write SQL for all 82 questions, and the judges review it
if (-not (Test-Path data\generated.jsonl)) {
    python -m wjury.generate --models $models
    python -m wjury.generate --models "gpt-oss:20b" --think low
}
python -m wjury.judges --models $models --candidates data\generated.jsonl
python -m wjury.judges --models "gpt-oss:20b" --no-json-mode --think low --candidates data\generated.jsonl

# 3. Analysis
python -m wjury.analyze --out results\planted
python -m wjury.analyze --candidates data\generated.jsonl --out results\natural
python -m wjury.analyze --candidates data\generated.jsonl --out results\natural_lenient --label lenient
python -m wjury.analyze --candidates data\generated.jsonl --out results\natural_audited --label audited
python -m wjury.followups
python -m wjury.followups --label audited --out results\natural_audited\followups
Write-Host "`nDone. Open results\natural\summary.md"
