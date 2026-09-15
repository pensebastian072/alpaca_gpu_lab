"""Batch offline news scoring -> market_data/scores/news_scores.parquet.

STRICTLY offline/batch (CLAUDE.md rule 6): this never runs on any decision
path; downstream features read precomputed numbers and default to neutral.

Primary scorer for the bulk backlog: FinBERT (ProsusAI/finbert) on CUDA —
fits easily in 6 GB and is orders of magnitude faster than an LLM for ~1e5
headlines. Spot-audit scorer: local Ollama qwen2.5:7b via the OpenAI-compat
endpoint (hq analytics/hf_client.py pattern) — used on samples to sanity-check
FinBERT, never for the bulk.

Idempotent by news id: already-scored ids are skipped on re-runs.
Near-duplicate headlines are deduped with difflib ratio >= 0.86 within a day
(hq news_poller.py trick) — reprints score once, feature counts stay honest.

CLI:
  python -m src.news.score                # score everything unscored (FinBERT)
  python -m src.news.score --audit 25    # Ollama spot-audit of 25 random scored rows
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

SCORES_PATH = config.SCORES / "news_scores.parquet"
FINBERT = "ProsusAI/finbert"
# main branch has only pytorch_model.bin; transformers>=5 refuses torch.load
# on torch<2.6 (CVE-2025-32434) and upgrading torch breaks the cu121 stack.
# refs/pr/29 is the official safetensors conversion of the same weights.
FINBERT_REV = "refs/pr/29"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEDUP_RATIO = 0.86
HIGH_IMPACT_KEYWORDS = (
    "fed", "fomc", "rate", "cpi", "inflation", "jobs report", "nonfarm",
    "recession", "default", "crash", "bankrupt", "war", "tariff", "downgrade",
)


def load_raw_news() -> pd.DataFrame:
    files = sorted((config.RAW / "news").glob("year=*/month=*/news.parquet"))
    if not files:
        raise FileNotFoundError("no raw news — run src.data.news_backfill first")
    df = pl.concat([pl.read_parquet(f) for f in files]).to_pandas()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")
    return df.drop_duplicates(subset="id").sort_values("created_at")


def _dedup_day(day: pd.DataFrame) -> pd.DataFrame:
    """Drop near-duplicate headlines within one day (reprints/syndication)."""
    keep, kept_texts = [], []
    for idx, h in zip(day.index, day["headline"].fillna("")):
        if any(difflib.SequenceMatcher(None, h.lower(), t).ratio() >= DEDUP_RATIO
               for t in kept_texts):
            continue
        keep.append(idx)
        kept_texts.append(h.lower())
    return day.loc[keep]


def score_finbert(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """score in [-1, 1] = P(pos) - P(neg); impact flag from keywords."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(FINBERT, revision=FINBERT_REV)
    model = (AutoModelForSequenceClassification
             .from_pretrained(FINBERT, revision=FINBERT_REV).to(device).eval())
    # ProsusAI/finbert label order: positive, negative, neutral
    texts = (df["headline"].fillna("") + ". " + df["summary"].fillna("")).str.slice(0, 512).tolist()
    scores = np.empty(len(texts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=128).to(device)
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu().numpy()
            scores[i:i + batch_size] = probs[:, 0] - probs[:, 1]
    out = df[["id", "created_at", "symbols"]].copy()
    out["score"] = scores
    low = (df["headline"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    out["high_impact"] = low.str.contains("|".join(HIGH_IMPACT_KEYWORDS), regex=True)
    out["scorer"] = "finbert"
    return out


def run_bulk() -> None:
    # TLS-interception box: export Windows CA bundle BEFORE transformers pulls
    # the model from huggingface.co (same fix as Kronos HF downloads)
    from src.data.alpaca_backfill import _trust_windows_certs
    _trust_windows_certs()

    raw = load_raw_news()
    raw = (raw.groupby(raw["created_at"].dt.date, group_keys=False)
              .apply(_dedup_day))
    done_ids: set[str] = set()
    if SCORES_PATH.exists():
        done_ids = set(pl.read_parquet(SCORES_PATH)["id"].to_list())
    todo = raw[~raw["id"].isin(done_ids)]
    if todo.empty:
        print("nothing new to score")
        return
    print(f"scoring {len(todo)} articles (FinBERT, cuda if available)...")
    scored = score_finbert(todo)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SCORES_PATH.exists():
        scored = pd.concat([pl.read_parquet(SCORES_PATH).to_pandas(), scored])
    scored.drop_duplicates(subset="id").to_parquet(SCORES_PATH, index=False)
    print(f"total scored: {len(scored)} -> {SCORES_PATH}")


def audit_ollama(n: int = 25) -> None:
    """Spot-audit: agreement rate between FinBERT sign and qwen2.5:7b sign."""
    scored = pl.read_parquet(SCORES_PATH).to_pandas().sample(n, random_state=config.SEED)
    raw = load_raw_news().set_index("id")
    agree = 0
    for _, row in scored.iterrows():
        head = raw.loc[row["id"], "headline"] if row["id"] in raw.index else ""
        body = json.dumps({
            "model": "qwen2.5:7b", "temperature": 0.1, "max_tokens": 8,
            "messages": [
                {"role": "system", "content":
                 "Classify financial headline sentiment. Reply with exactly one "
                 "word: positive, negative, or neutral."},
                {"role": "user", "content": str(head)},
            ],
        }).encode()
        try:
            req = urllib.request.Request(OLLAMA_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                word = json.loads(resp.read())["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:  # noqa: BLE001 — audit is best-effort
            print(f"ollama unreachable ({e}); audit aborted")
            return
        llm_sign = 1 if "positive" in word else -1 if "negative" in word else 0
        fb_sign = 1 if row["score"] > 0.15 else -1 if row["score"] < -0.15 else 0
        agree += int(llm_sign == fb_sign)
    print(f"FinBERT vs qwen2.5:7b sign agreement: {agree}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=int, default=0)
    a = ap.parse_args()
    if a.audit:
        audit_ollama(a.audit)
    else:
        run_bulk()
