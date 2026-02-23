#!/usr/bin/env python3
"""
Call Ollama sentiment service locally in batches of 5 and write all predictions
to a single JSON file.

Usage:
  # Start the service first: python services/ollama_sentiment_service.py
  python scripts/run_sentiment_batch_to_json.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000/v1/sentiment"
INPUT_CSV = Path(__file__).resolve().parent.parent / "llm_input/sentiment_input_1000.csv"
LLAMA_OUTPUT_JSON = Path(__file__).resolve().parent.parent / "llama_output/sentiment_1000.json"
BATCH_SIZE = 10
TIMEOUT_SEC = 480


def load_texts(csv_path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} must have a 'text' column")
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                idx = int(row.get("id", len(rows)))
                rows.append((idx, text))
    return rows


def post_sentiment(texts: list[str]) -> list[dict]:
    payload = {"texts": texts}
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    preds = parsed.get("predictions") or []
    if len(preds) != len(texts):
        raise ValueError(f"Expected {len(texts)} predictions, got {len(preds)}")
    return preds


def main() -> None:
    print(f"Loading texts from {INPUT_CSV}...")
    rows = load_texts(INPUT_CSV)
    print(f"Loaded {len(rows)} samples. Batches of {BATCH_SIZE} = {(len(rows) + BATCH_SIZE - 1) // BATCH_SIZE} requests.")

    all_predictions: list[dict] = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        print(f"  Batch {i // BATCH_SIZE + 1}: ids {ids[0]}-{ids[-1]}...", end=" ", flush=True)
        preds = post_sentiment(texts)
        for (idx, text), pred in zip(batch, preds):
            all_predictions.append({
                "id": idx,
                "text": text,
                "label": pred.get("label", "positive"),
                "score": pred.get("score", 0.0),
            })
        print("done")

    LLAMA_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with LLAMA_OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(all_predictions)} predictions to {LLAMA_OUTPUT_JSON}")


if __name__ == "__main__":
    main()
