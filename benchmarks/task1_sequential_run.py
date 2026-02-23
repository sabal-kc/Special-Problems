"""
Traditional Task 1 sequential benchmark runner (YAML-driven).

Usage:
  python3 benchmarks/task1_sequential_run.py
  python3 benchmarks/task1_sequential_run.py --config benchmarks/task1_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyYAML. Install with `pip install -r requirements.txt`."
    ) from exc


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must parse to an object")
    return data


def load_text_batch(csv_path: Path, batch_size: int) -> list[str]:
    texts: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} must include a 'text' column")
        for row in reader:
            if len(texts) >= batch_size:
                break
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
    if len(texts) < batch_size:
        raise ValueError(f"Requested batch_size={batch_size}, but only found {len(texts)} usable rows")
    return texts


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = resp.getcode()
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response")
    return status, parsed


def invoke_with_retry(
    fn: Callable[[], dict[str, Any]],
    max_attempts: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            status = exc.code
            is_retryable = status == 429 or status >= 500
            last_error = f"HTTPError {status}: {exc.reason}"
            if not is_retryable or attempt == max_attempts:
                break
            time.sleep(backoff_seconds)
        except urllib.error.URLError as exc:
            last_error = f"URLError: {exc.reason}"
            if attempt == max_attempts:
                break
            time.sleep(backoff_seconds)
        except Exception as exc:  # pragma: no cover
            last_error = f"{type(exc).__name__}: {exc}"
            break
    raise RuntimeError(last_error or "Unknown invocation error")


def call_traditional(
    endpoint: str,
    texts: list[str],
    timeout_s: float,
) -> dict[str, Any]:
    payload = {"texts": texts}
    status, data = post_json(endpoint, payload, timeout_s)
    if status != 200:
        raise RuntimeError(f"Traditional service returned status {status}")

    preds = data.get("predictions")
    if not isinstance(preds, list) or len(preds) != len(texts):
        raise ValueError("Traditional response has invalid predictions length")

    labels = [p.get("label") for p in preds if isinstance(p, dict)]
    if len(labels) != len(texts):
        raise ValueError("Traditional response contains invalid prediction objects")
    if any(l not in {"positive", "negative"} for l in labels):
        raise ValueError("Traditional response contains invalid sentiment label")

    server_ms = data.get("timing", {}).get("server_ms")
    return {
        "server_ms": float(server_ms) if server_ms is not None else None,
        "http_status": status,
    }


def run_request(
    endpoint: str,
    texts: list[str],
    timeout_s: float,
    max_attempts: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = invoke_with_retry(
            lambda: call_traditional(endpoint=endpoint, texts=texts, timeout_s=timeout_s),
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        success = True
        error = ""
        http_status = result["http_status"]
        server_ms = result["server_ms"]
    except Exception as exc:
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        success = False
        error = f"{type(exc).__name__}: {exc}"
        http_status = None
        server_ms = None

    e2e_s = e2e_ms / 1000.0
    samples_per_sec = (len(texts) / e2e_s) if success and e2e_s > 0 else 0.0

    return {
        "e2e_ms": round(e2e_ms, 4),
        "server_ms": server_ms,
        "samples_per_sec": round(samples_per_sec, 4),
        "success": success,
        "http_status": http_status,
        "error": error,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch_size"]), []).append(row)

    out: list[dict[str, Any]] = []
    for batch_size in sorted(by_batch.keys()):
        group = by_batch[batch_size]
        success_rows = [r for r in group if r["success"]]
        e2e_values = [float(r["e2e_ms"]) for r in success_rows]
        sps_values = [float(r["samples_per_sec"]) for r in success_rows]
        server_values = [float(r["server_ms"]) for r in success_rows if r["server_ms"] is not None]
        success_rate = (len(success_rows) / len(group) * 100.0) if group else 0.0

        out.append(
            {
                "batch_size": batch_size,
                "runs": len(group),
                "success_runs": len(success_rows),
                "success_rate_pct": round(success_rate, 2),
                "e2e_ms_mean": round(statistics.mean(e2e_values), 4) if e2e_values else None,
                "e2e_ms_p50": round(percentile(e2e_values, 0.50), 4) if e2e_values else None,
                "e2e_ms_p95": round(percentile(e2e_values, 0.95), 4) if e2e_values else None,
                "e2e_ms_min": round(min(e2e_values), 4) if e2e_values else None,
                "e2e_ms_max": round(max(e2e_values), 4) if e2e_values else None,
                "samples_per_sec_mean": round(statistics.mean(sps_values), 4) if sps_values else None,
                "server_ms_mean": round(statistics.mean(server_values), 4) if server_values else None,
            }
        )
    return out


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write in summary")
    fieldnames = [
        "batch_size",
        "runs",
        "success_runs",
        "success_rate_pct",
        "e2e_ms_mean",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_min",
        "e2e_ms_max",
        "samples_per_sec_mean",
        "server_ms_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Task 1 traditional sequential benchmark.")
    parser.add_argument(
        "--config",
        default=str(script_dir / "task1_config.yaml"),
        help="Path to benchmark YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    project_root = config_path.parent.parent
    endpoint = str(config["traditional"]["endpoint"])
    timeout_s = float(config["run"]["timeout_seconds"])
    retry_cfg = config["run"].get("retry", {})
    max_attempts = int(retry_cfg.get("max_attempts", 1))
    backoff_seconds = float(retry_cfg.get("backoff_seconds", 0.0))

    seq_cfg = config["sequential"]
    batch_sizes = [int(x) for x in seq_cfg["batch_sizes"]]
    runs_per_batch = int(seq_cfg["runs_per_batch"])

    input_csv = project_root / config["client"]["sentiment_csv_path"]
    out_dir = project_root / config["client"]["output_dir"]
    raw_path = out_dir / "task1_seq_raw.jsonl"
    summary_path = out_dir / "task1_seq_summary.csv"

    run_rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        texts = load_text_batch(input_csv, batch_size=batch_size)
        for run_index in range(1, runs_per_batch + 1):
            print(f"Running batch size {batch_size} run {run_index} of {runs_per_batch}...")
            result = run_request(
                endpoint=endpoint,
                texts=texts,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
            row = {
                "ts_utc": utc_now_iso(),
                "phase": "sequential",
                "batch_size": batch_size,
                "run_index": run_index,
                **result,
            }
            run_rows.append(row)

    write_jsonl(raw_path, run_rows)
    summary_rows = summarize(run_rows)
    write_summary_csv(summary_path, summary_rows)

    print("Sequential benchmark complete.")
    print(f"Raw runs:    {raw_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
