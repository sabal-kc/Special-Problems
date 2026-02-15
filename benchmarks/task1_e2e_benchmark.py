"""
Task 1 end-to-end benchmark: Traditional EC2 service vs Gemini API.

Usage:
  python3 benchmarks/task1_e2e_benchmark.py
  python3 benchmarks/task1_e2e_benchmark.py --config benchmarks/task1_config.yaml

Environment:
  GEMINI_API_KEY must be set for Gemini runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
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


def as_float(v: Any) -> float:
    return float(v) if v is not None else 0.0


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


def build_gemini_prompt(texts: list[str]) -> str:
    """Build prompt matching prompts.md Task 1 format (CSV input/output)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "text"])
    writer.writeheader()
    for i, t in enumerate(texts, start=1):
        clean = t.replace("\n", " ").strip()
        writer.writerow({"id": i, "text": clean})
    csv_data = buf.getvalue().strip()
    return (
        "You are a sentiment analysis expert. I will give you a CSV with columns \"id\" and \"text\".\n"
        "For each row, classify the sentiment as exactly one of: positive, negative.\n\n"
        "Return your answer as a CSV with exactly two columns: id, label\n"
        "Do NOT include any explanation or extra text. Only output the CSV (with header row).\n\n"
        "Here is the data:\n\n"
        f"{csv_data}"
    )


def call_traditional(
    endpoint: str,
    texts: list[str],
    timeout_s: float,
    request_id: str,
) -> dict[str, Any]:
    def _invoke() -> dict[str, Any]:
        payload = {"request_id": request_id, "texts": texts}
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
            "labels": labels,
            "server_ms": float(server_ms) if server_ms is not None else None,
            "input_tokens": None,
            "output_tokens": None,
            "http_status": status,
        }

    return _invoke()


def call_gemini(
    endpoint_template: str,
    model: str,
    api_key: str,
    texts: list[str],
    timeout_s: float,
) -> dict[str, Any]:
    endpoint = endpoint_template.format(model=model)
    prompt = build_gemini_prompt(texts)
    url = f"{endpoint}?{urllib.parse.urlencode({'key': api_key})}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }
    status, data = post_json(url, payload, timeout_s)
    if status != 200:
        raise RuntimeError(f"Gemini returned status {status}")

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates")

    first = candidates[0]
    content = first.get("content", {}) if isinstance(first, dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response missing content parts")

    # With thinking mode, parts[0] is thinking; the last part is the actual answer.
    text_blob = None
    for p in reversed(parts):
        if isinstance(p, dict) and p.get("text"):
            text_blob = p["text"]
            break
    if not isinstance(text_blob, str):
        raise ValueError("Gemini response missing content parts with text")

    # Parse CSV response (prompts.md format: id, label)
    reader = csv.DictReader(
        io.StringIO(text_blob.strip()),
        fieldnames=["id", "label"],
        restkey=None,
    )
    rows = list(reader)
    id_to_label: dict[int, str] = {}
    for row in rows:
        if "id" in row and "label" in row:
            try:
                rid = int(str(row["id"]).strip())
            except ValueError:
                continue
            id_to_label[rid] = str(row["label"]).strip().lower()
    labels = []
    for i in range(1, len(texts) + 1):
        if i not in id_to_label:
            raise ValueError(f"Gemini output missing label for id {i}")
        lbl = id_to_label[i]
        if lbl not in {"positive", "negative"}:
            raise ValueError(f"Gemini output contains invalid sentiment label: {lbl!r}")
        labels.append(lbl)

    usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")

    return {
        "labels": labels,
        "server_ms": None,
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "http_status": status,
    }


def execute_call(
    system: str,
    phase: str,
    texts: list[str],
    config: dict[str, Any],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    run_cfg = config["run"]
    retry_cfg = run_cfg["retry"]
    request_id = str(uuid.uuid4())
    timeout_s = float(run_cfg["timeout_seconds"])
    max_attempts = int(retry_cfg["max_attempts"])
    backoff_seconds = float(retry_cfg["backoff_seconds"])
    batch_size = len(texts)

    t0 = time.perf_counter()
    try:
        if system == "traditional":
            endpoint = config["traditional"]["endpoint"]
            result = invoke_with_retry(
                lambda: call_traditional(endpoint, texts, timeout_s, request_id),
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        elif system == "gemini":
            gcfg = config["gemini"]
            api_key = os.getenv(gcfg["api_key_env"], "").strip()
            if not api_key:
                raise RuntimeError(f"Missing API key env var: {gcfg['api_key_env']}")
            result = invoke_with_retry(
                lambda: call_gemini(
                    endpoint_template=gcfg["endpoint_template"],
                    model=gcfg["model"],
                    api_key=api_key,
                    texts=texts,
                    timeout_s=timeout_s,
                ),
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        else:
            raise ValueError(f"Unknown system: {system}")
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        success = True
        error = ""
    except Exception as exc:
        result = {
            "labels": [],
            "server_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "http_status": None,
        }
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        success = False
        error = f"{type(exc).__name__}: {exc}"

    e2e_s = e2e_ms / 1000.0
    samples_per_sec = (batch_size / e2e_s) if success and e2e_s > 0 else 0.0

    traditional_hourly = as_float(pricing["traditional"]["ec2_hourly_usd"])
    gemini_in = as_float(pricing["gemini"]["input_token_usd"])
    gemini_out = as_float(pricing["gemini"]["output_token_usd"])
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]

    if system == "traditional":
        infra_cost = traditional_hourly / 3600.0 * e2e_s
        api_cost = 0.0
    else:
        infra_cost = 0.0
        if input_tokens is None or output_tokens is None:
            api_cost = 0.0
        else:
            api_cost = input_tokens * gemini_in + output_tokens * gemini_out

    return {
        "run_id": config["run_meta"]["run_id"],
        "system": system,
        "phase": phase,
        "batch_size": batch_size,
        "e2e_ms": round(e2e_ms, 4),
        "server_ms": result["server_ms"],
        "samples_per_sec": round(samples_per_sec, 4),
        "success": success,
        "error": error,
        "http_status": result["http_status"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_cost_usd": round(api_cost, 8),
        "infra_cost_usd": round(infra_cost, 8),
        "total_cost_usd": round(api_cost + infra_cost, 8),
        "ts_utc": utc_now_iso(),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["system"]), str(row["phase"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (system, phase), group in sorted(grouped.items()):
        success_rows = [r for r in group if r["success"]]
        e2e_values = [float(r["e2e_ms"]) for r in success_rows]
        sps_values = [float(r["samples_per_sec"]) for r in success_rows]
        server_values = [float(r["server_ms"]) for r in success_rows if r["server_ms"] is not None]
        api_values = [float(r["api_cost_usd"]) for r in success_rows]
        infra_values = [float(r["infra_cost_usd"]) for r in success_rows]
        total_values = [float(r["total_cost_usd"]) for r in success_rows]

        success_rate = (len(success_rows) / len(group) * 100.0) if group else 0.0
        summary_rows.append(
            {
                "system": system,
                "phase": phase,
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
                "api_cost_usd_mean": round(statistics.mean(api_values), 8) if api_values else None,
                "infra_cost_usd_mean": round(statistics.mean(infra_values), 8) if infra_values else None,
                "total_cost_usd_mean": round(statistics.mean(total_values), 8) if total_values else None,
            }
        )
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write in summary")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_summary(
    summary_rows: list[dict[str, Any]],
    system: str,
    phase: str,
) -> dict[str, Any] | None:
    for row in summary_rows:
        if row["system"] == system and row["phase"] == phase:
            return row
    return None


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_report(
    report_path: Path,
    config: dict[str, Any],
    pricing: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = config["run_meta"]["run_id"]
    created_utc = config["run_meta"]["created_utc"]

    cold_trad = find_summary(summary_rows, "traditional", "cold")
    cold_gem = find_summary(summary_rows, "gemini", "cold")
    warm_trad = find_summary(summary_rows, "traditional", "warm")
    warm_gem = find_summary(summary_rows, "gemini", "warm")

    gemini_cost_per_req = (
        float(warm_gem["total_cost_usd_mean"])
        if warm_gem and warm_gem.get("total_cost_usd_mean") is not None
        else None
    )
    trad_cost_per_req = (
        float(warm_trad["total_cost_usd_mean"])
        if warm_trad and warm_trad.get("total_cost_usd_mean") is not None
        else None
    )
    ec2_hourly = as_float(pricing["traditional"]["ec2_hourly_usd"])

    scenario_rows: list[tuple[str, int, float | None, float | None, str]] = []
    for scenario in pricing.get("traffic_scenarios", []):
        if not isinstance(scenario, dict):
            continue
        name = str(scenario.get("name", "scenario"))
        req_hour = int(scenario.get("requests_per_hour", 0))
        gem_hour = gemini_cost_per_req * req_hour if gemini_cost_per_req is not None else None
        trad_hour = ec2_hourly if trad_cost_per_req is not None else None
        if trad_hour is None or gem_hour is None:
            cheaper = "n/a"
        else:
            cheaper = "Traditional" if trad_hour < gem_hour else "Gemini"
        scenario_rows.append((name, req_hour, trad_hour, gem_hour, cheaper))

    lines: list[str] = []
    lines.append("# Task 1 E2E Benchmark Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated (UTC): `{created_utc}`")
    lines.append("- Measurement mode: End-to-end from local client")
    lines.append(f"- Workload: fixed batch of {config['run']['batch_size']} sentiment samples")
    lines.append("")

    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Traditional endpoint: `{config['traditional']['endpoint']}`")
    lines.append(f"- Gemini model: `{config['gemini']['model']}`")
    lines.append(f"- Warm runs per system: `{config['run']['warm_runs']}`")
    lines.append(f"- Timeout per call: `{config['run']['timeout_seconds']}s`")
    lines.append("")

    lines.append("## Pricing Snapshot")
    lines.append("")
    lines.append(f"- Snapshot date: `{pricing['pricing_snapshot']['date']}`")
    lines.append(f"- Source note: {pricing['pricing_snapshot']['source']}")
    lines.append(
        f"- Traditional EC2: `{pricing['traditional']['ec2_instance_type']}` "
        f"({pricing['traditional']['ec2_region']}) at `${pricing['traditional']['ec2_hourly_usd']}/hour`"
    )
    lines.append(
        f"- Gemini token rates: input `${pricing['gemini']['input_token_usd']}` / token, "
        f"output `${pricing['gemini']['output_token_usd']}` / token"
    )
    lines.append("")

    lines.append("## Cold Latency (1 run)")
    lines.append("")
    lines.append("| System | Success Rate | E2E ms |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Traditional | {fmt(cold_trad['success_rate_pct'] if cold_trad else None, 2)}% | "
        f"{fmt(cold_trad['e2e_ms_mean'] if cold_trad else None)} |"
    )
    lines.append(
        f"| Gemini | {fmt(cold_gem['success_rate_pct'] if cold_gem else None, 2)}% | "
        f"{fmt(cold_gem['e2e_ms_mean'] if cold_gem else None)} |"
    )
    lines.append("")

    lines.append(f"## Warm Latency and Throughput ({config['run']['warm_runs']} runs)")
    lines.append("")
    lines.append("| System | Success Rate | p50 ms | p95 ms | Mean samples/sec | Traditional server ms (mean) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Traditional | {fmt(warm_trad['success_rate_pct'] if warm_trad else None, 2)}% | "
        f"{fmt(warm_trad['e2e_ms_p50'] if warm_trad else None)} | "
        f"{fmt(warm_trad['e2e_ms_p95'] if warm_trad else None)} | "
        f"{fmt(warm_trad['samples_per_sec_mean'] if warm_trad else None)} | "
        f"{fmt(warm_trad['server_ms_mean'] if warm_trad else None)} |"
    )
    lines.append(
        f"| Gemini | {fmt(warm_gem['success_rate_pct'] if warm_gem else None, 2)}% | "
        f"{fmt(warm_gem['e2e_ms_p50'] if warm_gem else None)} | "
        f"{fmt(warm_gem['e2e_ms_p95'] if warm_gem else None)} | "
        f"{fmt(warm_gem['samples_per_sec_mean'] if warm_gem else None)} | - |"
    )
    lines.append("")

    lines.append("## Cost Per Request (warm mean)")
    lines.append("")
    lines.append("| System | API USD | Infra USD | Total USD |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Traditional | {fmt(warm_trad['api_cost_usd_mean'] if warm_trad else None, 8)} | "
        f"{fmt(warm_trad['infra_cost_usd_mean'] if warm_trad else None, 8)} | "
        f"{fmt(warm_trad['total_cost_usd_mean'] if warm_trad else None, 8)} |"
    )
    lines.append(
        f"| Gemini | {fmt(warm_gem['api_cost_usd_mean'] if warm_gem else None, 8)} | "
        f"{fmt(warm_gem['infra_cost_usd_mean'] if warm_gem else None, 8)} | "
        f"{fmt(warm_gem['total_cost_usd_mean'] if warm_gem else None, 8)} |"
    )
    lines.append("")

    lines.append("## Decision Table (hourly cost interpretation)")
    lines.append("")
    lines.append("| Traffic | Requests/hour | Traditional USD/hour | Gemini USD/hour | Cheaper |")
    lines.append("|---|---:|---:|---:|---|")
    for name, req_hour, trad_hour, gem_hour, cheaper in scenario_rows:
        trad_hour_s = f"{trad_hour:.6f}" if trad_hour is not None else "-"
        gem_hour_s = f"{gem_hour:.6f}" if gem_hour is not None else "-"
        lines.append(f"| {name} | {req_hour} | {trad_hour_s} | {gem_hour_s} | {cheaper} |")
    if not scenario_rows:
        lines.append("| - | - | - | - | - |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- This benchmark is local-client end-to-end and includes network effects.")
    lines.append("- Traditional and Gemini quality parity is not evaluated in this report.")
    lines.append("- Cost is sensitive to token usage and configured pricing values.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    _script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Task 1 end-to-end latency and cost benchmark.")
    parser.add_argument(
        "--config",
        default=str(_script_dir / "task1_config.yaml"),
        help="Path to benchmark YAML config",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["traditional", "gemini"],
        default=["traditional", "gemini"],
        help="Systems to benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    # Paths in config are relative to project root (parent of benchmarks/)
    project_root = config_path.parent.parent
    pricing_path = project_root / config["pricing"]["file"]
    pricing = load_yaml(pricing_path)

    config["run_meta"] = {"run_id": str(uuid.uuid4()), "created_utc": utc_now_iso()}

    batch_size = int(config["run"]["batch_size"])
    input_csv_path = project_root / config["client"]["sentiment_csv_path"]
    texts = load_text_batch(input_csv_path, batch_size=batch_size)

    run_rows: list[dict[str, Any]] = []
    systems = config["run"].get("systems", args.systems)
    if isinstance(systems, str):
        systems = [s.strip() for s in systems.split(",") if s.strip()]
    for s in systems:
        if s not in ("traditional", "gemini"):
            raise ValueError(f"Invalid system in config: {s!r}. Use 'traditional' and/or 'gemini'")

    # Cold phase: one call per selected system.
    for system in systems:
        row = execute_call(system=system, phase="cold", texts=texts, config=config, pricing=pricing)
        run_rows.append(row)

    # Warm phase: alternating order to reduce order bias.
    warm_runs = int(config["run"]["warm_runs"])
    for i in range(warm_runs):
        if len(systems) == 2 and i % 2 == 1:
            order = list(reversed(systems))
        else:
            order = systems
        for system in order:
            row = execute_call(system=system, phase="warm", texts=texts, config=config, pricing=pricing)
            run_rows.append(row)

    out_dir = project_root / config["client"]["output_dir"]
    raw_path = out_dir / "task1_raw.jsonl"
    summary_path = out_dir / "task1_summary.csv"
    report_path = out_dir / "task1_report.md"

    write_jsonl(raw_path, run_rows)
    summary_rows = summarize_rows(run_rows)
    write_summary_csv(summary_path, summary_rows)
    write_report(report_path, config=config, pricing=pricing, summary_rows=summary_rows)

    print("Benchmark complete.")
    print(f"Raw runs:    {raw_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Report MD:   {report_path}")


if __name__ == "__main__":
    main()
