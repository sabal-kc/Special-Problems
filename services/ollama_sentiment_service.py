"""
Ollama-based sentiment service for Task 1 benchmarking.

Same endpoint as traditional_sentiment_service:
  POST /v1/sentiment

Uses Ollama with llama3.2:1b-instruct-q4_K_M model instead of VADER.

Request JSON:
  {
    "request_id": "optional-string",
    "texts": ["sentence 1", "sentence 2", ...]
  }

Response JSON:
  {
    "request_id": "...",
    "model": "llama3.2:1b-instruct-q4_K_M",
    "predictions": [{"label": "positive|negative", "score": 0.99}, ...],
    "timing": {"server_ms": 12.34}
  }
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ollama

# MODEL_NAME = "llama3.2:1b-instruct-q4_K_M"
MODEL_NAME = "llama3.2:1b-instruct-q2_K"

SENTIMENT_PROMPT = """You are a sentiment analysis expert. I will give you a CSV with columns "id" and "text".
For each row, classify the sentiment as exactly one of: positive, negative.

Return your answer as a CSV with exactly two columns: id, label
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

{input_csv}"""


class OllamaSentimentService:
    def __init__(self) -> None:
        pass

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        # Build CSV input: id,text
        rows: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts)]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "text"])
        for idx, text in rows:
            writer.writerow([idx, text])
        input_csv = buf.getvalue()

        prompt = SENTIMENT_PROMPT.format(input_csv=input_csv)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "top_p": 0.1},
        )
        content = response.get("message", {}).get("content", "") or ""

        predictions = self._parse_response(content, len(texts))
        return predictions

    def _parse_response(self, content: str, expected_count: int) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        id_to_result: dict[int, str] = {}

        # Extract CSV block (may be wrapped in markdown or extra text)
        content = content.strip()
        # Try to find CSV block
        csv_match = re.search(r"id\s*,\s*label\s*\n(.*?)(?:\n\n|$)", content, re.DOTALL | re.IGNORECASE)
        if csv_match:
            csv_text = "id,label\n" + csv_match.group(1).strip()
        elif "id" in content.lower() and "label" in content.lower():
            lines = content.splitlines()
            header_idx = -1
            for i, line in enumerate(lines):
                if re.match(r"id\s*,\s*label", line, re.IGNORECASE):
                    header_idx = i
                    break
            if header_idx >= 0:
                csv_text = "\n".join(lines[header_idx:])
            else:
                csv_text = content
        else:
            csv_text = content

        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                id_str = row.get("id", "").strip()
                label_raw = row.get("label", "").strip().lower()
                try:
                    idx = int(id_str)
                except ValueError:
                    continue
                if "pos" in label_raw or label_raw == "positive":
                    label = "positive"
                    score = 0.99
                elif "neg" in label_raw or label_raw == "negative":
                    label = "negative"
                    score = -0.99
                else:
                    label = "positive"
                    score = 0.0
                id_to_result[idx] = {"label": label, "score": round(score, 4)}
        except Exception:
            pass

        for i in range(expected_count):
            if i in id_to_result:
                predictions.append(id_to_result[i])
            else:
                predictions.append({"label": "positive", "score": 0.0})

        print(predictions)
        return predictions


SERVICE = OllamaSentimentService()


def _json_error(message: str) -> dict[str, str]:
    return {"error": message}


class SentimentRequestHandler(BaseHTTPRequestHandler):
    server_version = "OllamaSentimentService/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
            return
        self._send_json(404, _json_error("Not found"))

    def do_POST(self) -> None:
        if self.path != "/v1/sentiment":
            self._send_json(404, _json_error("Not found"))
            return

        content_length_raw = self.headers.get("Content-Length")
        if not content_length_raw:
            self._send_json(400, _json_error("Missing Content-Length"))
            return

        try:
            content_length = int(content_length_raw)
        except ValueError:
            self._send_json(400, _json_error("Invalid Content-Length"))
            return

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json(400, _json_error("Request body must be valid JSON"))
            return

        request_id = payload.get("request_id")
        texts = payload.get("texts")

        if request_id is None:
            request_id = str(uuid.uuid4())
        if not isinstance(request_id, str):
            self._send_json(400, _json_error("'request_id' must be a string if provided"))
            return
        if not isinstance(texts, list) or not texts:
            self._send_json(400, _json_error("'texts' must be a non-empty list of strings"))
            return
        if not all(isinstance(t, str) for t in texts):
            self._send_json(400, _json_error("'texts' must contain only strings"))
            return

        t0 = time.perf_counter()
        try:
            predictions = SERVICE.classify(texts)
        except Exception as exc:
            self._send_json(500, _json_error(f"Ollama error: {exc}"))
            return
        server_ms = (time.perf_counter() - t0) * 1000.0

        response = {
            "request_id": request_id,
            "model": MODEL_NAME,
            "predictions": predictions,
            "timing": {"server_ms": round(server_ms, 4)},
        }
        self._send_json(200, response)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ollama sentiment HTTP service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SentimentRequestHandler)
    print(f"Ollama sentiment service listening on http://{args.host}:{args.port}")
    print("Model:", MODEL_NAME)
    print("Health: GET /health")
    print("Predict: POST /v1/sentiment")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
