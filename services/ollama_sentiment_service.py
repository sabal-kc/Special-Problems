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
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ollama

DEBUG = False  # Set True for local development to print debug logs


def _debug(msg: str, *args: Any) -> None:
    if DEBUG:
        print(msg % args if args else msg)

# MODEL_NAME = "llama3.2:1b-instruct-q4_K_M"
# MODEL_NAME = "llama3.2:1b-instruct-q2_K"
# MODEL_NAME = "llama3.2:1b-instruct-q5_K_M"
# MODEL_NAME = "llama3.2:1b"
# MODEL_NAME = "llama3.2:1b-instruct-q5_K_M"
# MODEL_NAME = "llama3.2:3b-instruct-q4_K_M"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q3_K_S")
OLLAMA_TIMEOUT_SEC = 480  # 8 minutes for long inference

SYSTEM_PROMPT = """You are a sentiment analysis expert. These are movie review sentences. Classify sentiment for each CSV row. Output ONLY a JSON object.
Keys: "0", "1", "2", ... (one per row, in order). Values: "positive" or "negative" only.
Example for 3 rows: {"0":"positive","1":"negative","2":"positive"}
Classify objectively; do not assume negative when unclear. No other text. No neutral. No explanation."""

USER_PROMPT_TEMPLATE = """Here is the data:

{input_csv}

Respond with a JSON object. Keys must be "0" through "{max_id}" (one per row). Values: "positive" or "negative" only."""


class OllamaSentimentService:
    def __init__(self) -> None:
        pass

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        _debug("[classify] start, n_texts=%d", len(texts))
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
        max_id = len(texts) - 1

        prompt = USER_PROMPT_TEMPLATE.format(input_csv=input_csv, max_id=max_id)
        _debug("[classify] calling ollama.chat, prompt_len=%d", len(prompt))
        _debug("system prompt: %s", SYSTEM_PROMPT)
        _debug("prompt: %s", prompt)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0, "top_p": 0.2},
        )
        content = response.get("message", {}).get("content", "") or ""
        _debug("[classify] ollama.chat returned, content_len=%d", len(content))
        _debug("%s", content)
        predictions = self._parse_response(content, len(texts))
        _debug("[classify] done, n_predictions=%d", len(predictions))
        return predictions

    def _parse_response(self, content: str, expected_count: int) -> list[dict[str, Any]]:
        id_to_result: dict[int, dict[str, Any]] = {}

        content = content.strip()
        data = json.loads(content)

        if isinstance(data, dict):
            for key, val in data.items():
                try:
                    idx = int(key)
                except (ValueError, TypeError):
                    continue
                if idx < 0 or idx >= expected_count:
                    continue
                raw = str(val).strip().lower()
                label = "negative" if ("neg" in raw or raw == "negative") else "positive"
                score = -0.99 if label == "negative" else 0.99
                id_to_result[idx] = {"label": label, "score": round(score, 4)}

        predictions = []
        for i in range(expected_count):
            if i in id_to_result:
                predictions.append(id_to_result[i])
            else:
                predictions.append({"label": "positive", "score": 0.0})

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

        # Endpoint timeout: 8 minutes for long inference
        self.request.settimeout(OLLAMA_TIMEOUT_SEC)

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

        _debug("[handler] POST /v1/sentiment request_id=%s n_texts=%d", request_id, len(texts))
        t0 = time.perf_counter()
        try:
            predictions = SERVICE.classify(texts)
        except Exception as exc:
            _debug("[handler] classify raised: %s", exc)
            self._send_json(500, _json_error(f"Ollama error: {exc}"))
            return
        server_ms = (time.perf_counter() - t0) * 1000.0
        _debug("[handler] classify done in %.2f ms, sending response", server_ms)

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
