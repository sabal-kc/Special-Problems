"""
Traditional sentiment service for Task 1 benchmarking.

Endpoint:
  POST /v1/sentiment

Request JSON:
  {
    "request_id": "optional-string",
    "texts": ["sentence 1", "sentence 2", ...]
  }

Response JSON:
  {
    "request_id": "...",
    "model": "vader",
    "predictions": [{"label": "positive|negative", "score": 0.1234}, ...],
    "timing": {"server_ms": 12.34}
  }
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_nltk_data() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        return
    except LookupError:
        pass

    try:
        nltk.download("vader_lexicon", quiet=True)
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception as exc:
        raise RuntimeError(
            "NLTK vader_lexicon is missing. Run this once in an environment with internet: "
            "`python3 -c \"import nltk; nltk.download('vader_lexicon')\"`"
        ) from exc


class SentimentService:
    def __init__(self) -> None:
        _ensure_nltk_data()
        self._sid = SentimentIntensityAnalyzer()

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for text in texts:
            score = self._sid.polarity_scores(text)["compound"]
            label = "positive" if score >= 0 else "negative"
            predictions.append({"label": label, "score": round(float(score), 4)})
        return predictions


SERVICE = SentimentService()


def _json_error(message: str) -> dict[str, str]:
    return {"error": message}


class SentimentRequestHandler(BaseHTTPRequestHandler):
    server_version = "TraditionalSentimentService/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep logs concise.
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": "vader"})
            return
        self._send_json(404, _json_error("Not found"))

    def do_POST(self) -> None:  # noqa: N802
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
        predictions = SERVICE.classify(texts)
        server_ms = (time.perf_counter() - t0) * 1000.0

        response = {
            "request_id": request_id,
            "model": "vader",
            "predictions": predictions,
            "timing": {"server_ms": round(server_ms, 4)},
        }
        self._send_json(200, response)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traditional sentiment HTTP service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SentimentRequestHandler)
    print(f"Traditional sentiment service listening on http://{args.host}:{args.port}")
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
