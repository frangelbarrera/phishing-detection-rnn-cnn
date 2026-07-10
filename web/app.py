"""Minimal Flask web UI for classifying URLs with the trained model.

Run:
    python -m web.app --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from flask import Flask, jsonify, render_template, request  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishing_detector import PhishingDetector  # noqa: E402

MAX_URL_LENGTH = 2048

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
_detector: PhishingDetector | None = None


def get_detector() -> PhishingDetector:
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector


@app.route("/")
def index():
    return render_template("index.html", metrics=get_detector().metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    raw = data.get("url")
    if not isinstance(raw, str):
        return jsonify({"error": "'url' must be a string"}), 400
    url = raw.strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    if len(url) > MAX_URL_LENGTH:
        return jsonify({"error": f"URL exceeds maximum length of {MAX_URL_LENGTH} characters"}), 400
    try:
        result = get_detector().predict(url)
        return jsonify(result.as_dict())
    except Exception as e:  # pragma: no cover
        app.logger.exception("prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # Warm up the model so the first request isn't slow.
    get_detector()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
