"""CLI for predicting phishing URLs with the trained model.

Usage:
    python predict.py URL [URL ...]
    python predict.py --interactive
    cat urls.txt | python predict.py -
"""

from __future__ import annotations

import argparse
import json
import sys

from phishing_detector import PhishingDetector


def _format(p) -> str:
    label = "PHISHING" if p.is_phishing else "LEGITIMATE"
    return (
        f"  {p.url}\n"
        f"    label       = {label}\n"
        f"    probability = {p.probability:.4f}\n"
        f"    threshold   = {p.threshold:.4f}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urls", nargs="+",
        help="URL(s) to classify. Use '-' to read URLs from stdin (one per line).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    if args.urls == ["-"]:
        urls = [line.strip() for line in sys.stdin if line.strip()]
    else:
        urls = args.urls

    detector = PhishingDetector()
    results = detector.predict_batch(urls)

    if args.json:
        json.dump([r.as_dict() for r in results], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            sys.stdout.write(_format(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
