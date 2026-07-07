"""Tests for the feature extractor and the trained detector.

Run with: python -m pytest tests/ -v
Or simply: python tests/test_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishing_detector import (
    FEATURE_NAMES,
    PhishingDetector,
    extract_features,
)


def test_feature_count() -> None:
    assert len(FEATURE_NAMES) == 49


def test_feature_vector_shape_and_dtype() -> None:
    v = extract_features("https://www.google.com")
    assert v.shape == (49,)
    assert v.dtype == np.float64
    assert not np.isnan(v).any()
    assert not np.isinf(v).any()


def test_feature_order_is_canonical() -> None:
    # length_url must always be the first feature
    v = extract_features("https://www.example.com")
    assert v[0] == len("https://www.example.com")
    # suspecious_tld must be the last feature
    assert v[-1] in (0.0, 1.0)


def test_obvious_phishing_url_scores_high() -> None:
    detector = PhishingDetector()
    url = "http://secure-account-verify-login.tk/login.html"
    p = detector.predict(url)
    assert p.is_phishing, f"expected phishing, got prob={p.probability}"
    assert p.probability > 0.9


def test_obvious_legitimate_url_scores_low() -> None:
    detector = PhishingDetector()
    url = "https://www.google.com"
    p = detector.predict(url)
    assert not p.is_phishing, f"expected legitimate, got prob={p.probability}"
    assert p.probability < 0.2


def test_batch_matches_singletons() -> None:
    detector = PhishingDetector()
    urls = [
        "https://www.google.com",
        "http://secure-account-verify-login.tk/login.html",
        "https://github.com/torvalds/linux",
    ]
    batch = detector.predict_batch(urls)
    singles = [detector.predict(u) for u in urls]
    for b, s in zip(batch, singles):
        assert abs(b.probability - s.probability) < 1e-5


def test_handles_url_without_scheme() -> None:
    v = extract_features("www.example.com/path")
    assert v.shape == (49,)
    assert not np.isnan(v).any()


def test_handles_empty_url() -> None:
    v = extract_features("")
    assert v.shape == (49,)
    assert not np.isnan(v).any()


def test_threshold_is_sane() -> None:
    detector = PhishingDetector()
    assert 0.3 < detector.threshold < 0.7


if __name__ == "__main__":
    # Allow running without pytest.
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
