"""Load trained artifacts and run phishing predictions on URLs."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras

from .features import extract_features

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
MODEL_PATH = _REPO / "my_model.keras"
SCALER_PATH = _REPO / "scaler.pkl"
METRICS_PATH = _REPO / "metrics.json"
FEATURE_NAMES_PATH = _REPO / "feature_names.json"
SHA256SUMS_PATH = _REPO / "SHA256SUMS"


def _load_expected_sha256(name: str, sums_path: Path = SHA256SUMS_PATH) -> str | None:
    """Return the expected SHA256 for ``name`` from a SHA256SUMS file."""
    if not sums_path.exists():
        return None
    for line in sums_path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, fname = parts
        if fname.strip() == name:
            return digest.strip()
    return None


def _verify_sha256(path: Path, expected: str | None) -> None:
    """Raise RuntimeError if ``path`` does not match ``expected`` SHA256."""
    if expected is None:
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"integrity check failed for {path.name}: "
            f"expected {expected}, got {actual}"
        )


@dataclass
class Prediction:
    url: str
    is_phishing: bool
    probability: float
    threshold: float

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "is_phishing": self.is_phishing,
            "label": "phishing" if self.is_phishing else "legitimate",
            "probability": round(self.probability, 4),
            "threshold": round(self.threshold, 4),
        }


class PhishingDetector:
    """Loaded model + scaler + threshold, ready for inference."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        metrics_path: Path = METRICS_PATH,
        verify_artifacts: bool = True,
    ) -> None:
        if verify_artifacts:
            _verify_sha256(scaler_path, _load_expected_sha256(scaler_path.name))
            _verify_sha256(model_path, _load_expected_sha256(model_path.name))
        self.model = keras.models.load_model(model_path, compile=False)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        self.threshold = float(metrics.get("threshold", 0.5))
        self.metrics = metrics

    def predict(self, url: str) -> Prediction:
        feats = extract_features(url).reshape(1, -1)
        feats_s = self.scaler.transform(feats).astype(np.float32)
        prob = float(self.model.predict(feats_s, verbose=0).ravel()[0])
        is_phish = prob >= self.threshold
        return Prediction(
            url=url,
            is_phishing=is_phish,
            probability=prob,
            threshold=self.threshold,
        )

    def predict_batch(self, urls: list[str]) -> list[Prediction]:
        if not urls:
            return []
        feats = np.vstack([extract_features(u) for u in urls])
        feats_s = self.scaler.transform(feats).astype(np.float32)
        probs = self.model.predict(feats_s, verbose=0).ravel()
        return [
            Prediction(
                url=u,
                is_phishing=float(p) >= self.threshold,
                probability=float(p),
                threshold=self.threshold,
            )
            for u, p in zip(urls, probs)
        ]
