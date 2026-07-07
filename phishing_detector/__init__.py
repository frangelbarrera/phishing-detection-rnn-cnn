"""Phishing URL detection package."""

from .features import FEATURE_NAMES, extract_features, extract_features_batch
from .detector import PhishingDetector, Prediction

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_features_batch",
    "PhishingDetector",
    "Prediction",
]
__version__ = "1.0.0"
