"""Feature extraction for URL phishing detection.

Computes the same 49 lexical/structural features used at training time
(see train.py). All features are computed offline from the URL string;
no network access is required.
"""

from __future__ import annotations

import itertools
import re
from urllib.parse import urlparse

import numpy as np

# Features that are dropped from the original 55-column dataset because
# they cannot be computed reliably offline (require brand lists, redirect
# following, or have zero variance in the training data).
_DROPPED_FEATURES = {
    "nb_or",                      # zero variance in training data
    "random_domain",              # requires entropy model not available offline
    "domain_in_brand",            # requires curated brand list
    "brand_in_subdomain",         # requires curated brand list
    "brand_in_path",              # requires curated brand list
    "nb_external_redirection",    # requires following redirects (online)
}

# Canonical feature order. MUST match the column order produced by
# train.py::build_feature_matrix.
FEATURE_NAMES = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",
    "prefix_suffix",
    "shortening_service",
    "path_extension",
    "nb_redirection",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
    "suspecious_tld",
]

assert len(FEATURE_NAMES) == 49

_TLDS = (
    ".com", ".org", ".net", ".edu", ".gov", ".mil", ".int",
    ".biz", ".info", ".mobi", ".name", ".ly",
)
_SUSPICIOUS_TLDS = (".tk", ".xyz", ".top", ".ml", ".ga", ".cf", ".gq")
_SHORTENERS = (
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "is.gd",
    "cli.gs", "on.ly", "short.cm", "tiny.cc", "shorte.st",
    "x.co", "prettylinkpro.com", "viralurl.com",
    "qr.net", "lurl.no", "tweez.me", "v.gd", "tr.im",
    "link.zip.net",
)
_PATH_EXTENSIONS = (
    ".php", ".html", ".htm", ".asp", ".aspx",
    ".jsp", ".js", ".css", ".py",
)
_PHISHING_WORDS = (
    "secure", "account", "verify", "login", "update",
    "signin", "banking", "confirm",
)
_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _min_len(words: list[str]) -> int:
    return min((len(w) for w in words), default=0)


def _max_len(words: list[str]) -> int:
    return max((len(w) for w in words), default=0)


def _avg_len(words: list[str]) -> float:
    return _safe_div(sum(len(w) for w in words), len(words)) if words else 0.0


def extract_features(url: str) -> np.ndarray:
    """Return a (49,) float64 vector in canonical FEATURE_NAMES order."""
    f: dict[str, float] = {}

    f["length_url"] = len(url)

    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = parsed.netloc
    path = parsed.path

    f["length_hostname"] = len(hostname)

    host_no_port = hostname.split(":")[0]
    f["ip"] = 1.0 if _IPV4_RE.match(host_no_port) else 0.0

    f["nb_dots"] = url.count(".")
    f["nb_hyphens"] = url.count("-")
    f["nb_at"] = url.count("@")
    f["nb_qm"] = url.count("?")
    f["nb_and"] = url.count("&")
    f["nb_eq"] = url.count("=")
    f["nb_underscore"] = url.count("_")
    f["nb_tilde"] = url.count("~")
    f["nb_percent"] = url.count("%")
    f["nb_slash"] = url.count("/")
    f["nb_star"] = url.count("*")
    f["nb_colon"] = url.count(":")
    f["nb_comma"] = url.count(",")
    f["nb_semicolumn"] = url.count(";")
    f["nb_dollar"] = url.count("$")
    f["nb_space"] = url.count(" ")

    host_lower = hostname.lower()
    path_lower = path.lower()
    url_lower = url.lower()

    f["nb_www"] = 1.0 if "www" in host_lower else 0.0
    f["nb_com"] = 1.0 if "com" in host_lower else 0.0
    f["nb_dslash"] = url.count("//")
    f["http_in_path"] = 1.0 if "http" in path_lower else 0.0
    f["https_token"] = 1.0 if url.startswith("https://") else 0.0

    f["ratio_digits_url"] = _safe_div(
        sum(c.isdigit() for c in url), len(url)
    )
    f["ratio_digits_host"] = _safe_div(
        sum(c.isdigit() for c in hostname), len(hostname)
    )

    f["punycode"] = 1.0 if "xn--" in host_lower else 0.0
    f["port"] = 1.0 if (
        ":" in hostname
        and any(c.isdigit() for c in hostname.split(":", 1)[1])
    ) else 0.0

    f["tld_in_path"] = 1.0 if any(t in path_lower for t in _TLDS) else 0.0
    f["tld_in_subdomain"] = 1.0 if (
        hostname.count(".") > 1
        and any(t in host_lower.split(".")[0] for t in _TLDS)
    ) else 0.0
    f["abnormal_subdomain"] = 1.0 if hostname.count(".") > 2 else 0.0
    f["nb_subdomains"] = hostname.count(".")
    f["prefix_suffix"] = 1.0 if "-" in hostname else 0.0

    f["shortening_service"] = 1.0 if any(
        s in host_lower for s in _SHORTENERS
    ) else 0.0
    f["path_extension"] = 1.0 if any(
        e in path_lower for e in _PATH_EXTENSIONS
    ) else 0.0

    http_count = url.count("http")
    f["nb_redirection"] = float(http_count - 1) if http_count > 1 else 0.0

    raw_words = _WORD_RE.findall(url)
    host_words = _WORD_RE.findall(hostname)
    path_words = _WORD_RE.findall(path) if path else []

    f["length_words_raw"] = len(raw_words)
    f["char_repeat"] = max(
        (len(list(g)) for _, g in itertools.groupby(url)),
        default=0,
    )
    f["shortest_words_raw"] = _min_len(raw_words)
    f["shortest_word_host"] = _min_len(host_words)
    f["shortest_word_path"] = _min_len(path_words)
    f["longest_words_raw"] = _max_len(raw_words)
    f["longest_word_host"] = _max_len(host_words)
    f["longest_word_path"] = _max_len(path_words)
    f["avg_words_raw"] = _avg_len(raw_words)
    f["avg_word_host"] = _avg_len(host_words)
    f["avg_word_path"] = _avg_len(path_words)

    f["phish_hints"] = 1.0 if any(
        w in url_lower for w in _PHISHING_WORDS
    ) else 0.0
    f["suspecious_tld"] = 1.0 if host_lower.endswith(_SUSPICIOUS_TLDS) else 0.0

    return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_features_batch(urls: list[str]) -> np.ndarray:
    """Stack extract_features over a list of URLs -> (N, 49) float64."""
    return np.vstack([extract_features(u) for u in urls])
