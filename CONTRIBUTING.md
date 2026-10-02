# Contributing

## Reproducible environment

Use the supported Python version declared by the repository and install dependencies from `requirements.txt` in an isolated environment. Keep training datasets, downloaded samples, and generated model artifacts out of commits unless their provenance and redistribution rights are documented.

## Validation

Run `pytest` before submitting a change. Keep feature extraction and model-evaluation changes covered by tests. CI must remain offline with respect to phishing targets: tests must use local fixtures and must not fetch URLs.

## Data and privacy

Use synthetic or legally redistributable samples. Remove credentials, tracking identifiers, personal data, and live phishing infrastructure from issues, pull requests, logs, and notebooks.
