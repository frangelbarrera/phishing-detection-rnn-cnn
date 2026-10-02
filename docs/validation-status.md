# Validation status

**Last checked:** 2026-10-02
**Maintainer:** Frangel Raúl Crespo Barrera
**Scope:** repository tests and static checks; this page does not assess model quality or live URL handling.

| Control | Current state | Evidence | Verification |
|---|---|---|---|
| Unit tests | Passing locally: 9 passed | `tests/test_features.py` | `pytest -q` |
| Python compilation | Passing locally | Python sources and `web/` | `python -m compileall -q .` |
| Ruff | Advisory baseline: 21 findings, including imports, unused imports, exception handling, and unused variables | Ruff report from the current tree | `ruff check .` |
| Dependency audit | Must be reviewed from the CI artifact; no claim of a clean result is made here | `requirements.txt` | `pip-audit -r requirements.txt` |

The CI workflow publishes Ruff and dependency-audit reports without treating the existing Ruff findings as a passing quality claim. Converting the advisory baseline into a blocking lint gate is a separate code-quality change and is intentionally outside this documentation correction.
