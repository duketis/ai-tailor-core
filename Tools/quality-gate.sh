#!/usr/bin/env bash
#
# Local quality gate. Mirrors CI: ruff check, ruff format --check, mypy, pytest.
#
# Sets up a scratch venv at /tmp/ai-tailor-core-tools on first run and reuses
# it thereafter. Re-run with FORCE_REINSTALL=1 to rebuild from scratch.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${TOOLS_DIR:-/tmp/ai-tailor-core-tools}"

cd "${REPO_ROOT}"

if [[ "${FORCE_REINSTALL:-0}" == "1" ]] || [[ ! -x "${TOOLS_DIR}/bin/ruff" ]]; then
  echo "[quality-gate] (re)building venv at ${TOOLS_DIR}"
  rm -rf "${TOOLS_DIR}"
  python3 -m venv "${TOOLS_DIR}"
  "${TOOLS_DIR}/bin/pip" install --quiet --upgrade pip
  "${TOOLS_DIR}/bin/pip" install --quiet -e ".[dev]"
fi

echo "[quality-gate] ruff check"
"${TOOLS_DIR}/bin/ruff" check .

echo "[quality-gate] ruff format --check"
"${TOOLS_DIR}/bin/ruff" format --check .

echo "[quality-gate] mypy"
"${TOOLS_DIR}/bin/mypy"

echo "[quality-gate] pytest"
"${TOOLS_DIR}/bin/pytest" -q

echo "[quality-gate] OK"
