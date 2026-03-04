#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export PYDANTIC_DISABLE_PLUGINS="${PYDANTIC_DISABLE_PLUGINS:-__all__}"
export PYTHONPATH="$ROOT_DIR"

# Detect venv pytest/python
PYTEST_BIN="pytest"
PYTHON_BIN="python"
if [[ -x "$ROOT_DIR/.venv/bin/pytest" ]]; then
  PYTEST_BIN="$ROOT_DIR/.venv/bin/pytest"
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/pytest" ]]; then
  PYTEST_BIN="$ROOT_DIR/venv/bin/pytest"
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  MAIN_WORKTREE="$(git worktree list --porcelain | awk '/^worktree / {print $2; exit}')"
  if [[ -n "$MAIN_WORKTREE" && -x "$MAIN_WORKTREE/.venv/bin/pytest" ]]; then
    PYTEST_BIN="$MAIN_WORKTREE/.venv/bin/pytest"
    PYTHON_BIN="$MAIN_WORKTREE/.venv/bin/python"
  elif [[ -n "$MAIN_WORKTREE" && -x "$MAIN_WORKTREE/venv/bin/pytest" ]]; then
    PYTEST_BIN="$MAIN_WORKTREE/venv/bin/pytest"
    PYTHON_BIN="$MAIN_WORKTREE/venv/bin/python"
  fi
fi

# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------
ALL_MODULES=(voice orchestrator handlers tools evals)

module_test_paths() {
  case "$1" in
    voice)        echo "scripts/test_voice_agent_suite.sh / scripts/test_voice_agent_coverage.sh" ;;
    orchestrator) echo "tests/core/orchestrator/" ;;
    handlers)     echo "tests/core/handler/" ;;
    tools)        echo "tests/core/tools/" ;;
    evals)        echo "tests/eval/ super_services/evals/" ;;
    *) echo ""; return 1 ;;
  esac
}

module_coverage_targets() {
  case "$1" in
    voice)        echo "super/core/voice/ super/core/memory/index/chroma.py super/core/memory/search/reranker.py" ;;
    orchestrator) echo "super/core/orchestrator/" ;;
    handlers)     echo "super/core/handler/" ;;
    tools)        echo "super/core/tools/" ;;
    evals)        echo "" ;;  # evals measure model quality, not code coverage
    *) echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

count_test_files() {
  local module="$1"
  if [[ "$module" == "voice" ]]; then
    # Keep count in sync with the source-of-truth voice suite file list.
    awk '
      /^VOICE_AGENT_TESTS=\(/ {in_list=1; next}
      in_list && /^\)/ {exit}
      in_list && /"tests\/.*\.py"/ {count++}
      END {print count + 0}
    ' "$ROOT_DIR/scripts/test_voice_agent_suite.sh"
    return 0
  fi

  local paths
  paths="$(module_test_paths "$module")"
  local count=0
  for p in $paths; do
    if [[ -d "$p" ]]; then
      count=$((count + $(find "$p" -name 'test_*.py' | wc -l | tr -d ' ')))
    elif [[ -f "$p" ]]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

cmd_list() {
  echo "Available test modules:"
  for module in "${ALL_MODULES[@]}"; do
    local paths count
    paths="$(module_test_paths "$module")"
    count="$(count_test_files "$module")"
    printf "  %-16s %3s tests   %s\n" "$module" "$count" "$paths"
  done
}

run_module() {
  local module="$1"
  shift

  if [[ "$module" == "voice" ]]; then
    echo ""
    echo "=== Running module: voice ==="
    if [[ "${COV:-0}" == "1" ]]; then
      echo "[runner] Delegating to scripts/test_voice_agent_coverage.sh"
      ./scripts/test_voice_agent_coverage.sh "$@"
    else
      echo "[runner] Delegating to scripts/test_voice_agent_suite.sh"
      ./scripts/test_voice_agent_suite.sh "$@"
    fi
    return
  fi

  local paths
  paths="$(module_test_paths "$module")" || {
    echo "Unknown module: $module" >&2
    echo "Run '$0 list' to see available modules." >&2
    return 1
  }

  echo ""
  echo "=== Running module: $module ==="
  echo "[runner] Pytest: $PYTEST_BIN"
  echo "[runner] Paths:  $paths"

  # Split paths into array
  local -a path_args
  read -ra path_args <<< "$paths"
  local -a extra_args=()

  if [[ "$module" == "evals" ]] && ! "$PYTHON_BIN" -c "import mongomantic.core" >/dev/null 2>&1; then
    echo "[filter] mongomantic.core not installed; skipping super_services/evals/ tests" >&2
    local -a filtered_paths=()
    for p in "${path_args[@]}"; do
      [[ "$p" == "super_services/evals/" ]] && continue
      filtered_paths+=("$p")
    done
    path_args=("${filtered_paths[@]}")
  fi
  if [[ "$module" == "evals" ]] && ! "$PYTHON_BIN" -c "import ragas" >/dev/null 2>&1; then
    echo "[filter] ragas not installed; skipping tests/eval/test_ragas_eval.py" >&2
    extra_args+=("--ignore=tests/eval/test_ragas_eval.py")
  fi

  if [[ "${COV:-0}" == "1" ]]; then
    local cov_targets
    cov_targets="$(module_coverage_targets "$module")"
    if [[ -z "$cov_targets" ]]; then
      echo "[runner] No coverage targets for $module; running without coverage."
      "$PYTEST_BIN" -ra -p pytest_asyncio.plugin "${extra_args[@]}" "${path_args[@]}" "$@"
    elif [[ "${ENABLE_PYTEST_COV:-0}" == "1" ]] && "$PYTHON_BIN" -c "import pytest_cov" >/dev/null 2>&1; then
      # pytest-cov mode (opt-in)
      echo "[runner] Coverage via pytest-cov (fail-under: ${COV_FAIL_UNDER:-20}%)"
      local -a cov_flags=()
      for target in $cov_targets; do
        # Convert file paths to module-style for --cov (strip .py, replace / with .)
        local mod="${target%.py}"
        mod="${mod%/}"
        mod="${mod//\//.}"
        cov_flags+=("--cov=$mod")
      done
      "$PYTEST_BIN" -ra -p pytest_asyncio.plugin -p pytest_cov \
        "${cov_flags[@]}" --cov-report=term-missing \
        --cov-fail-under="${COV_FAIL_UNDER:-20}" \
        "${extra_args[@]}" "${path_args[@]}" "$@"
    else
      # coverage.py mode (default)
      echo "[runner] Coverage via coverage.py (fail-under: ${COV_FAIL_UNDER:-20}%)"
      local -a source_flags=()
      for target in $cov_targets; do
        # Convert paths to module-style for coverage --source (strip .py, trim /).
        local mod="${target%.py}"
        mod="${mod%/}"
        mod="${mod//\//.}"
        source_flags+=("--source=$mod")
      done
      "$PYTHON_BIN" -m coverage erase
      "$PYTHON_BIN" -m coverage run \
        "${source_flags[@]}" \
        -m pytest -ra -p pytest_asyncio.plugin "${extra_args[@]}" "${path_args[@]}" "$@"
      "$PYTHON_BIN" -m coverage report -m \
        --fail-under="${COV_FAIL_UNDER:-20}"
    fi
  else
    "$PYTEST_BIN" -ra -p pytest_asyncio.plugin "${extra_args[@]}" "${path_args[@]}" "$@"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
  # Run all modules with coverage enabled by default
  export COV="${COV:-1}"
  for module in "${ALL_MODULES[@]}"; do
    run_module "$module"
  done
  exit 0
fi

if [[ "$1" == "list" ]]; then
  cmd_list
  exit 0
fi

# Separate module names from pytest pass-through flags
MODULES=()
PYTEST_EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == -* ]]; then
    PYTEST_EXTRA+=("$arg")
  elif [[ ${#PYTEST_EXTRA[@]} -gt 0 ]]; then
    # After first flag, treat everything as pytest args (e.g. -k "name")
    PYTEST_EXTRA+=("$arg")
  else
    MODULES+=("$arg")
  fi
done

if [[ ${#MODULES[@]} -eq 0 ]]; then
  echo "Error: no module specified. Run '$0 list' to see available modules." >&2
  exit 1
fi

for module in "${MODULES[@]}"; do
  run_module "$module" "${PYTEST_EXTRA[@]}"
done
