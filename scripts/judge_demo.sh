#!/usr/bin/env bash
# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0

set -u
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"

NAMES=()
RESULTS=()
DETAILS=()
FAILED=0
run_gate() {
  local name="$1"
  shift
  local output
  if output="$("$@" 2>&1)"; then
    NAMES+=("$name"); RESULTS+=("PASS"); DETAILS+=("$(printf '%s' "$output" | tail -n 1)")
  else
    NAMES+=("$name"); RESULTS+=("FAIL"); DETAILS+=("$(printf '%s' "$output" | tail -n 1)")
    printf '%s\n' "$output" >&2
    FAILED=1
  fi
}

run_gate "submission surface" "$PYTHON_BIN" scripts/submission_audit.py
run_gate "camera tests" "$PYTHON_BIN" -m pytest -q tests/test_camera.py
run_gate "thermography tests" "$PYTHON_BIN" -m pytest -q tests/test_thermography.py
run_gate "Python syntax" "$PYTHON_BIN" -m compileall -q coupler scripts tests tesseracts/thermal-camera

printf '\n%-22s %-6s %s\n' "GATE" "RESULT" "DETAIL"
printf '%-22s %-6s %s\n' "----------------------" "------" "------"
for ((i = 0; i < ${#NAMES[@]}; i++)); do
  printf '%-22s %-6s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${DETAILS[$i]}"
done
if ((FAILED)); then printf '\nJUDGE VERDICT: FAIL\n'; exit 1; fi
printf '\nJUDGE VERDICT: PASS\n'
