# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Fail when the companion's judged surface drifts from committed evidence."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors: list[str] = []
    required = (
        "README.md", "LICENSE", "NOTICE.md", "CITATION.cff", "requirements.txt",
        "writeup/PROTOCOL.md", "figures/e2e_gradient_check.json",
        "tests/test_camera.py", ".github/workflows/verify.yml",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" not in line:
            errors.append(f"dependency is not exactly pinned: {line}")

    artifact = json.loads((ROOT / "figures/e2e_gradient_check.json").read_text())
    if artifact.get("verdict") != "PASS" or artifact.get("best_rel_err", 1.0) >= 1e-4:
        errors.append("stored end-to-end gradient artifact does not pass the declared gate")

    tree = ast.parse((ROOT / "tests/test_camera.py").read_text())
    tests = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    if len(tests) < 12:
        errors.append(f"camera suite has only {len(tests)} tests; expected at least 12")

    readme = (ROOT / "README.md").read_text()
    for phrase in ("Track 05 companion", "3.6e-07", "not a second form submission"):
        if phrase.lower() not in readme.lower():
            errors.append(f"README is missing required disclosure/evidence: {phrase}")

    if errors:
        print("SUBMISSION AUDIT: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("SUBMISSION AUDIT: PASS")


if __name__ == "__main__":
    main()
