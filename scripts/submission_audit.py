# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Fail when the judged surface drifts from committed evidence.

Asserts the Track 05 submission surface: the track is named where judges
read, the license is Apache-2.0, the numbers trace to figure artifacts, no
result placeholders remain, and no machine-local paths leak into docs.
While the <!-- RESULT: ... --> placeholders for experiment B v2 exist, this
audit FAILS by design — the repo is not submittable until the numbers land.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRACK_PHRASE = "Track 05: Differentiable graphics & rendering"
PLACEHOLDER_RE = re.compile(
    r"<!--\s*RESULT:|\[\[[A-Z][A-Z0-9_]*(?:[^\]]*)\]\]", re.IGNORECASE
)
ABS_PATH_RE = re.compile(r"/Users/[A-Za-z0-9_.-]+/")
BANNED_PHRASES = ("companion", "not a second form submission")
FALSE_CLAIMS = (
    "only the gradient changed",
    "only the physics in the gradient differs",
    "converged data loss",
    "a converged fit",
    "model-mismatch floor",
)


def _doc_files() -> list[Path]:
    docs = [ROOT / "README.md", ROOT / "NOTICE.md", ROOT / "CITATION.cff"]
    docs += sorted((ROOT / "writeup").glob("*.md"))
    docs += sorted(ROOT.rglob("*.ipynb"))
    return [p for p in docs if p.is_file()]


def main() -> None:
    errors: list[str] = []
    required = (
        "README.md", "LICENSE", "NOTICE.md", "CITATION.cff", "requirements.txt",
        "writeup/PROTOCOL.md", "writeup/WRITEUP.md",
        "figures/e2e_gradient_check.json",
        "figures/experiment_a.json", "figures/experiment_b.json",
        "tests/test_camera.py", ".github/workflows/verify.yml",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    # License must be Apache-2.0.
    license_path = ROOT / "LICENSE"
    if license_path.is_file() and "Apache License" not in license_path.read_text():
        errors.append("LICENSE is not the Apache License")

    # Exactly pinned host dependencies.
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" not in line:
            errors.append(f"dependency is not exactly pinned: {line}")

    # Stored end-to-end gradient artifact must pass its declared gate.
    artifact = json.loads((ROOT / "figures/e2e_gradient_check.json").read_text())
    if artifact.get("verdict") != "PASS" or artifact.get("best_rel_err", 1.0) >= 1e-4:
        errors.append("stored end-to-end gradient artifact does not pass the declared gate")

    # Experiment metadata is the authority for optimizer and stopping claims.
    b2_path = ROOT / "figures/experiment_b_v2.json"
    if not b2_path.is_file():
        errors.append("missing required file: figures/experiment_b_v2.json")
        b2 = {}
    else:
        b2 = json.loads(b2_path.read_text())
        if b2.get("optimizer", {}).get("method") != "L-BFGS-B":
            errors.append("experiment B v2 optimizer is not L-BFGS-B")
        for arm in ("coupled", "one_way"):
            if arm not in b2.get("optimizer", {}):
                errors.append(f"experiment B v2 missing optimizer metadata for {arm}")

    # Camera suite depth.
    tree = ast.parse((ROOT / "tests/test_camera.py").read_text())
    tests = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    if len(tests) < 12:
        errors.append(f"camera suite has only {len(tests)} tests; expected at least 12")

    # README: Track 05 framing plus the standing gradient evidence.
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text() if readme_path.is_file() else ""
    for phrase in (TRACK_PHRASE, "3.6e-07"):
        if phrase.lower() not in readme.lower():
            errors.append(f"README is missing required framing/evidence: {phrase}")

    # Writeup must name the track explicitly.
    writeup_path = ROOT / "writeup/WRITEUP.md"
    writeup = writeup_path.read_text() if writeup_path.is_file() else ""
    if TRACK_PHRASE.lower() not in writeup.lower():
        errors.append(f"writeup does not name the track: {TRACK_PHRASE}")

    # Judged docs: no companion framing, no result placeholders, no local
    # paths, and every referenced figures/ artifact must exist.
    for doc in _doc_files():
        rel = doc.relative_to(ROOT)
        text = doc.read_text(errors="replace")
        for banned in BANNED_PHRASES:
            if banned.lower() in text.lower():
                errors.append(f"{rel}: contains retired framing: {banned!r}")
        for false_claim in FALSE_CLAIMS:
            if false_claim.lower() in text.lower():
                errors.append(f"{rel}: contains claim contradicted by artifacts: {false_claim!r}")
        n_placeholders = len(PLACEHOLDER_RE.findall(text))
        if n_placeholders:
            errors.append(f"{rel}: {n_placeholders} unresolved result placeholder(s)")
        for match in ABS_PATH_RE.findall(text):
            errors.append(f"{rel}: absolute local path leaked: {match}")
        for fig_ref in re.findall(r"figures/[A-Za-z0-9_./-]+", text):
            ref = ROOT / fig_ref
            if not (ref.is_file() or ref.is_dir()):
                errors.append(f"{rel}: references missing artifact: {fig_ref}")

    figure_source = (ROOT / "scripts/make_figures.py").read_text()
    for stale_label in ("Adam iteration", "{gap:.0f}× apart"):
        if stale_label in figure_source:
            errors.append(f"make_figures.py contains stale label: {stale_label!r}")

    if errors:
        print("SUBMISSION AUDIT: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("SUBMISSION AUDIT: PASS")


if __name__ == "__main__":
    main()
