# Tesseract Hackathon 2026 - Track 05 primary entry
.PHONY: judge test verify experiment-a experiment-b audit

PYTHON ?= python3

judge:
	bash scripts/judge_demo.sh

test:
	$(PYTHON) -m pytest -q tests

verify:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	$(PYTHON) scripts/verify_e2e_gradient.py

experiment-a:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	$(PYTHON) scripts/experiment_a.py

experiment-b:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	$(PYTHON) scripts/experiment_b.py

audit:
	$(PYTHON) scripts/submission_audit.py
