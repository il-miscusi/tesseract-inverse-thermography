# Tesseract Hackathon 2026 - Track 05 primary entry
.PHONY: judge test verify experiment-a experiment-b experiment-b-v2 audit figures animation landing

PYTHON ?= python3

# Visual-layer artifact inputs. The judged surface defaults to the final v2 run;
# v1 remains available by overriding FIELDS and RESULTS explicitly.
FIELDS ?= figures/experiment_b_v2_fields.npz
RESULTS ?= figures/experiment_b_v2.json
EXPA ?= figures/experiment_a.json
SNAPSHOTS ?=

figures:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/make_figures.py --fields $(FIELDS) --results $(RESULTS) --expa $(EXPA)

animation:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	OMP_NUM_THREADS=1 \
	"$(PYTHON)" scripts/make_animation.py $(if $(SNAPSHOTS),--snapshots $(SNAPSHOTS),--demo)

landing:
	"$(PYTHON)" scripts/make_landing.py --results $(RESULTS) --expa $(EXPA)

judge:
	PYTHON_BIN="$(PYTHON)" bash scripts/judge_demo.sh

test:
	"$(PYTHON)" -m pytest -q tests

verify:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/verify_e2e_gradient.py

experiment-a:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_a.py

experiment-b:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_b.py

experiment-b-v2:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_b_v2.py

audit:
	"$(PYTHON)" scripts/submission_audit.py
