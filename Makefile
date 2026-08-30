# Tesseract Hackathon 2026 - differentiable thermography companion
.PHONY: judge test verify experiment-a experiment-b audit figures animation landing

PYTHON ?= python3

# Visual-layer artifact inputs. Point these at the v2 artifacts when they land:
#   make figures landing FIELDS=figures/experiment_b_v2.npz RESULTS=figures/experiment_b_v2.json
FIELDS ?= figures/experiment_b_fields.npz
RESULTS ?= figures/experiment_b.json
EXPA ?= figures/experiment_a.json
SNAPSHOTS ?=

figures:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	$(PYTHON) scripts/make_figures.py --fields $(FIELDS) --results $(RESULTS) --expa $(EXPA)

animation:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	OMP_NUM_THREADS=1 \
	$(PYTHON) scripts/make_animation.py $(if $(SNAPSHOTS),--snapshots $(SNAPSHOTS),--demo)

landing:
	$(PYTHON) scripts/make_landing.py --results $(RESULTS) --expa $(EXPA)

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
