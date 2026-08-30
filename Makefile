# Tesseract Hackathon 2026 - Track 05 primary entry
.PHONY: build build-images judge test verify verify-containers experiment-a experiment-b experiment-b-v2 experiment-c audit figures renderer-figure animation landing

PYTHON ?= python3
TESSERACT ?= tesseract
FORTRAN_DIR := tesseracts/darcy-flow/fortran
FORTRAN_BIN := $(FORTRAN_DIR)/darcy

# Visual-layer artifact inputs. The judged surface defaults to the final v2 run;
# v1 remains available by overriding FIELDS and RESULTS explicitly.
FIELDS ?= figures/experiment_b_v2_fields.npz
RESULTS ?= figures/experiment_b_v2.json
EXPA ?= figures/experiment_a.json
SNAPSHOTS ?=

figures:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/make_figures.py --fields $(FIELDS) --results $(RESULTS) --expa $(EXPA)

renderer-figure:
	"$(PYTHON)" scripts/make_renderer_figure.py

animation:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	OMP_NUM_THREADS=1 \
	"$(PYTHON)" scripts/make_animation.py $(if $(SNAPSHOTS),--snapshots $(SNAPSHOTS),--demo)

landing:
	"$(PYTHON)" scripts/make_landing.py --results $(RESULTS) --expa $(EXPA)

build:
	gfortran -O2 -fno-fast-math -c $(FORTRAN_DIR)/darcy.f90 -o $(FORTRAN_DIR)/darcy.o
	gfortran -O2 -fno-fast-math $(FORTRAN_DIR)/darcy.o $(FORTRAN_DIR)/main.f90 -o $(FORTRAN_BIN)

build-images:
	for component in darcy-flow heat-transport viscosity-closure thermal-camera; do \
		"$(TESSERACT)" build "tesseracts/$$component" || exit 1; \
	done

judge: build
	PYTHON_BIN="$(PYTHON)" bash scripts/judge_demo.sh

test:
	"$(PYTHON)" -m pytest -q tests

verify:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/verify_e2e_gradient.py

verify-containers: build-images
	COUPLER_INPROCESS= DARCY_SOLVER_BIN= OMP_NUM_THREADS=1 \
	"$(PYTHON)" scripts/verify_e2e_gradient.py --nx 8 --ny 4 --n-u 24 --n-v 16 \
		--out figures/container_e2e_gradient_check.json

experiment-a:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_a.py

experiment-b:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_b.py

experiment-b-v2:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	"$(PYTHON)" scripts/experiment_b_v2.py

experiment-c:
	COUPLER_INPROCESS=1 DARCY_SOLVER_BIN="$(CURDIR)/tesseracts/darcy-flow/fortran/darcy" \
	OMP_NUM_THREADS=1 \
	"$(PYTHON)" scripts/experiment_c_renderer_necessity.py

audit:
	"$(PYTHON)" scripts/submission_audit.py
