# Differentiable thermography through a coupled equilibrium

**Tesseract Hackathon 2026 - Track 05 companion showcase**

A physically based thermal-camera Tesseract turns temperature fields into LWIR
sensor counts through Planck-band emission, emissivity, camera pose, optical
blur, vignetting, gain, and offset. Its pixel VJP is composed with a coupled
Fortran/JAX/PyTorch cold-plate equilibrium, so a loss on one thermal image can
recover the hidden volumetric heat-source map that produced it.

This repository is a companion to
[`tesseract-coupled-adjoint`](https://github.com/il-miscusi/tesseract-coupled-adjoint),
the primary Track 02 entry. The event terms allow one submission per person or
team, so this is **not a second form submission unless the organisers explicitly
approve it**.

## The differentiable chain

```text
counts = Camera(T*, emissivity, PSF, gain, offset)       JAX Tesseract
T* solves T = Heat(gamma, Flow(gamma, Viscosity(T)); q)
                   JAX          Fortran       PyTorch

pixel loss -> camera VJP -> coupled implicit adjoint -> dJ/dq
```

The camera does not treat temperature as brightness. It integrates Planck
radiance over the 8-14 micrometre LWIR band, adds reflected ambient radiance
under a grey opaque surface model, projects through a homography with bilinear
sampling, applies a differentiable Gaussian PSF and cos-to-the-fourth
vignetting, then converts radiance to digital counts. Measurement noise is
added outside the differentiable renderer.

## Correctness gate

`scripts/verify_e2e_gradient.py` checks the complete pixels-to-source gradient
against central finite differences of the camera and the full coupled
equilibrium. On the declared 16x8 physics grid and 48x32 sensor:

| quantity | result |
|---|---:|
| best directional relative error | **3.6e-07** |
| required threshold | 1e-04 |
| verdict | **PASS** |

The fast test suite adds 18 independent gates for Planck monotonicity,
emissivity/reflection limits, homography, sampling, PSF normalisation,
vignetting, rendering behavior, a temperature VJP finite-difference check,
source metrics, regularisation, and optimisation utilities.

## Pre-declared experiments

The complete protocol was committed before the reported runs
([`writeup/PROTOCOL.md`](writeup/PROTOCOL.md)).

**A - camera self-calibration.** From one image and a known temperature field,
recover spatial emissivity, PSF width, gain, and offset across five fixed noise
levels. Gain-emissivity identifiability is explicitly measured rather than
hidden by reporting only image loss.

**B - source recovery through the physics.** From one noisy thermal image,
recover a 32x16 non-negative two-hotspot heat-source map. The headline arm uses
the full coupled adjoint. The comparison arm uses frozen-viscosity physics
against the same coupled measurement, quantifying bias from ignoring feedback.
Both arms keep the same data, prior, optimiser, and iteration budget; the sign
of the comparison is reported whatever it is.

The registered success criterion for the coupled arm is relative source error
below 0.5 and centroid shift below 1.5 cells. Experiment outputs are JSON and
NPZ artifacts; smoke-run files are excluded from the judged surface.

## Reported results

The complete, seed- and configuration-stamped outputs are checked in as
[`figures/experiment_a.json`](figures/experiment_a.json) and
[`figures/experiment_b.json`](figures/experiment_b.json), with field arrays in
the matching NPZ files. Experiment A recovered gain to 0.20--0.21% relative
error across the five noise levels, while separate PSF and offset estimates
remained weakly identified from one image (PSF absolute error 0.91 px; offset
relative error 18.98%). This is the pre-declared gain/emissivity ambiguity, not
a hidden success claim.

Experiment B did not meet its pre-declared coupled target: coupled relative
source L2 error was 0.9706 with a 6.85-cell centroid shift. It nevertheless
recovered total source power to a 1.041 ratio, versus 1.364 for the identical
one-way arm; one-way relative L2 was 0.9814. These are measurements of the
inverse problem's difficulty and of the feedback-model bias, reported without
cherry-picking.

## Reproduce

Requires Python 3.10 or newer and `gfortran`; Docker is optional because every
component can run from its `tesseract_api.py` in-process.

```bash
pip install -r requirements.txt
make judge       # 18 fast tests + stored end-to-end artifact audit
make verify      # rerun pixels-to-q finite differences
make experiment-a
make experiment-b
```

All host and component dependencies are exactly pinned. Apache-2.0; the copied
Track 02 physics core retains its original notice in [`NOTICE.md`](NOTICE.md).
