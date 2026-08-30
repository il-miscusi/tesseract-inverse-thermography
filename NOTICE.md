# NOTICE — shared code and provenance

## Relationship to the author's Track 02 repository

This repository is the author's **primary hackathon entry (Track 05 —
Differentiable graphics & rendering)**. Its coupled physics core was copied
from the same author's earlier Track 02 work,
[`il-miscusi/tesseract-coupled-adjoint`](https://github.com/il-miscusi/tesseract-coupled-adjoint)
(Multi-physics & coupled systems). That attribution stands regardless of that
repository's visibility, and we state the shared surface here so no judge has
to discover it:

**Shared (copied from the Track 02 entry, same author, Apache-2.0):**

- `tesseracts/darcy-flow/` — Fortran Darcy/Brinkman solver with a
  hand-written discrete adjoint.
- `tesseracts/heat-transport/` — JAX advection-diffusion solver with
  implicit-function-theorem differentiation. **Extended here**: `q_source`
  is now a differentiable input (the inverse problem recovers it), which the
  Track 02 version did not need.
- `tesseracts/viscosity-closure/` — PyTorch learned viscosity closure,
  including its trained weights.
- `coupler/` core (`coupled.py`, `components.py`, `session.py`, `problem.py`,
  `optimize.py`, `jax_frontend.py`, `torch_frontend.py`) — the fixed-point
  solver and matrix-free implicit-function-theorem adjoint across the three
  containers.

**New in this entry (nowhere in the Track 02 repository):**

- `tesseracts/thermal-camera/` — a physically based differentiable LWIR
  camera renderer (band-integrated Planck radiometry, per-pixel emissivity
  with reflected ambient, homographic projection with differentiable bilinear
  sampling, differentiable-width Gaussian PSF, cos^4 vignetting, gain/offset
  sensor model), packaged as a fourth Tesseract.
- `coupler/camera.py`, `coupler/thermography.py` — the pixels-to-q adjoint
  chain: camera VJP -> implicit-function-theorem solve across the coupled
  equilibrium -> heat-source contraction.
- The inverse-rendering experiments (camera self-calibration; single-image
  heat-source recovery through the live multiphysics equilibrium, with a
  one-way-physics bias comparison), their protocol, verification gates,
  tests, figures and writeup.

The Track 02 entry contains none of the rendering, inverse-rendering or
source-recovery work; this entry contains none of the topology-optimization,
Enzyme-provider or closure-calibration work. This repository stands on its
own merits as the Track 05 submission.

## Third-party components

- [Tesseract](https://github.com/pasteurlabs/tesseract-core) and
  [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax) by Pasteur
  Labs, Apache-2.0.
- JAX, PyTorch, NumPy, SciPy, Matplotlib — their own licenses; none is
  redistributed here.
- The viscosity closure was trained (in the Track 02 work) on synthetic data
  from published correlation forms (ASTM D341 viscosity-temperature
  behaviour); no proprietary data is included.
- Physical constants: CODATA 2018. No datasets, no external models, no
  generated assets from third parties.
