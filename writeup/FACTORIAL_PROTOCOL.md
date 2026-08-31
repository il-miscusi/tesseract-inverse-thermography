# Experiment E protocol — forward fidelity versus gradient fidelity

This experiment addresses a limitation of Experiment B: changing from coupled
to frozen-viscosity physics changed both the forward model and its derivative.
Experiment E adds the missing factorial arm before its result is generated.

## Fixed arms

All arms use the Experiment B v2 measurement, grid, source, camera, noise draw,
flat initialization, TV prior, softplus parameterization, L-BFGS-B settings,
and 250-iteration / 750-evaluation cap.

1. `coupled_exact`: coupled forward equilibrium and full implicit adjoint;
   existing immutable Experiment B v2 artifact.
2. `coupled_truncated`: the same coupled forward equilibrium, but the reverse
   solve sets lambda=dJ/dT and drops `(dG/dT)^T` feedback. This isolates
   gradient fidelity while leaving every forward value unchanged.
3. `frozen_matching`: viscosity-frozen forward model and its matching direct
   gradient; existing immutable Experiment B v2 artifact.

The new runner executes only arm 2 and reads arms 1 and 3 from
`figures/experiment_b_v2.json`; it refuses a changed problem configuration.

## Fixed reporting

Report final and best evaluated coupled-objective data loss, source relative
L2, centroid error, total-power ratio, optimizer status, evaluations, and
gradient norm. At the shared initialization, report the cosine and relative L2
error between the truncated and exact gradients of the same coupled objective.
No arm is called converged unless SciPy status is zero.

The experiment is a mechanism ablation, not a generalization result. A worse
truncated arm supports the claim that the implicit adjoint improves optimization
of the coupled objective on this fixed problem. It does not establish that
truncation fails universally. Any ordering is reported.
