# Experiment B v2 sweep record (16x8 grid, 48x32 sensor, coupled physics)

Diagnostic runs behind writeup/PROTOCOL.md Amendment v2, produced by
scripts/sweep_b.py and scripts/diag_b.py before the v2 run was launched.

- Truth-init probe (scripts/diag_b.py 2): data loss 6.6e-05 at the true
  source, |grad| O(1) vs prior grad O(5e-6) -> the truth is a near-exact
  global minimum; the v1 failure was optimizer dynamics, not identifiability.
- S1-adam-tv.log — Adam lr 0.2 cosine-> 0.02, init 0.05*Q_SCALE, TV 3e-3,
  noise-free, 300 iters: rel_l2 plateaus at 0.588 (amp 0.969).
- S2-lbfgs-noisefree.log — same problem, L-BFGS-B: rel_l2 0.080 at 125
  evaluations, still descending (run stopped there; enough to decide).
- S3-lbfgs-noise2.log — L-BFGS-B with the protocol noise (sigma = 2 counts):
  rel_l2 0.070, amplitude ratio 0.989 at 125 evaluations; the transient
  amp 1.38 at ev 75 settles back, so lambda_tv = 3e-3 holds at this noise.
