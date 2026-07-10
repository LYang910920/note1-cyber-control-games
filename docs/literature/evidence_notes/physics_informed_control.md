# Physics-Informed Control Evidence Notes

## Design decisions supported by reviewed sources

- Mowlavi and Nabi compare PINN control with a direct-adjoint method and validate
  a learned control through an independent forward computation. Note 2 follows
  that pattern by reporting simulator rollout metrics separately from training
  residuals.
- Physics-informed PointNet conditions field prediction on irregular geometry.
  It motivates reusable node encoders and held-out geometry tests, but its
  steady PDE setting is not evidence that a graph SIPS inverse problem is
  identifiable.
- PINC conditions a short-interval neural dynamics model on the initial state and
  control, then chains intervals. This is listed as an extension route; the
  current examples do not claim long-horizon surrogate accuracy.
- LyZNet couples physics-informed control learning to a separate formal
  verification workflow. The tutorial therefore states explicitly that a small
  residual is not a proof of stability or optimality.
- The Pontryagin neural operator uses costate consistency to learn families of
  constrained games. It is an advanced feedback-learning direction, distinct
  from the foundation's open-loop FBSM solver.

## Claim boundary

The node-SIPS inverse experiment is a synthetic identifiability study. Its
community and feature-conditioned rates are evaluated on held-out times, nodes
and graph size, but do not identify real cyber rates without an observation and
measurement model supported by data.
