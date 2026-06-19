# Note 1 Training Diagnostic Glossary

Use this page while reading the training-diagnostic figures and CSV histories.

| Term | Meaning | How to read it |
|---|---|---|
| `iteration` | One optimizer, FBSM, or residual-update step. | Use for solver or neural-training progress on the x-axis. |
| `episode` | One complete sampled-data rollout used by RL/MARL training. | Use for DDQN, PPO, MAPPO, or CTDE learning curves. |
| `rollout` | A forward simulation under a fixed policy, control, or parameter set. | Use for validation outside the training loss. |
| `control-update change` | Maximum change between consecutive FBSM controls or strategies. | A convergence diagnostic; should decay toward tolerance. |
| `training return` | Cumulative reward collected during learning, often with exploration. | Noisy; do not treat one point as policy quality. |
| `evaluation return` | Cumulative reward from the current policy under a fixed evaluation setting. | Use rolling trends and compare with cyber metrics. |
| `loss` | The scalar objective minimized by an optimizer. | Must be read with its component losses. |
| `rolling mean` | Moving average of recent noisy values. | Shows trend without hiding stochastic variability. |
| `baseline comparison` | Same-model comparison with no-control, fixed, random, or simple learned policies. | Use before making a stronger method claim. |
