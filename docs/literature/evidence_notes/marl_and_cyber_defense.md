# MARL and Cyber-Defense Evidence Notes

## Design decisions supported by reviewed sources

- The NeurIPS MAPPO study supports a decentralized actor and centralized value
  function as a strong cooperative baseline. It also makes implementation
  choices such as GAE, clipping, normalization and mini-batch updates part of
  the experimental record. It does not establish an equilibrium or a cyber
  safety guarantee.
- The 2024 constrained-MARL study separates reward optimization from explicit
  safety constraints and studies local interaction. In this repository,
  intervention budgets are therefore enforced by the action map rather than
  represented only by a reward penalty.
- The hierarchical CybORG preprint reports clean-host ratio, recovery precision,
  false positives and recovery time in addition to episodic return. Note 1 uses
  the same reporting principle: training return is accompanied by infection,
  action-budget and unilateral-response diagnostics.
- The peer-reviewed Computers & Security hierarchical game is relevant to the
  high-level target/low-level action decomposition. Its PDF remains requested,
  so no unverified layer dimensions or benchmark values are copied here.

## Claim boundary

The current Note 1 experiments are controlled SIPS simulators, not CybORG or
operational networks. The literature motivates baselines and diagnostics; it
does not transfer external benchmark performance to this repository.
