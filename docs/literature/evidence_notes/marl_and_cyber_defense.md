# MARL and Cyber-Defense Evidence Notes

## Full-text-reviewed evidence

- The NeurIPS MAPPO study supports decentralized actors with a centralized value
  function as a cooperative baseline. It also records GAE, clipping,
  normalization and mini-batch choices. It does not establish a cyber-defense
  safety guarantee or a game equilibrium.
- The 2024 constrained-MARL paper separates reward improvement from explicit
  constraints. Note 1 therefore enforces intervention budgets in the action map
  and does not call that mechanism a formal safety guarantee.
- Singh et al., now published in the 2025 Reinforcement Learning Journal, report
  hierarchical PPO in CybORG CAGE 4 and operational metrics alongside return.
  This supports the reporting categories and hierarchy discussion, not transfer
  of their benchmark performance to the SIPS simulator.

## Metadata or abstract-level leads

- The IEEE safe-RL review supplies constraint and deployment-safety terminology;
  its full text has not been reviewed for this repository.
- The Purves et al. cyber-defense paper motivates reporting operational state
  metrics in addition to return. No causal-identification claim is transferred.
- The Computers & Security hierarchical game remains in `PDF_REQUESTS.md`.
  Architecture details are not used until a legally obtained full text is
  reviewed.

## Claim boundary

The current experiments are controlled SIPS simulators, not CybORG or
operational networks. Fixed-policy cross-play is not best-response retraining,
an exploitability estimate, or an equilibrium certificate.
