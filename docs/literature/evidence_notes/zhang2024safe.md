# Scalable Constrained MARL: Full-Text Evidence

## 1. Bibliographic record

- Citekey: `zhang2024safe`
- Title: *Scalable Constrained Policy Optimization for Safe Multi-agent Reinforcement Learning*
- Authors: Lijun Zhang, Lin Li, Wei Wei, Huizhong Song, Yaodong Yang and Jiye Liang
- Year and venue: 2024, NeurIPS 37, Main Conference Track
- DOI: `10.52202/079017-4400`
- Official URL: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html>
- Peer-review status: peer-reviewed official proceedings paper
- Final/preprint relationship: official proceedings version

## 2. Retrieval and text status

- Full-text source: official NeurIPS PDF
- Access basis: freely accessible publisher full text
- Local PDF filename: `zhang2024_scalable_safe_marl.pdf`
- SHA-256: `56941849f81b82e77e67384309e8871c754f206321a1b479e11aa99108d1907d`
- Page count: 33
- Text extraction method: native PDF text; no OCR
- Material inspected: Equations (1)-(20), Theorem 3.7, Algorithm 1, Figures 1-4,
  limitations and compute discussion

## 3. Repository question

- Question: can Note 1 describe its hard intervention budget or graph-context
  PPO as formally safe or scalable?
- Target repository: `note1-cyber-control-games`
- Target section/file/API: safety claim boundary in the MAPPO guide section

## 4. Research problem and contribution

The paper develops a decentralized sequential constrained-policy framework
using k-hop local information, trust-region bounds and truncated advantages,
then derives a practical Scal-MAPPO-L algorithm. Evidence: pp. 1-2 and 3-7.

## 5. Mathematical model

- Model: graph-structured constrained Markov game with product state/action
  spaces, joint reward and per-agent local costs.
- Objective: maximize discounted joint reward subject to every discounted cost
  threshold.
- Assumptions: spatial correlation decay for dynamics and policies.
- Semantics: discrete-time infinite-horizon constrained game, not a hard one-
  step action budget.

Evidence: pp. 2-3, Equations (1)-(5).

## 6. Solution or learning method

The paper decomposes joint advantages, bounds k-hop truncation error, and gives
reward and cost bounds. Theorem 3.7 covers an ideal sequential constrained
update. The practical Scal-MAPPO-L method adds per-agent Lagrange multipliers
and PPO clipping. The authors explicitly state that these approximations may
prevent the practical method from retaining the theorem's guarantee.

Evidence: pp. 4-7 and 24; Equations (8), (11)-(20); Theorem 3.7; Algorithm 1.

## 7. Neural architecture, when applicable

The paper specifies separate per-agent actors, value networks and cost-value
networks using k-hop state information without parameter sharing. Numeric depth,
width, activation and parameter count are not specified in the paper. Defaults
in the supplemental code are not treated as a universal architecture claim.

Evidence: pp. 6-7 and 24-25.

## 8. Training details, when applicable

Algorithm 1 gives symbolic batch, horizon, discount and step-size settings. A
supplemental Ant command uses Adam, actor learning rate `9e-5`, critic learning
rate `5e-3`, 16 rollout environments, 40 minibatches, five PPO epochs and
`10^7` steps. It is one experiment setting, not a general default. Hardware is
an RTX 4090 and i9-13900K. Evidence: pp. 24 and 26 plus the official supplement.

## 9. Experiments

Safe MAMuJoCo experiments use ManyAgent Ant, Ant and Coupled HalfCheetah with
2-12 agents. Baselines are IPPO, HAPPO and MAPPO-L. Results average at least
three seeds and report reward/cost curves and k-hop sensitivity, but there is no
held-out topology test or realistically measured communication cost. Reported
maximum-k runtimes range from 8.43 to 11.65 hours. Evidence: pp. 7-8 and 25-26,
Figures 1-4.

## 10. Limitations and failure modes

The guarantee can be uninformative when spatial decay fails. The practical PPO
approximation is not covered by the ideal theorem. Experiments are simulator-
only, use few seeds, do not measure actual communication and do not validate
cyber dynamics. Evidence: pp. 7, 24 and 26.

## 11. Code/data availability

- Code/data: the official proceedings record links a supplemental ZIP
- License: no clear license was found; reuse rights remain unresolved
- Reproduction status: not run in this review

## 12. Transfer assessment

- Directly reusable: terminology separating reward, explicit costs and
  conditional guarantees.
- Adaptable: k-hop graph policies only after adding per-agent costs, cost
  critics, Lagrange updates and evidence for spatial decay.
- Context only: MAMuJoCo architectures and performance.
- Incompatible: Note 1's coordinated one-action budget is a hard feasible
  action map, not a discounted cost constraint covered by Theorem 3.7.
- Required future evidence: decay diagnostics, k-hop ablations, held-out graphs,
  matched compute, multi-seed cost/reward and measured communication.

## 13. Decision

- Score: 15/21 (fit 2, technical evidence 3, transferability 1, evidence quality
  3, reproducibility 2, evaluation quality 2, novel information 2)
- Decision: background-only
- Target change: state the assumptions and prevent a formal-safety claim for
  the current action budget
- Allowed claim: the paper distinguishes reward optimization from cumulative
  constraints and studies local-information constrained MARL
- Claims that must not be made: Note 1 implements Scal-MAPPO-L, inherits
  Theorem 3.7 or obtains a safety guarantee from its action budget
- Reviewer and date: Codex full-text review, 2026-07-13

## 14. Evidence ledger

| Repository claim or design decision | Paper evidence | Location (page / Eq. / Alg. / Fig. / Table) | Confidence |
|---|---|---|---|
| Guarantees require spatial decay and sequential constrained updates | Assumptions 2.1-2.2 and Theorem 3.7 | pp. 3, 5-6; Eqs. (4)-(5), (16) | high |
| Practical PPO may not preserve the theorem | Authors' caveat | pp. 7 and 24 | high |
| A one-step hard budget is not a discounted safety constraint | Paper cost constraint versus repository action map | Eq. (3); Note 1 MAPPO guide | high |
