# Hierarchical MARL for Cyber Defense: Full-Text Evidence

## 1. Bibliographic record

- Citekey: `singh2025hierarchical`
- Title: *Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense*
- Authors: Aditya Vikram Singh, Ethan Rathbun, Emma Graham, Lisa Oakley,
  Simona Boboila, Peter Chin and Alina Oprea
- Year and venue: 2025, *Reinforcement Learning Journal*, 6:790-810
- Publisher and official URL: Reinforcement Learning Journal,
  <https://rlj.cs.umass.edu/2025/papers/Paper77.html>
- DOI: none stated
- Peer-review status: final peer-reviewed article; the supplement carries a
  separate peer-review caveat
- Final/preprint relationship: the reviewed file is the journal version

## 2. Retrieval and text status

- Full-text source: official journal PDF
- Access basis: freely accessible publisher full text
- Local PDF filename: `singh2025_hierarchical_cyber_marl.pdf`
- SHA-256: `bcda0ce937bd46abdeaf6161a6dbe105bf2b6f4e16ee716766d2bbdb91d41e5e`
- Page count: 21
- Text extraction method: native PDF text; no OCR
- Figures/tables inspected manually: Algorithms 1-2, Figures 1-12 and Tables 1-4

## 3. Repository question

- Question: what full-text evidence supports hierarchical PPO decomposition and
  cyber-operational metrics without transferring CybORG results to SIPS?
- Target repository: `note1-cyber-control-games`
- Target section/file/API: hierarchy and evaluation discussion in the main
  guide; no new learner API is introduced

## 4. Research problem and contribution

The paper studies decentralized defense of a partially observed enterprise
network with large local action spaces, shared blue reward and deceptive red
agents. It proposes expert-selected and learned-master hierarchical PPO,
indicator-enriched observations, sub-policy transfer and operational metrics.
The paper's priority claim for cyber hierarchical MARL was not independently
verified.

Evidence: PDF pp. 3-6 and 9-13.

## 5. Mathematical model

- Model: a Dec-POMDP with joint actions, local observations and a shared blue
  reward; the underlying simulator state order is not enumerated.
- Setting: five blue agents defend seven zones, with an additional persistent
  contractor foothold.
- Actions: Analyse, Deploy Decoy, Remove, Restore, Block and Allow; actions may
  last one to five decision steps.
- Objective: penalties for disrupted services, OT impact and Restore use.
- Semantics: discrete-time partially observed interaction with temporally
  extended actions and action masking, not a continuous-time SIPS flow.
- Game classification: the paper uses both general-sum and zero-sum language;
  this inconsistency is not transferred to the guide.

Evidence: PDF pp. 4-5 and 10; supplemental Table 4 on p. 18.

## 6. Solution or learning method

H-MARL Expert uses an indicator rule to choose Investigate, Recover or Control
Traffic and trains the selected PPO sub-policy. H-MARL Meta freezes pretrained
sub-policies and trains a PPO master over meta-actions. Per-agent IPPO supplies
the base learner; a collectively trained variant is an ablation. The paper
gives no convergence, equilibrium or safety theorem and no wall-clock analysis.

Evidence: Figure 1, p. 6; Algorithms 1-2, p. 7; Figure 2, p. 8.

## 7. Neural architecture, when applicable

- Actor and critic: feedforward networks with two hidden layers of 256 units.
- Outputs: categorical actor distribution and scalar critic; the master actor
  outputs a categorical sub-policy choice.
- Missing details: activation, normalization, initialization, PPO output
  parameterization and parameter count are not reported.
- Invalid actions are masked while a temporally extended action is executing.

Evidence: PDF pp. 9-10.

## 8. Training details, when applicable

- Learning rate: `5e-5`; discount: 0.99.
- Experience buffer: 1,000,000 samples; GAE is used.
- Minibatch: 32,768; 30 minibatch SGD iterations per outer update.
- Data collection: separate workers over 30 randomized network versions.
- Evaluation: 100 randomized episodes of 500 steps with episode-level standard
  deviations.
- Missing: optimizer class, PPO clip, GAE lambda, loss coefficients, training
  seeds, hardware and runtime.

Evidence: PDF pp. 9-10.

## 9. Experiments

The CybORG CAGE 4 experiments use Default, Aggressive, Stealthy and Impact
finite-state red policies; these are not adaptive learners. Baselines include
decentralized IPPO, a centralized-critic method, collective hierarchy, Expert
and Meta hierarchy, and fine-tuned Meta. Metrics include episodic return,
clean-host and non-escalated-host ratios, recovery time and precision, wasted
recoveries and OT impact. Parameter-count, wall-clock and independent-training-
seed budgets are not matched. Expert has the best reported mean return against
the four tested red policies, but this is benchmark-specific evidence.

Evidence: Figure 5, p. 10; Table 2, p. 11; Figure 6, p. 12; Table 3, p. 13.

## 10. Limitations and failure modes

Adaptive/evolving adversaries are left for future work. Expert indicator rules
may be difficult to define, one Control Traffic expert is unstable, and
communication gains are small. Additional limitations are missing topology
holdouts, training seeds, hardware, complete PPO settings and formal
guarantees. The results depend on CybORG observations, actions and rewards and
do not establish operational-network or SIPS performance.

Evidence: p. 12 and supplemental pp. 18-21.

## 11. Code/data availability

- Code URL: <https://github.com/adityavs14/Hierarchical-MARL>
- Environment: <https://github.com/cage-challenge/cage-challenge-4/tree/main>
- License: not stated in the paper and not inspected here
- Reproduction status: code and experiments were not run in this review

## 12. Transfer assessment

- Directly reusable: distinguish expert-selected from learned masters and
  report operational state metrics alongside return.
- Adaptable: master/sub-policy decomposition after defining genuine SIPS action
  classes and running a new budget-matched experiment.
- Context only: indicator observations, PPO hyperparameters and all CybORG
  performance values.
- Incompatible: recovery precision and clean-host metrics require host labels
  and detection semantics absent from aggregate SIPS.
- Required future tests: action partitioning, frozen sub-policies, matched
  training budgets, multiple seeds and held-out attackers/parameters.

## 13. Decision

- Score: 17/21 (fit 3, technical evidence 3, transferability 2, evidence quality
  2, reproducibility 3, evaluation quality 2, novel information 2)
- Decision: integrate, documentation and evaluation rationale only
- Target change: clarify the hierarchy paragraph and operational-metric
  reporting; do not add H-MARL code in this bounded pass
- Allowed claim: the paper evaluates expert-selected and learned-master PPO in
  CybORG CAGE 4 with operational metrics and transfer among four fixed red
  variants
- Claims that must not be made: equivalent SIPS performance, adaptive-attacker
  transfer, formal safety, equilibrium, or wall-clock speedup
- Reviewer and date: Codex full-text review, 2026-07-13

## 14. Evidence ledger

| Repository claim or design decision | Paper evidence | Location (page / Eq. / Alg. / Fig. / Table) | Confidence |
|---|---|---|---|
| Cyber defense can be represented as a Dec-POMDP | Tuple, local observations, joint actions and shared reward | p. 4 | high |
| Hierarchy separates master selection from primitive actions | Master and transformed sub-policy observations | Fig. 1, p. 6; Algs. 1-2, p. 7 | high |
| Expert and learned masters have different training semantics | Expert trains sub-policies; Meta freezes them | Algs. 1-2, p. 7; Fig. 2, p. 8 | high |
| The reported networks use two 256-unit hidden layers | Actor/critic settings | pp. 9-10 | high |
| Results concern four stationary CybORG adversaries | Returns by adversary | Table 2, p. 11 | high |
| Return should be accompanied by operational metrics | Recovery and security-posture measures | Table 3, p. 13 | high |
| Transfer evidence is limited | Fixed red variants; adaptive attackers deferred | Fig. 6, p. 12; supplemental Fig. 12, p. 21 | medium |
