# Cooperative MAPPO: Full-Text Evidence

## 1. Bibliographic record

- Citekey: `yu2022mappo`
- Title: *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
- Authors: Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang,
  Alexandre Bayen and Yi Wu
- Year and venue: 2022, NeurIPS 35, Datasets and Benchmarks Track
- Official URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html>
- DOI: none shown by the official proceedings record
- Peer-review status: peer-reviewed conference paper
- Final/preprint relationship: official proceedings paper and supplement

## 2. Retrieval and text status

- Full-text source: official NeurIPS paper and supplement
- Access basis: freely accessible publisher full text
- Local PDF filenames: `yu2022_mappo.pdf` and
  `yu2022_mappo_supplement.pdf`
- SHA-256: `3de7db922d915b4e6d7b3a05b07163a6eeaa5bf0d56fca019eef044b469686ff`
  and `046db303c1652f493b870831b27873ac778871561a67220a29053b5d2b676976`
- Page counts: 14 and 19
- Text extraction method: native PDF text; no OCR
- Figures/tables inspected: main Figures 1-8 and Tables 1-3; supplement
  Algorithm 1 and Tables 4, 6-16

## 3. Repository question

- Question: which MAPPO implementation choices are supported by full-text
  evidence, and which benchmark claims remain out of scope?
- Target repository: `note1-cyber-control-games`
- Target section/file/API: MAPPO guide section, `cybergames.mappo` and its
  recorded configuration

## 4. Research problem and contribution

The paper tests whether PPO is a competitive baseline for cooperative MARL. It
does not introduce a new algorithm; its contribution is a four-benchmark study
and ablation of implementation choices. Evidence: main pp. 1-2 and 7-10.

## 5. Mathematical model

- Model: shared-reward DEC-POMDP with global state, local observations, joint
  action, transition kernel and discounted team reward.
- Actor information: each actor chooses `a_i` from its local observation.
- Critic information: MAPPO uses global information during training; IPPO uses
  local information for both actor and critic.
- Semantics: discrete-time, partially observed and cooperative; no safety
  constraints, resets or equilibrium claim.

Evidence: main p. 3, Sections 3.1-3.2.

## 6. Solution or learning method

MAPPO uses separate actor and critic networks, optional parameter sharing,
GAE, advantage normalization, value clipping, PPO ratio clipping and repeated
updates on an on-policy rollout. The study provides empirical comparisons and
no convergence theorem. Evidence: main p. 3; supplement pp. 1-2, Algorithm 1.

## 7. Neural architecture, when applicable

- Actor input/output: local observation to categorical probabilities; the
  paper also describes Gaussian outputs for continuous actions, although the
  reported benchmarks are discrete.
- Critic: global state input and scalar value output.
- MPE/SMAC/football: two 64-wide fully connected layers, a 64-unit GRU and a
  final fully connected layer.
- Hanabi: two 512-wide fully connected layers.
- ReLU/tanh choice is scenario specific; orthogonal initialization and feature
  normalization are used.
- No graph encoder, pooling layer or parameter count is reported.

Evidence: supplement pp. 1 and 12-14, Tables 6-9 and 13-16.

## 8. Training details, when applicable

- Adam with epsilon `1e-5`, zero weight decay, discount 0.99 and GAE lambda
  0.95.
- Gradient norm 10; Huber value loss with delta 10; reward and feature
  normalization; recurrent chunk length 10.
- Learning rates, epochs and minibatches vary by benchmark; the paper's main
  ablations show that these choices materially affect performance.
- Repetitions: 10 seeds for MPE, 6 for SMAC and football, and at least 3 for
  Hanabi.
- Hardware: 64-core CPU, 256 GB RAM and RTX 3090.

Evidence: main pp. 4-10; supplement pp. 12-14.

## 9. Experiments

The experiments cover MPE, SMAC, Google Research Football and Hanabi. Baselines
include MADDPG, QMix and several benchmark-specific methods. Metrics include
return, win/success rate, sample efficiency and seed standard deviation. Some
baseline results are imported from cited work and one football comparison is
explicitly indirect. There is no held-out-domain, cyber, safety or graph-policy
evaluation. Evidence: main pp. 4-6, Tables 1-3, and pp. 7-10, Figures 2-8.

## 10. Limitations and failure modes

The authors limit the evidence to cooperative, discrete-action and mostly
homogeneous benchmarks and identify the lack of theory. Additional transfer
risks are environment-specific tuning, imported baseline results and no
held-out, cyber, safety or equilibrium study. Evidence: main p. 10.

## 11. Code/data availability

- Code URL: <https://github.com/marlbenchmark/on-policy>
- License: currently MIT in the upstream repository; the paper itself does not
  state a license
- Data: third-party simulators, no new static dataset
- Reproduction status: not run in this evidence review

## 12. Transfer assessment

- Directly reusable: centralized value information, GAE, clipped PPO and
  implementation ablations.
- Adaptable: normalization, epoch/minibatch and critic-input ablations on held-
  out graph seeds with matched parameter budgets.
- Context only: benchmark architectures, returns and hardware.
- Incompatible: strict MAPPO execution semantics. This repository's allocator
  compares communities and selects one joint action rather than letting each
  actor select independently from a local observation.

## 13. Decision

- Score: 20/21 (fit 3, technical evidence 3, transferability 3, evidence quality
  3, reproducibility 3, evaluation quality 3, novel information 2)
- Decision: integrate
- Target change: retain the MAPPO-style label and expose implementation choices
  and claim boundaries in the guide/configuration
- Allowed claim: carefully configured MAPPO is a strong cooperative baseline on
  the four tested benchmark families
- Claims that must not be made: cyber effectiveness, formal safety,
  equilibrium, graph-encoder validation or strict decentralized execution
- Reviewer and date: Codex full-text review, 2026-07-13

## 14. Evidence ledger

| Repository claim or design decision | Paper evidence | Location (page / Eq. / Alg. / Fig. / Table) | Confidence |
|---|---|---|---|
| Local actors and a training-only centralized critic define MAPPO | MAPPO/IPPO definitions | main p. 3 | high |
| GAE, clipping, normalization and data reuse affect outcomes | Implementation study and ablations | main pp. 7-10; supplement pp. 1, 12-14 | high |
| The repository learner is MAPPO-style, not strict MAPPO | Paper actors choose local actions; repository chooses a coordinated joint action | main p. 3; `cybergames.mappo` | high |
