# FinTool-RL: signal-efficient curriculum for long-horizon tool RL

## 1. Project identity

> Study whether a model-conditional graph curriculum can increase the useful policy-gradient signal produced
> per rollout token in long-horizon tool RL, using programmatically generated financial tasks with exact
> execution rewards.

Working resume title:

> **Signal-Efficient Curriculum DrGRPO for Long-Horizon Tool Agents**

The financial domain is the controlled experimental substrate, not the product. The contribution is an
agentic-RL data-selection method and its causal ablations, not another customer-facing agent.

## 2. Separation from the e-commerce project

| E-commerce agent | FinTool-RL |
|---|---|
| production trustworthiness and runtime architecture | post-training algorithm and learning dynamics |
| dialogue, memory, writes, handoff, RAG | read-only executable computation graphs |
| failure taxonomy drives product improvement | graph difficulty controls learning-signal availability |
| guardrails and operational reliability | advantage mass, rollout-token efficiency, and generalization |
| LoRA as domain adaptation | SFT only initializes the RL policy |
| external evaluation of a system | controlled ablation of RL sampling strategies |

Grounding, provenance, hallucination checks, and runtime correctness remain validity constraints. Abstention is
an optional follow-up experiment after the core RL study; it is not part of the main reward or headline claim.

## 3. Research question

For a binary terminal reward, groups that are all-fail or all-pass provide no relative learning signal. Uniform
sampling can therefore waste expensive multi-turn rollouts on task regions that are currently too hard or too
easy. The useful region moves after every policy update.

Main question:

> At a fixed generated-rollout-token budget, can a policy-version-aware graph curriculum produce more expected
> DrGRPO advantage mass than uniform sampling, a frozen offline band, and post-generation filtering—without
> abandoning long-horizon graph strata or changing the held-out objective?

This separates four claims: H1 signal production, H2 sample efficiency, H3 capability, and H4 transfer. Later
claims are not implied by earlier ones.

## 4. Core method

### 4.1 Parameterized task graph

Every task belongs to a controllable graph cell:

```text
(template family,
 tool-call count,
 graph depth,
 observation reuse count,
 discovery requirement,
 distractor count,
 entity/time split)
```

The generator creates fresh oracle-verified instances from a frozen SEC snapshot while changing one structural
factor at a time.

| Graph family | Calls | Controlled difficulty |
|---|---:|---|
| fact lookup | 1 | enum selection and final grounding |
| growth/margin/ratio | 3 | two observations into one calculator |
| latest-period discovery | 3–4 | environment query before fact selection |
| growth of growth | 6 | calculator outputs reused as observations |
| margin change | 7 | two branches followed by a merge |
| cross-company comparison | 5–7 | entity binding and symmetric branches |

### 4.2 Pure terminal outcome and chosen optimizer

The main experiment uses one binary terminal outcome:

```text
r = 1  iff  no hard failure AND answer correct AND grounded
r = 0  otherwise
```

Contract, format, efficiency, and intermediate-step scores are diagnostics only. They do not enter the main
reward, routing estimator, or zero-variance filter. This keeps the curriculum model identifiable; shaping is a
separate one-seed ablation.

The main optimizer is **DrGRPO**, fixed before training. Standard GRPO divides group advantages by the group
reward standard deviation, which can amplify tiny empirical variance and changes the optimal curriculum. With
group size `G` and `k` successes:

```text
standardized-GRPO absolute advantage mass = 2 * sqrt(k * (G-k))
DrGRPO absolute advantage mass            = 2 * k * (G-k) / G
E[DrGRPO mass | p]                         = 2 * (G-1) * p * (1-p)
```

The mixed-group probability

```text
q(p) = 1 - p^G - (1-p)^G
```

is still reported, but it is only the probability that a group is non-degenerate. It is not the expected
gradient signal. A pre-registered ablation compares `q`-driven acquisition with expected-advantage-mass-driven
acquisition.

### 4.3 Offline band and the two readiness gates

After RS-SFT, sample 32 trajectories per train task with the future rollout policy. Let `c` be the number of
binary successes.

- core band: `c=4..28`;
- frontier: `c=1..3` or `29..31`;
- apparent all-fail/all-pass: `c=0` or `32`, confirmed with another 32 samples.

Two different facts must not be conflated:

1. **Learnability gate:** a meaningful train subset has `0 < p < 1`, so some groups can carry signal.
2. **Opportunity gate:** under the target layer weights, uniform sampling is predicted to waste a substantial
   rollout-token share on all-fail/all-pass groups.

The initial engineering opportunity threshold is

```text
W_uniform = token-weighted E[p^G + (1-p)^G] >= 0.30
```

and the pool must also retain learnable cells in every graph-depth stratum used for the main claim. A uniform
zero-variance rate is therefore an opportunity for the curriculum, not automatically a data defect. If nearly
all cells sit around `p=0.5`, RL may be learnable but there is little curriculum upside. If the band is empty,
RL is not ready. If the band exists but `W_uniform < 0.30`, vanilla RL may be viable but this curriculum project
is not identified. Thresholds are engineering values frozen before test evaluation.

### 4.4 Cell-exchangeability kill test

A cell-level Beta model assumes instances within a cell are sufficiently exchangeable. M2.5 therefore performs
a pre-registered variance decomposition on the existing 32 samples per task.

- estimate between-cell and within-cell/instance variance and bootstrap the intra-class correlation (ICC);
- compare held-out calibration/Brier score of a cell-only model with a global baseline and a hierarchical model;
- the cell-only controller is allowed only if the bootstrap lower bound for ICC exceeds `0.10` **and** it improves
  held-out Brier score over the global model.

`0.10` is an engineering threshold and must be stress-tested in synthetic simulations before it is frozen. If
the test fails, do not average away bimodal tasks. Upgrade to a partially pooled Beta-Binomial controller with a
cell prior and instance covariates such as entity, value scale, and fiscal year. Report both models; do not choose
the hierarchy after seeing test results.

### 4.5 Policy-version-aware posterior

A cumulative posterior estimates historical average difficulty, not current-policy difficulty. After each
rollout-policy weight synchronization, update each cell `b` with discounted sufficient statistics:

```text
alpha_b <- gamma_t * alpha_b + successes_b
beta_b  <- gamma_t * beta_b  + failures_b
gamma_t in [0.90, 0.98]
```

`gamma_t` is fixed by a pre-registered monotone schedule from measured policy KL: larger sync-to-sync KL implies
smaller `gamma_t`, because old rollouts are more stale. The schedule is calibrated in simulation and train-only
replay, then frozen. Every observation records policy version; rollouts beyond the allowed version lag are not
used to update the controller. Ablations compare cumulative (`gamma=1`), fixed discount, and KL-adaptive discount.

### 4.6 Stratified token-budget acquisition

A global score divided by expected tokens collapses toward short tasks. Coverage is therefore a constraint, not
a multiplicative bonus.

1. Partition tasks into registered graph-depth strata, with discovery/reuse strata split further when required.
2. Allocate a minimum **generated rollout-token quota** to every stratum; quotas sum to the total budget and are
   identical across methods.
3. Compare acquisition scores only among cells inside the same stratum.
4. Within a stratum, select by posterior expected DrGRPO advantage mass per expected generated token:

```text
A_b = E[2 * (G-1) * p_b * (1-p_b)] / E[generated rollout tokens_b]
```

5. Instantiate a fresh entity/year graph from the chosen cell.

The token quota, rather than prompt count, is the protection for long graphs. This makes H3 about improving long-
horizon capability under an explicit long-graph training allocation, not merely harvesting cheap short examples.

### 4.7 Distribution-bias question

Adaptive sampling changes the prompt distribution. Compare:

1. the curriculum objective evaluated on the full frozen target distribution;
2. clipped target-to-sampling correction `d(b)/mu(b)` within each registered stratum.

This ablation separates more useful signal from an unreported objective shift.

## 5. Falsifiable hypotheses

### H1 — signal density

At matched generated rollout tokens, the curriculum produces higher total absolute DrGRPO advantage mass per
generated token than E1, E3, and E4. Non-degenerate-group probability is secondary. For E4, the denominator is
all generated tokens, including groups discarded after generation.

### H2 — sample efficiency

Learning curves use `x = cumulative generated rollout tokens` (primary) and GPU-hours (systems secondary), with
`y = full held-out execution success`. The scalar endpoint is tokens/GPU-hours required to reach a pre-registered
held-out success level. Episodes and optimizer updates are reported, not simultaneously matched.

### H3 — capability

With identical starting checkpoint, trainable parameters, target stratum token quotas, and primary rollout-token
budget, curriculum DrGRPO improves held-out long-graph success over the SFT checkpoint and matched additional SFT.

### H4 — generalization

The gain transfers to unseen entities and held-out graph compositions, rather than only frequently selected cells.

Failure of H1 rejects the sampler mechanism. H1 success with H2 failure means additional signal did not translate
into learning efficiency. H2 success without H3 means the chosen target was too weak or gains were transient. H3
success without H4 indicates graph-distribution overfitting.

## 6. Data contribution

```text
SEC companyfacts JSON
    -> point-in-time canonical snapshot
    -> parameterized computation graph
    -> executable oracle
    -> native multi-turn trajectory
    -> exact binary terminal outcome
```

Required properties include canonical XBRL mapping, filing-time cutoff, company-disjoint and graph-disjoint
evaluation, fresh train instances, snapshot hashes, graph metadata, and measured token cost. Analysis reports
instance-level as well as cell-level success so that the exchangeability assumption remains testable.

## 7. Training chain

### M0 — fair baseline

Freeze `Qwen/Qwen3-4B-Instruct-2507`, revision, native function calls/final answer, and official sampled decoding.
Greedy decoding is a separate deployment diagnostic.

### M1 — RS-SFT initialization

RS-SFT teaches environment contracts and creates non-zero support for successful long graphs. Stop when contract
errors saturate, failure-matched SFT has diminishing dev return, a mixed-success region remains, and the matched
additional-SFT control is frozen.

### M2.5 — theory gate with measurements

Collect n=32 train-only outcomes, token costs, opportunity/learnability gates, ICC kill test, and controller
calibration. These measurements instantiate pre-written decisions; they do not invent the method after training.

### M3 — optimizer baselines

Run uniform DrGRPO and frozen-band DrGRPO with pure binary outcome reward and fixed stratum token quotas.

### M3.5 — online curriculum

Run discounted, policy-version-aware, expected-advantage-mass acquisition within each token-budget stratum.

### M3.6 — causal ablations

Test acquisition statistic, posterior forgetting, hierarchy, distribution correction, fresh instances, and reward
shaping without expanding the three-seed core.

## 8. Experiment matrix and compute discipline

Core experiments receive three seeds:

| ID | Method | What it isolates |
|---|---|---|
| E1 | uniform DrGRPO | uniform-waste and cost baseline |
| E3 | static n=32 band DrGRPO | offline model-conditional selection |
| E4 | post-generation non-degenerate filtering | cost of paying before filtering |
| E5 | online discounted advantage-mass curriculum | proposed method |
| E7 | matched additional SFT | whether RL adds value beyond supervision |

One-seed mechanism ablations, promoted to three seeds only if they change the conclusion:

| ID | Ablation |
|---|---|
| A1 | `q(p)` acquisition versus expected DrGRPO advantage mass |
| A2 | cumulative versus fixed-discount versus KL-adaptive posterior |
| A3 | cell-only versus hierarchical controller, conditional on kill test |
| A4 | clipped distribution correction |
| A5 | fresh instances disabled |
| A6 | terminal reward plus shaping versus pure binary terminal outcome |

All methods share the starting checkpoint, trainable parameters, decoding, group size, target stratum quotas, and
primary generated-token budget. Generated tokens are the sample-efficiency axis; GPU-hours are the systems axis.
Updates and episodes are reported covariates, not impossible additional matching constraints.

## 9. Metrics

### Primary mechanism metric

- total absolute DrGRPO advantage mass / all generated rollout tokens.

### Secondary mechanism metrics

- non-degenerate groups / generated groups;
- predicted versus observed advantage mass and `q(p)`;
- generated and retained tokens by graph stratum;
- posterior calibration, policy-version lag, and sync-to-sync KL;
- ICC, Brier score, and cell/instance residuals;
- entropy, KL, clip fraction, gradient norm, and importance-weight effective sample size.

### Capability and efficiency metrics

- full held-out execution success and success by graph depth/family/reuse/discovery;
- unseen-entity and held-out-composition success;
- cumulative generated tokens and GPU-hours to registered success;
- tool validity and grounding as non-inferiority constraints;
- sampled pass@1/pass@8 as diagnostics for probability concentration and diversity.

## 10. Interview depth

The project should support derivations and design defenses, not only pipeline narration:

- derive `q(p)` and explain why it is not gradient magnitude;
- derive expected DrGRPO advantage mass and contrast standard GRPO normalization;
- defend generated tokens as the denominator and explain why E4's post-filter rate is otherwise tautological;
- explain posterior staleness and the KL-to-discount schedule;
- show how stratified token quotas prevent short-task collapse;
- present the ICC kill test and hierarchical fallback;
- separate curriculum opportunity, RL learnability, capability improvement, and transfer;
- explain why matched SFT remains mandatory.

## 11. Twenty-minute narrative

1. **Question:** expensive tool rollouts often generate no relative signal.
2. **Data:** executable graphs expose controlled difficulty and fresh instances.
3. **Theory:** DrGRPO advantage mass identifies useful current-policy regions; `q` alone does not.
4. **Method:** discounted posterior plus within-stratum acquisition under token quotas.
5. **Validity:** exchangeability kill test, objective-shift correction, and pure binary reward.
6. **Evidence:** three-seed core against uniform, static band, post-filtering, and additional SFT.

## 12. Allowed resume claims

Before experiments:

> Built a replayable financial tool environment and designed a policy-version-aware DrGRPO curriculum with
> pre-registered falsification gates.

After H1 only:

> Increased DrGRPO advantage mass per generated rollout token using stratified online graph selection.

After H1–H3:

> Improved long-horizon tool execution at matched generated-token budget over uniform RL, static band selection,
> post-generation filtering, and additional SFT.

After H4:

> Demonstrated transfer to unseen entities and held-out computation-graph compositions.

Do not headline abstention, production reliability, or generic agent runtime quality in this project.
