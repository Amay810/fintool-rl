# RL readiness and failure-prevention protocol

Status: pre-registered engineering protocol. Values marked **engineering threshold** are calibrated with
simulation/train/dev only and frozen before test evaluation.

## 0. No-experiment-before-theory rule

Pilot runs may verify that code realizes this protocol; they may not be used to discover avoidable objective or
experimental-design choices. Before GPU training, every item below needs a written answer and executable check.

### Mainline proof obligations

1. **Reward identity:** the curriculum, filter, and optimizer all use the same pure binary terminal outcome:
   success iff there is no hard failure, the answer is correct, and it is grounded.
2. **Policy support:** the SFT policy has non-zero probability of successful trajectories on a meaningful train
   subset.
3. **Learnability:** at least one registered graph stratum contains tasks with `0 < p(success) < 1`.
4. **Curriculum opportunity:** uniform sampling has substantial token-weighted all-fail/all-pass waste; otherwise
   RL may be worthwhile but the proposed curriculum is not identified.
5. **Informative contrast:** reward depends on policy-controlled choices, not parser repair, response length,
   entity identity, leakage, or an environment artifact.
6. **Estimator validity:** cell exchangeability passes the ICC/calibration kill test, or the controller is upgraded
   to the pre-specified hierarchical model.
7. **Policy recency:** posterior observations carry policy versions and stale evidence is discounted after weight
   synchronization.
8. **On-policy identity:** generated token IDs, masks, template/tool serialization, policy version, and old-policy
   log probabilities match the sampled rollout.
9. **Trust region:** KL, clipping, learning rate, and gradient-norm limits are frozen before the run.
10. **Generalization and power:** splits, seeds, paired estimator, minimum effect, and sample size can distinguish
    the intended claim from noise.
11. **Resource feasibility:** the static four-GPU placement fits with memory margin.

If any relevant obligation is unresolved, the decision is `NO-GO`; it is not delegated to a terminal experiment.

### Required artifacts

- binary terminal reward truth table and adversarial verifier suite;
- symbolic derivation of non-degenerate probability and optimizer-specific advantage mass;
- frozen group size, decoding, pass@k estimator, optimizer variant, and loss normalization;
- learnability and opportunity gate definitions;
- graph-stratum token quotas and target evaluation weights;
- ICC variance decomposition, calibration test, and hierarchical fallback;
- KL-to-posterior-discount schedule and allowed policy-version lag;
- token/logprob/weight-sync invariance specification;
- static GPU placement/memory budget;
- frozen split manifest, leakage proof, primary curve/endpoint, sample-size analysis, seeds, and stop rule;
- a causal table mapping every null/regression pattern to the assumption it falsifies.

## 1. Fixed model and decoding

Primary model: `Qwen/Qwen3-4B-Instruct-2507` (non-thinking only), with exact revision frozen.

| Name | Parameters | Purpose |
|---|---|---|
| greedy diagnostic | temperature 0 | deterministic deployment diagnostic |
| sampled baseline/headroom/RL | temperature 0.7, top-p 0.8, top-k 20, min-p 0 | pass@k and rollout distribution |

Never label a temperature-zero result pass@1. Headroom and RL use the same sampling policy. Source:
[Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

## 2. Optimizer and reward contract

### Mainline optimizer

Use DrGRPO for E1/E3/E4/E5. Do not switch between standardized GRPO and DrGRPO after observing outcomes.
For group size `G`, `k` binary successes, and task success probability `p`:

```text
non-degenerate probability q(p)          = 1 - p^G - (1-p)^G
standard GRPO absolute advantage mass    = 2 * sqrt(k * (G-k))
DrGRPO absolute advantage mass           = 2 * k * (G-k) / G
expected DrGRPO mass                     = 2 * (G-1) * p * (1-p)
```

`q(p)` is a reporting and ablation statistic, not the primary acquisition target. The main curriculum maximizes
posterior expected DrGRPO mass per generated token inside each graph stratum. A one-seed ablation compares `q`-
driven routing. Rationale and evidence for DrGRPO-style normalization:
[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783).

### Mainline reward

```text
r = 1 iff no hard failure AND answer_correct AND grounded
r = 0 otherwise
```

All group variance, filtering, posterior updates, and acquisition labels use this terminal outcome only. Format,
tool-contract, intermediate, and efficiency dimensions remain diagnostics. Reward shaping is isolated to a one-
seed appendix ablation and may not redefine mainline zero-variance groups.

## 3. Framework decision and systems proof

Run a time-boxed Agent Lightning + verl integration smoke first because it can preserve the current Python agent
loop. If exact token identity or native tool-call fidelity is not demonstrated within two working days, switch to
native verl `ToolAgentLoop`/`BaseTool`; do not write a custom GRPO loop.

| Framework | Multi-turn tools | GRPO/LoRA | Placement | Decision |
|---|---|---|---|---|
| Agent Lightning + verl | yes | through verl | verl-managed | first smoke |
| native verl | native | native | hybrid engine | immediate fallback |
| SkyRL | native | native | colocated mode | second fallback |
| ROLL | native | native | device mapping | capable, too heavy initially |
| verl-tool | strong | inherited | inherited | avoid pinned older stack initially |

Four A100 40 GB feasibility remains a gate, not an assumption. Initial smoke: 4B LoRA rank 16/alpha 32,
FSDP/FSDP2, gradient checkpointing, actor/optimizer/reference offload, TP=1 rollout replicas, vLLM utilization
0.35–0.45, maximum length 8192, and one optimizer update.

Pass conditions: peak reserved memory below 38 GB/GPU; finite loss/gradient/KL; exact token/loss-mask identity;
successful weight synchronization; at least one mixed binary-reward group; no placement deadlock.

Primary sources: [Agent Lightning](https://github.com/microsoft/agent-lightning),
[verl agentic RL](https://github.com/volcengine/verl/blob/main/docs/start/agentic_rl.rst), and
[verl](https://github.com/verl-project/verl).

## 4. Headroom, opportunity, and model-validity gates

### Sampling protocol

For eligible train tasks, collect `n=32` independent trajectories and retain invalid/truncated failures. With `c`
successes:

```text
sampled pass@1 = c / n
pass@k = 1 - C(n-c, k) / C(n, k)
```

Source: [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374).

Initial routing (**engineering threshold**): core `c=4..28`; frontier `c=1..3` or `29..31`; apparent dead or
saturated `c in {0,32}` receives 32 confirmatory samples.

### Gate A — RL learnability

GO only if every graph-depth stratum needed for the main long-horizon claim contains a pre-registered minimum
number of tasks with posterior support for `0 < p < 1`. Initial requirement: at least 30 core/frontier tasks per
claimed stratum and at least 150 across train. If this fails, expand the task generator or narrow the claim before
training.

### Gate B — curriculum opportunity

Estimate under the target graph-stratum token weights:

```text
W_uniform = token-weighted E[p^G + (1-p)^G]
```

The curriculum experiment proceeds only if `W_uniform >= 0.30` (**engineering threshold**) while Gate A passes.
High uniform zero-variance is then evidence of opportunity, not a reason to rebalance it away. If Gate A passes
but `W_uniform < 0.30`, run vanilla RL if desired but do not claim the curriculum has room for a 30–50% signal-
efficiency gain. If Gate A fails, no amount of uniform waste makes RL learnable.

### Gate C — cell-exchangeability kill test

Using the n=32 task outcomes:

1. estimate between-cell, between-instance, and rollout variance;
2. bootstrap ICC and evaluate held-out Brier score/calibration;
3. allow a cell-only Beta controller only if the ICC lower confidence bound is above `0.10` and cell-only Brier
   score beats a global model;
4. otherwise use the pre-specified partially pooled Beta-Binomial hierarchy with cell prior plus entity, value-
   scale, and fiscal-year features.

The `0.10` threshold is frozen only after synthetic simulation establishes its operating characteristics. Test
data never chooses between controller forms.

### Statistical reporting

- task-level bootstrap for macro metrics and paired stage deltas;
- hierarchical bootstrap when rollout uncertainty matters;
- three seeds for E1/E3/E4/E5/E7, one seed for mechanism ablations;
- 200–400 independent held-out tasks initially, finalized by paired-disagreement power analysis;
- full held-out distribution is primary; selected-band results are diagnostic only.

## 5. Online controller rules

### Posterior forgetting

After each rollout-policy weight sync:

```text
alpha_b <- gamma_t * alpha_b + successes_b
beta_b  <- gamma_t * beta_b  + failures_b
gamma_t in [0.90, 0.98]
```

Larger sync-to-sync policy KL maps monotonically to smaller `gamma_t`. The piecewise schedule and KL bins are
calibrated by simulation/train-only replay, then frozen. Every rollout carries a policy version; observations
older than the permitted version lag are excluded. Compare cumulative (`gamma=1`), fixed discount, and adaptive
discount as a one-seed ablation.

### Stratified generated-token budgets

Define graph-depth strata before training and reserve a minimum generated-rollout-token quota for each. Quotas
are equal across methods. Acquisition scores are compared only within a stratum:

```text
A_b = E[2 * (G-1) * p_b * (1-p_b)] / E[generated rollout tokens_b]
```

Unused quota cannot silently migrate to short graphs; any reallocation follows a pre-registered rule and remains
visible in reporting. This replaces multiplicative coverage bonuses.

### Post-generation filtering accounting

E4 filters using terminal binary-outcome variance only. Its numerator may count retained advantage mass, but its
denominator includes **all generated rollout tokens**, including discarded groups. “Non-zero variance after
filtering” is not a valid H1 metric because it is mechanically 100%.

## 6. Why RL can show no gain or regress

| Failure | Pre-training diagnosis or earliest signal | Prevention/action |
|---|---|---|
| no policy support | n=32/64 all-fail across claimed strata | add valid SFT support or redesign tasks; NO-GO |
| no curriculum opportunity | cells concentrated near p≈0.5; low `W_uniform` | drop curriculum claim; use uniform RL |
| no learnable band | no cells with `0<p<1` | expand difficulty range; NO-GO |
| cell averaging hides bimodality | low ICC or poor cell-only calibration | hierarchical partially pooled controller |
| posterior staleness | predicted mass lags observed mass after sync | discount evidence; reduce allowed version lag |
| short-task collapse | long-stratum token quota underfilled | hard quota and within-stratum acquisition |
| reward shaping corrupts routing | shaped variance with identical terminal outcomes | pure terminal labels; shaping only in ablation |
| reward hacking | train reward rises while held-out success/grounding falls | adversarial verifier suite and binary truth table |
| narrow-band forgetting | selected slice rises while full dev falls | full-distribution canary, KL and non-inferiority gate |
| entropy collapse | pass@1 rises while pass@8/entropy falls | report both; adjust KL/entropy, stop if capability falls |
| rollout/train mismatch | token IDs/logprobs disagree or ratios spike at step 1 | same serialization/tokenizer; weight-sync check |
| objective shift | curriculum slice rises, corrected/full target does not | stratum quotas and clipped correction ablation |
| sparse credit | advantage mass rises but H2 remains flat | falsifies usefulness of outcome-only signal; report null |

Dynamic filtering evidence: [DAPO](https://arxiv.org/abs/2503.14476). A high zero-variance share is not itself a
runtime stop once Gate A passes; the central question is whether E5 converts that known waste into more advantage
mass per generated token than E1/E3/E4.

## 7. Training-time warning, stop, and success gates

Evaluate the frozen full-distribution dev canary every 50–100 updates.

Monitor: held-out execution success by stratum, grounding/validity, generated tokens, advantage mass, observed and
predicted `q`, posterior calibration, entropy, pass@1/pass@8, KL, clip fraction, gradient norm, response length,
invalid rate, policy-version lag, quota fulfillment, and peak memory.

Immediate stop:

- non-finite loss, gradient, or KL;
- token-ID/mask/logprob mismatch or failed weight synchronization;
- adversarial verifier-suite failure;
- any registered long-graph token quota missed on two consecutive windows;
- full-dev execution success drops by at least 5 points versus SFT on two consecutive evaluations;
- paired 95% CI is entirely below the registered non-inferiority margin;
- posterior calibration error exceeds the pre-registered bound on two evaluations.

Do **not** stop merely because uniform or current sampled groups have high zero variance; that may be the measured
opportunity. Stop when learnable strata disappear, quotas fail, the controller becomes miscalibrated, or held-out
capability regresses.

GRPO is accepted only if the registered full-held-out endpoint improves with paired uncertainty and grounding/
validity constraints remain non-inferior. H1 alone unlocks only the signal-efficiency claim.

## 8. Budget matching and experiment scope

Primary curves plot full-held-out execution success against cumulative generated rollout tokens. GPU-hours are a
secondary systems axis. Report episodes and optimizer updates; do not claim all four axes are simultaneously
matched. Starting checkpoint, trainable parameters, decoding, group size, target strata, and primary token budget
are controlled.

Three-seed core: E1 uniform, E3 static band, E4 post-filter, E5 online curriculum, E7 additional SFT. All other
ablations begin with one seed and are promoted only if they could reverse the conclusion.

## 9. Staged execution after sign-off

1. Freeze the reward truth table, optimizer derivation, strata/quotas, posterior schedule, kill test, and metrics.
2. Complete reward-hacking and split/leakage suites.
3. Run native-tool baseline and train-only n=32 measurement.
4. Apply Gates A/B/C exactly as written and sign GO/NO-GO or change the claim before training.
5. Verify framework token identity, placement, and one update.
6. Run the five pre-registered core experiments with automatic stops.
7. Run one-seed mechanism ablations only after the core comparison is interpretable.
8. Touch test once for the frozen final evaluation.

GPU-hour is measured from complete multi-turn episodes and real update steps, not guessed from token throughput.

## Appendix A. Optional abstention study—not on the core path

Abstention requires a different, outcome-conditioned reward and can collapse to universal refusal. It is deferred
until E1/E3/E4/E5/E7 are complete. It receives its own task schema, reward truth table, answerable coverage
constraint, A-Acc/A-FU/U-Ref metrics, stop gates, and experiment budget. Its shaped/multiclass reward must never be
mixed into the binary terminal labels used by the main curriculum study.

Useful starting evidence: [Abstain-R1](https://arxiv.org/html/2604.17073) and
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).
