# FinTool-RL project plan

Single-sentence scope:

> Train a Qwen3-4B tool-use policy on executable financial graphs and test whether a
> policy-version-aware, token-stratified curriculum increases DrGRPO advantage mass and
> held-out success per generated rollout token.

This document is the planning source of truth. `M1_DESIGN.md` remains the environment/data
contract; `BASELINE_RUNBOOK.md` remains the operational procedure.

The project thesis, falsifiable hypotheses, data layers, experiment matrix, and interview narrative are
defined in [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md). The pre-experiment GRPO proof obligations and GO/NO-GO
rules are defined in [`RL_READINESS_PROTOCOL.md`](RL_READINESS_PROTOCOL.md). Milestones in this file may not
override those two protocols.

## 1. Positioning

### What this project is

An **Agentic RL algorithm** project. The deliverable is the training pipeline and the
experimental evidence, not a financial product.

### What this project is not

Explicitly out of scope, because the e-commerce agent project already covers it:

general agent runtime, memory, MCP, HTTP/SSE serving, identity/authorization, write
operations, human handoff, RAG, online feedback loops, a large frontend, generic failure
store, multi-size model comparison.

### Division of labour against the e-commerce agent project

| E-commerce agent | FinTool-RL |
|---|---|
| How to build a trustworthy production agent | How to train a tool-use policy with verifiable rewards |
| Multi-turn user dialogue, stateful writes | Short-horizon read-only numeric reasoning |
| Guardrails at runtime | Verifier as reward |
| Domain / failure-driven LoRA | RS-SFT → uniform DrGRPO → online graph curriculum |
| External eval harness | advantage-mass/token mechanism study and causal ablations |

The e-commerce project keeps `RL gate not passed` in its README. That statement is the stated
motivation for this project and should not be edited away.

The failure taxonomy here is **an input to RL data selection**, not a standalone evaluation
product. That framing is what keeps the two projects from looking like one framework applied
to two datasets.

## 2. Model

Primary and only trained model: **`Qwen/Qwen3-4B-Instruct-2507`**, with the exact revision frozen in every
experiment manifest.

No size comparison experiments. The comparison axis is the training stage:

```text
M0  Qwen/Qwen3-4B-Instruct-2507, native tools, prompt only
M1  + RS-SFT (LoRA)
M2  + uniform and static-band DrGRPO
M3  + discounted, stratified online graph curriculum
```

Qwen3-1.7B is a fallback only if 4B GRPO proves infeasible on 4×A100 40GB. It is not an
experiment.

## 3. Current state

Done:

- 12 typed read-only tools over an immutable SEC XBRL snapshot (15 companies, 1,015 annual facts)
- calculators consume `observation_id`, never raw numbers
- 8-dimension decomposable reward with hard gates
- 800 oracle-backed tasks, exact 500/100/200 company-disjoint splits, oracle smoke 800/800
- trajectory harness, replay, append-only store
- failure taxonomy (13 primary labels) and frozen baseline reporting
- one exploratory baseline: DeepSeek-chat, dev, n=20

Not done: any trained checkpoint, pass@k analysis, RL data selection, GRPO.

Accurate resume claim today: *built the training environment*. Not: *completed a financial
agent post-training system*.

### Two defects found in the dev20 trajectories (blocking)

**D1 — the metric vocabulary is not exposed in the tool schema.**
`get_financial_fact` / `list_available_periods` describe `metric` as "canonical metric such as
revenue or net_income" without enumerating legal values. Observed attempts:

| metric | calls | ok |
|---|---:|---|
| `revenue` | 14 | yes |
| `net_income` | 12 | yes |
| `Liabilities` | 10 | no |
| `liabilities` | 2 | no |
| `Assets` | 2 | no |
| `total_liabilities` | 1 | yes |
| `total_assets` | 1 | yes |

The model succeeded exactly on the two metrics named in the docstring and guessed raw XBRL tag
names for the rest. The 20% success on `liabilities_to_assets` therefore measures an
environment affordance gap, not compositional reasoning.

**D2 — `efficiency` conflates "explored one extra step" with "wrong".**
All four `soft_failure` cases had `answer_correct=1.0` and `grounded=1.0`, and lost points only
because they called `list_available_periods` before fetching facts:

```text
[list_available_periods, get_financial_fact, get_financial_fact, calculate_growth]
4 calls vs oracle 3 → efficiency 0.667 → total 0.967 → not "success"
```

`success_rate=0.55` decomposes into 11 true successes, 4 correct-but-exploring, 3 format
failures, 2 D1-contaminated. Real answer capability was 15/20.

D1 and D2 are the same root cause twice: the model must probe because the enum is hidden, then
gets penalised for probing. Both must be fixed before any frozen baseline, otherwise the
baseline measures the environment.

## 4. Milestones

Each experiment states purpose, output, what it decides, and whether it is resume-bearing.

### M1.-1 Contract and label repair

| Field | Value |
|---|---|
| Purpose | Stop measuring environment defects as model capability |
| Work | enumerate `metric` in tool schema and prompt; drop `efficiency` from the success definition (`success = no hard failure ∧ answer_correct ∧ grounded`); make `efficiency` a budgeted diagnostic (oracle steps + exploration allowance); add `correct_but_inefficient` label; bump reward to `m1-v3` |
| Output | updated schema/reward/taxonomy; dev20 report archived as `v2`, marked non-comparable |
| Decides | nothing — precondition |
| Resume | no |

### M1.1 Native task difficulty expansion — highest priority

| Field | Value |
|---|---|
| Purpose | Create genuine capability gradient and RL headroom; defeat "the model just memorised four templates" |
| Work | 4–7 step families: gross-margin YoY change, net-margin vs gross-margin gap, growth-of-growth, equity ratio (`assets − liabilities` then ratio), cross-company comparison. Observation reuse (one `revenue` observation feeding two calculators). **Discovery tasks** ("most recent disclosed fiscal year") that force `list_available_periods` before the fact call. Distractor entities/metrics in the question text. Per-task `as_of_time` with post-cutoff facts retained in the snapshot |
| Output | expanded task set, template × step-count distribution, oracle smoke |
| Decides | whether the task distribution can support RL at all |
| Resume | yes — the benchmark itself |

Discovery tasks matter twice: they are the hardest to memorise, and they repair the semantics
of `efficiency` — exploration becomes required rather than wasteful.

### M1.0 Frozen Qwen3-4B baseline

| Field | Value |
|---|---|
| Purpose | Establish M0 and the failure profile that defines the SFT target |
| Work | test 200 (+ dev 100), report per template × step count × failure family |
| Output | frozen report, failure table, worked examples |
| Decides | SFT data mix; whether difficulty is sufficient |
| Resume | yes |

Gate: judge per (template × step count × failure family), never on the aggregate. A saturated
`financial_fact_lookup` and a 20% 6-step family must not be averaged.

### M2 RS-SFT

| Field | Value |
|---|---|
| Purpose | Fix format and tool-convention failures; establish the SFT capability ceiling |
| Work | sample with the strong API model + base 4B, filter by verifier, LoRA SFT. Core mix: verified success trajectories and hard-task trajectories. Abstention trajectories belong only to the optional follow-up |
| Output | checkpoint, learning curve, SFT badcase set |
| Decides | what remains for RL |
| Resume | yes |

### M2.5 RL readiness gate — highest analytical value

| Field | Value |
|---|---|
| Purpose | Instantiate the pre-written learnability, curriculum-opportunity, and controller-validity tests |
| Work | sample n=32 per eligible train task under the future RL rollout policy using the pure terminal outcome; record trajectories, generated tokens, graph cell, policy version, pass@1/pass@8, and successes; estimate token-weighted uniform degeneracy; run the ICC variance decomposition and held-out controller-calibration kill test; use G=8 as the initial group size |
| Output | learnability/opportunity GO-NO-GO, stratum token quotas, cell-only or hierarchical controller decision, and frozen routing report |
| Decides | whether RL is learnable, whether the curriculum has identifiable upside, and which controller is statistically defensible |
| Resume | yes — this is the differentiator |

Interpretation: n=32 with 0 or 32 successes requires a second 32-sample confirmation before routing
away; it does not prove inability or saturation. Two gates are separate: some `0<p<1` cells are needed
for RL learnability, while high token-weighted all-fail/all-pass mass under uniform sampling is the
curriculum opportunity. If all cells cluster around `p=0.5`, RL can be ready while the curriculum has
little upside. If within-cell instances are not exchangeable, the pre-specified partially pooled
Beta-Binomial controller replaces the cell-only controller.

### M3 DrGRPO baselines

| Field | Value |
|---|---|
| Purpose | Establish the uniform-waste and sample-efficiency baselines before proposing a curriculum |
| Work | M3A uniform DrGRPO on the eligible train distribution, then M3B DrGRPO on the frozen M2.5 band; use pure binary terminal reward and identical graph-stratum generated-token quotas; log absolute advantage mass, all generated tokens, graph cell, KL, gradients, and full held-out metrics |
| Output | uniform-vs-static-band success-over-generated-token curves, advantage-mass analysis, paired SFT/RL comparison |
| Decides | whether model-conditional selection helps and whether an online curriculum is justified |
| Resume | yes |

### M3.5 Online variance-aware graph curriculum

| Field | Value |
|---|---|
| Purpose | Increase DrGRPO advantage mass per generated rollout token as the policy's learnable band moves |
| Work | after each policy weight sync, discount Beta sufficient statistics with `gamma in [0.90,0.98]`, decreasing gamma as sync KL grows; acquire by posterior expected DrGRPO advantage mass per token only within graph-depth strata; enforce minimum generated-token quotas per stratum; generate fresh oracle-backed tasks; compare uniform, frozen band, and post-generation filtering; ablate `q(p)` acquisition, posterior forgetting, hierarchical pooling, shaping, and clipped distribution correction |
| Output | advantage-mass/token and success-over-token curves plus held-out graph generalization |
| Decides | end of the main line |
| Resume | yes |

### M4 FinQA challenge adapter

| Field | Value |
|---|---|
| Purpose | Structural transfer, not a data merge |
| Work | independent adapter and split; do not mix into the 800 native tasks; report separately with licence and attribution |
| Output | transfer result |
| Decides | optional |
| Resume | optional |

### M5 Abstention — optional follow-up after the core RL study

| Field | Value |
|---|---|
| Purpose | Separate safety/calibration study; not a prerequisite for E1/E3/E4/E5/E7 |
| Work | `TaskSpec.expected_outcome = answer \| abstain`; outcome-conditioned reward; unanswerable generator; oracle/replay support; hallucination and refusal-collapse test suites |
| Output | abstention split, Abstention F1, hallucinated-number rate |
| Decides | whether abstention needs dedicated SFT data |
| Resume | optional; never mixed into the core binary-reward claim |

This study must distinguish *answerable but the model looked in the wrong place* from *not answerable
at this cutoff*. It uses its own reward, A-Acc/A-FU/U-Ref metrics, coverage constraint, and stop gates.

DPO is not on the main line. It returns only if rollouts naturally produce many
same-task, rankable, near-miss pairs.

## 5. Metrics

Primary mechanism metric: total absolute DrGRPO advantage mass divided by **all generated rollout
tokens**. Primary capability curve: full held-out execution success versus cumulative generated
rollout tokens. GPU-hours are a secondary systems axis; episodes and optimizer updates are reported,
not simultaneously matched.

Also report, sliced by graph stratum: non-degenerate-group rate, answer/grounded/tool validity,
pass@1/pass@8, posterior calibration, ICC, policy-version lag, quota fulfillment, tokens, and wall-clock
cost. Abstention F1 and hallucinated-number behavior belong to the optional abstention appendix.

Reporting rules: fixed data version, fixed decoding parameters, paired task-level comparison
between stages, bootstrap confidence intervals on the primary metric, test split never enters
badcase augmentation.

## 6. NSCC execution

### Node responsibilities

Login node: `qsub` / `qstat` / `qdel`, receiving code and data, light file inspection.
Never vLLM, LoRA training, large rollouts, or CUDA JIT.

Compute node: base vLLM, LoRA-merged vLLM, FlashInfer/CUDA JIT, single-node 4-GPU LoRA
training and rollout.

### Fixed PBS resource form

```bash
#PBS -q normal
#PBS -P personal
#PBS -l select=1:ncpus=16:ngpus=4:mem=110gb
```

`normal` is the only queue users submit to. Names such as `g1` shown by `qstat` are internal
execution queues and must not appear in `#PBS -q`.

Never write `select=4:ngpus=1` — that may span four nodes.

### vLLM

```text
data_parallel_size = 4
tensor_parallel_size = 1
port = 8123
```

The evaluation client runs inside the same job against `http://127.0.0.1:8123/v1`.

Template: `nscc/baseline.pbs`.

### Code and data sync

Code moves by GitHub commit/pull (`origin` = `Amay810/fintool-rl`). The Windows drive mapping
points at the same NSCC repository and is for viewing/editing only; it is not a transport for
large artifacts. Snapshots and generated task files are rebuilt on the cluster from the frozen
raw SEC JSON rather than copied over SSHFS.

### Working style

No defensive scaffolding: no repeated verification passes, no re-hashing artifacts that are
already manifested, no test runs beyond what a change requires. Every job must have a stated
purpose and a named output file before it is submitted.

## 7. Expected gains and honest risk

No guarantee is claimed. The risk is not uniform across stages.

**High confidence — base → RS-SFT improves environment-specific convention adherence.** Even with a fair
native function-calling baseline, the canonical metric vocabulary, observation-ID-only calculators,
provenance-carrying final answers, discovery behavior, and second-order observation reuse are local
contracts. Format validity, tool selection, reference validity, and enum adherence are exactly what
supervised imitation is expected to repair. The claim must still be tested on held-out entities and graphs.

**Conditional — SFT → DrGRPO improves.** The project separates two necessary conditions before
training. First, the learnability gate requires successful and failed rollouts in every graph stratum
needed for the claim. Second, the opportunity gate requires substantial token-weighted all-fail/all-pass
waste under uniform sampling; high uniform zero variance is the curriculum's opportunity, not by itself
a defect. If the first fails, expand graph difficulty/support. If the second fails because cells cluster
near `p=0.5`, uniform RL may work but the curriculum has no defensible 30–50% efficiency upside.

The mainline uses pure binary terminal success. It does not manufacture variance with shaping or switch
the claim to abstention after seeing results. M2.5 also tests whether graph cells are exchangeable; a low-
ICC or poorly calibrated cell model triggers the pre-specified hierarchical Beta-Binomial controller.

**Interpretation if RL does not help.** A null result is mapped to a pre-written falsification:
no support rejects readiness; no uniform waste rejects curriculum opportunity; higher advantage mass
without better learning rejects H2; training-slice gain without full-held-out gain identifies objective
shift or overfitting. This makes failure scientifically interpretable, but it is not used as a substitute
for designing conditions under which the main hypothesis can actually be tested.

This is why M1.1 precedes everything: with only four templates and 1- or 3-step trajectories,
pass@k, SFT, and GRPO would all be precisely measuring template memorisation.
