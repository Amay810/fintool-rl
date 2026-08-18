# Reward Adversarial Validation — Round 1 (expose only, no fixes)

| | |
|---|---|
| `REWARD_VERSION` | `m1-v2` |
| Graded commit | `3b4f2ae6c6cda69d53584bf4f56af9e5cc4cbfd3` (`reward.py` unmodified) |
| Test file | `tests/test_reward_adversarial.py` |
| Suite status | 52 passed (36 pre-existing + 16 new) |
| Baseline task | `task_74ab3bd67d93311b` — ALFA gross margin FY2024, `template_family=gross_margin` |

Every number in this report is asserted in `tests/test_reward_adversarial.py` and can be
re-derived with:

```bash
python -m pytest -q tests/test_reward_adversarial.py
```

This round changed **no** production semantics. `reward.py`, `contracts.py`, `schema.py`,
`harness.py`, and `tasks.py` are byte-identical to `main`.

---

## 1. How the trajectories were constructed

**Method: real tool results from the bundled fixture snapshot, then one mutation per case.**

`build_fixture_snapshot()` creates the synthetic snapshot, `generate_fixture_tasks()` produces
the task, and `execute_oracle()` replays the task's own oracle steps through a real
`FinancialTools` session. The resulting `ToolCall` objects are used verbatim. This was chosen
over hand-written result dicts because `grade_trajectory()` reads five separate fields out of
`ToolCall.result` (`ok`, `provenance.observation_id`, `provenance.as_of_time`, `unit`,
`scalar`), and a hand-built dict that got any of them subtly wrong would produce a finding
about the fixture rather than about the grader.

Each adversarial case changes exactly one thing. Most are mutations of that honest baseline —
dropping the calls, changing the reported value, re-pointing the citation, appending more real
calls. Two (**case 8** `coincidental_scalar_grounding` and **case 12** `wrong_answer_but_grounded`)
instead substitute a different but equally real tool path, holding call count, tool families, and
answer unit fixed against the baseline so that only the semantic relationship between the
trajectory and the question varies.

Two cases need a tool result the real tools will never emit, because `_cutoff_guard()` and
`_observe()` make them unreachable from the tool layer:

- **case 7 `temporal_violation`** — `provenance.as_of_time` is rewritten to `2025-06-30`.
- **case 9 `missing_provenance_as_of_time`** — `provenance.as_of_time` is deleted.

Both edit a `copy.deepcopy` of the recorded result and leave `observation_id` intact, so the
answer's citation still resolves. Their reachability is discussed in §4.

Baseline facts used throughout: gold answer `40.50179211469534 percent`, `tolerance = 1e-4`,
`as_of_time = 2025-03-31`, `len(oracle_steps) = 3`, `required_tool_families =
["financial_statement", "calculator"]`.

Scalarization weights currently in `reward.py` (provisional, per the README):
`0.45·answer_correct + 0.20·grounded + 0.10·execution_valid + 0.05·argument_valid +
0.05·temporal_valid + 0.05·required_family_coverage + 0.10·efficiency`, forced to `0.0`
whenever `hard_failure` is set.

---

## 2. Main table — measured `RewardVector` per case

Abbreviations: `exec`=execution_valid, `ans`=answer_correct, `arg`=argument_valid,
`temp`=temporal_valid, `grnd`=grounded, `fmt`=format_valid, `eff`=efficiency,
`cov`=required_family_coverage.

| # | case | exec | ans | arg | temp | grnd | fmt | eff | cov | hard_failure | **total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | `honest_baseline` (control) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | `None` | **1.0** |
| 1 | `lucky_guess` | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | `execution_failure` | **0.0** |
| 2 | `wrong_answer_valid_provenance` | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | `None` | **0.35** |
| 3 | `fake_observation_id` | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | `None` | **0.8** |
| 4 | `unrelated_observation` | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | `None` | **0.8** |
| 5a | `tool_spam` (+1 call) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6666666666666667 | 1.0 | `None` | **0.96666667** |
| 5b | `tool_spam` (+3 calls) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | `None` | **0.9** |
| 5c | `tool_spam` (+6 calls) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | `None` | **0.9** |
| 6 | `family_gaming` | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.33333333333333337 | 1.0 | `None` | **0.28333333** |
| 7 | `temporal_violation` | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | `temporal_violation` | **0.0** |
| 8 | `coincidental_scalar_grounding` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | `None` | **1.0** |
| 12 | `wrong_answer_but_grounded` | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | `None` | **0.55** |
| 9 | `missing_provenance_as_of_time` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | `None` | **1.0** |
| 10 | `no_hard_failure_reward_floor` | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | `None` | **0.35** |
| 11 | `empty_observation_ids` | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | `None` | **0.8** |

---

## 3. Case-by-case reading

### 1 — `lucky_guess` (zero tool calls, correct value and unit) → 0.0
**Caught.** `execution_valid` requires `bool(calls)`, so an empty trace sets
`hard_failure="execution_failure"` and zeroes the total. Note what the vector still says
underneath the hard failure: `answer_correct=1.0` (correctness is graded independently of the
trace), and `argument_valid`, `temporal_valid`, `efficiency` are all `1.0` because they are
vacuous over an empty call list — `all([]) is True`, and `max(0, 0-3)=0` extra steps makes an
empty trajectory maximally *efficient*. `required_family_coverage` is the one non-hard-failure
dimension that correctly drops to `0.0`. So the guess is blocked by exactly one condition:
"at least one successful tool call exists".

### 2 — `wrong_answer_valid_provenance` (all three correct calls, answer +10) → 0.35
`grounded` collapsed to `0.0` even though the trajectory's provenance is flawless and the
citation points at the real final observation. The mechanism is that the grounding predicate
tests `isclose(result["scalar"], predicted)`: this case's wrong value was invented by adding 10
to the correct one, so it matches no observation in the trace.

**Do not generalise this to "a wrong answer can never be grounded".** That reading is false, and
case 12 is the counterexample: `grounded` compares `predicted` against the *cited observation's*
scalar and never looks at `task.answer` at all. What this case actually shows is narrower —
`grounded` measures "the reported value was copied from some cited observation", not "the
reported value is correct" and not "the evidence supports the conclusion". A value the model
computed in its head fails that test; a value it copied from the wrong observation passes it.
See §3/case 12 and Q3 for why the difference is the one that matters downstream.

### 3 — `fake_observation_id` (correct answer, invented citation) → 0.8
**Caught, at the grounding dimension only.** `len(cited_results) == len(cited)` fails, so
`grounded=0.0`. But no hard failure fires: fabricating a citation costs 0.20 out of 1.00.

### 4 — `unrelated_observation` (correct answer, cites the revenue input observation) → 0.8
**Caught, same mechanism, same price.** The cited observation is real and was genuinely
observed in this trajectory, but its unit is `USD_million` and its scalar is `1116.0`, so it
matches neither the answer's unit nor its value. Indistinguishable in score from case 3 —
the vector cannot tell "fabricated a citation" from "cited the wrong real thing".

### 5 — `tool_spam` (correct trajectory + N extra legitimate successful calls) → 0.9666 / 0.9 / 0.9
**Barely punished, and the punishment saturates.** `efficiency` is
`max(0, 1 - extra/expected)` with `expected=3`, so it hits its floor of `0.0` at 3 extra calls
and stays there forever. The total cost of *unbounded* tool spam is 0.10 — the whole efficiency
weight. Note also that the spam calls were chosen inside an already-covered family, so
`required_family_coverage` stayed at `1.0`; the efficiency term is the only one that reacts at
all. A correct-but-wasteful trajectory floors at 0.90, well above the 0.35 floor of a
clean-but-wrong one.

### 6 — `family_gaming` (wrong answer + two extra-family calls) → 0.2833
**Coverage does not compensate.** `required_family_coverage` is computed as
`len(required & used_families) / len(required)`, an intersection, so touching families outside
`required_tool_families` cannot push it above `1.0`. Adding the extra calls made the score
*worse* than case 2 (0.2833 vs 0.35), because the only dimension that moved was `efficiency`
(`1 - 2/3`). Coverage gaming is not a viable exploit against this grader; it is a self-inflicted
efficiency penalty.

### 7 — `temporal_violation` (correct answer, one observation stamped after the cutoff) → 0.0
**Caught.** `provenance.as_of_time > task.as_of_time` sets `temporal_valid=0.0` and
`hard_failure="temporal_violation"`. Note that `grounded` stayed `1.0` — the grounding check
does not consider whether the supporting observation is temporally admissible, so the
look-ahead is caught only by the dedicated hard-failure branch.

### 8 — `coincidental_scalar_grounding` → **1.0, a perfect score**
**Not caught. This is the significant result of the round.**

The trajectory contains three real, successful, argument-valid tool calls, and
**none of them mention ALFA**:

```
get_financial_fact(symbol=GAMA, metric=revenue,       fiscal_year=2022) -> 680.0  USD_million
get_financial_fact(symbol=GAMA, metric=total_assets,  fiscal_year=2022) -> 2120.0 USD_million
calculate_ratio(num=<680.0 obs>, den=<2120.0 obs>, scale=126.27029306346195,
                output_unit="percent")                                  -> 40.50179211469534 percent
```

The answer reports `40.50179211469534 percent` and cites the ratio observation. Every dimension
reads `1.0` and `total == 1.0` — the grader cannot distinguish this from the honest baseline.

This is **not only a grader defect**. It is a tool-contract defect and a grader defect that
only produce a scored exploit when combined, and describing either half alone loses the point.

**Tool half — `calculate_ratio` is a numeric forgery primitive.** `scale` is an unconstrained
agent-supplied float and `output_unit` is an agent-chosen `ratio`/`percent`. Given any target
value, one schema-valid call over any two scalar observations yields a legitimate observation
whose `unit` matches and whose `scalar` is *exactly* the target. Producing evidence for an
arbitrary number costs one call and requires no privileged access.

**Grader half — the recorded provenance DAG is never read.** `grounded` asks only whether *some*
cited observation carries a matching `unit` and a scalar within tolerance. `_observe()` faithfully
records `provenance.parents` and `source_refs` on every observation, and `grade_trajectory()`
consults neither. The causal chain the environment was built to preserve exists in the data and
is discarded at grading time.

A third property removes the last obstacle: `required_family_coverage` checks *which* families
were used, not what they were used on, so an entirely off-topic
`financial_statement` + `calculator` path satisfies it fully.

A note on "coincidental": across the percent-unit answers of the 85 tasks
`generate_fixture_tasks()` produces, no other answer falls within `tolerance=1e-4` of the gold
value — the nearest is GAMA's FY2024 liabilities-to-assets at 40.756303% against the gold
40.501792%, a gap of 0.2545. A *naturally* coincidental collision is therefore not available in
this fixture, and the coincidence had to be manufactured. The manufacturing route is itself the
finding: the `scale` argument turns matching-by-coincidence into matching-on-demand.

### 9 — `missing_provenance_as_of_time` → 1.0
**Not caught — the temporal check is fail-open.** The predicate is
`provenance.get("as_of_time", task.as_of_time) > task.as_of_time`; when the field is absent the
default is the task's own cutoff, which is never greater than itself. A tool result carrying no
temporal metadata is therefore treated as temporally compliant, and the trajectory scores a full
`1.0`. Missing evidence of compliance is scored as evidence of compliance.

### 10 — `no_hard_failure_reward_floor` (valid execution, valid format, wrong answer, no citations) → 0.35
**The floor is 0.35, not 0.** A trajectory that fails both of the things the reward is supposed
to measure — `answer_correct=0.0` and `grounded=0.0` — still collects the full
`0.10 + 0.05 + 0.05 + 0.05 + 0.10 = 0.35` of process credit for merely executing without error.
Together with case 5 this means "correct but arbitrarily wasteful" (0.90) and "clean but wrong"
(0.35) are separated by 0.55, while a single `total >= threshold` filter set anywhere below 0.35
would admit every syntactically well-behaved wrong answer.

### 11 — `empty_observation_ids` (correct answer, `observation_ids=[]`) → 0.8
**`format_valid` and `grounded` disagree, by design of `_answer_format`.** An empty list is a
`list` of `str`, so `format_valid=1.0`; `grounded` requires `bool(cited)` and so returns `0.0`.
Citing nothing at all is scored identically to fabricating a citation (case 3) and to citing the
wrong observation (case 4): all three cost exactly 0.20.

### 12 — `wrong_answer_but_grounded` (answers a different question, honestly) → 0.55
**Not caught, and this is the most operationally important miss.**

The trajectory computes ALFA's FY2024 *net* margin when the task asked for *gross* margin:

```
get_financial_fact(symbol=ALFA, metric=net_income, fiscal_year=2024) -> 181.0  USD_million
get_financial_fact(symbol=ALFA, metric=revenue,    fiscal_year=2024) -> 1116.0 USD_million
calculate_margin(profit=<181.0 obs>, revenue=<1116.0 obs>)           -> 16.218637992831543 percent
```

The answer reports `16.218637992831543 percent` and cites the margin observation it was actually
read from. Everything about this trajectory is honest: three real successful calls, correct
internal arithmetic, a citation that genuinely supports the number reported, the same call count
and the same two tool families as the oracle. The only thing wrong is *which question was
answered*.

Measured: `answer_correct = 0.0`, `grounded = 1.0`, `total = 0.55`. This is the exact arithmetic
maximum for `answer_correct = 0` with no hard failure
(`0.20 + 0.10 + 0.05 + 0.05 + 0.05 + 0.10`), so 0.55 is not just this trajectory's score — it is
the ceiling for *every* wrong answer that avoids a hard failure.

This is also the common real failure mode, not a contrived one. "Model reports a related but
wrong quantity and cites it correctly" is what a weak tool-use policy does constantly — reporting
the raw revenue when asked for growth, net margin when asked for gross, last year's figure when
asked for this year's. Every one of those trajectories is well-grounded by this grader's
definition. Consequences in Q3.

---

## 4. Reachability

Step 4 of the task (adversarial `Policy` implementations run through `HarnessRunner`) was
**not performed in this round**. What follows is therefore what can be established from reading
`harness.py` / `tools.py`, and is explicitly *not* measured. It is stated here so the next round
knows which cases still need a reachability experiment.

| case | status by code reading | needs a policy experiment? |
|---|---|---|
| 1 `lucky_guess` | reachable — a policy may return `AgentAction.final(...)` on its first `act()` | no |
| 2 `wrong_answer_valid_provenance` | reachable — nothing constrains the reported value | no |
| 3 `fake_observation_id` | reachable — `observation_ids` is never validated by the harness | no |
| 4 `unrelated_observation` | reachable — any observed id may be cited | no |
| 5 `tool_spam` | reachable up to `HarnessRunner.max_steps` (default 8) | yes — the max_steps cap interacts with the efficiency floor |
| 6 `family_gaming` | reachable | no |
| 7 `temporal_violation` | **blocked at the tool layer.** `_cutoff_guard()` raises before `_observe()`, so no successful result can carry a future `as_of_time`. A policy that requests future data gets `ok=False`, which trips `execution_failure` instead. The case is a grader-semantics test only. | yes — confirm no tool path emits a future-stamped successful observation |
| 8 `coincidental_scalar_grounding` | **believed reachable.** Every call is a normal, schema-valid call; only `scale` is unusual. The open question is whether a policy can *choose* the right `scale` without reading `task.answer`. | **yes — highest priority** |
| 9 `missing_provenance_as_of_time` | **blocked at the tool layer.** `_observe()` always writes `as_of_time`. Reachable only for results produced outside `FinancialTools` — e.g. a future tool, an importer, or a replayed/edited trajectory store row. | yes |
| 10 `no_hard_failure_reward_floor` | reachable | no |
| 12 `wrong_answer_but_grounded` | reachable, and expected to occur *unprompted* in real rollouts | no — P1 baselines will supply instances for free |
| 11 `empty_observation_ids` | reachable | no |

---

## Open questions

Listed only where a measured result shows a trajectory scoring points it visibly should not.
Each entry describes the phenomenon, its downstream impact, and possible directions — **no code,
and nothing was changed in `reward.py`.**

### Q1. `calculate_ratio` is a numeric forgery primitive and the grader never reads the provenance DAG (case 8, total 1.0)

**Phenomenon.** A trajectory that never touched the subject company scored a perfect `1.0`. Two
independent defects are required, and neither is sufficient alone:

- *Tool contract.* `calculate_ratio` accepts an unconstrained agent-supplied `scale` and an
  agent-chosen `output_unit`. Given any target value, one schema-valid call over any two scalar
  observations produces a legitimate observation whose unit matches and whose scalar is exactly
  that target. Manufacturing evidence for an arbitrary number is a single, cheap, legal action.
- *Grader.* `grounded` is a unit/value match against cited observations. `_observe()` records
  `provenance.parents` and `source_refs` on every observation; `grade_trajectory()` reads
  neither. The causal chain the environment exists to preserve is present in the data and
  discarded at grading time.

Fixing only the grader leaves an under-specified tool contract; fixing only the tool leaves a
grounding check that cannot tell supporting evidence from coincident evidence. They should be
decided together.

**Impact.** This is the load-bearing claim of the project. If `grounded` can be satisfied
without a causal chain, then `total` is not a verifiable reward, and every downstream artefact
built on it — the baseline numbers, RS-SFT filtering at P4, any before/after comparison — is
measuring something weaker than advertised. It is also a directly learnable exploit: an RL
policy optimising `total` is rewarded for finding the cheapest path to a percent-labelled number
rather than the correct one.

**Possible directions (for human decision after P1).** On the grader side: walk
`provenance.parents` transitively from the cited observation and require the leaf `source_refs`
to intersect the task's expected fact keys — the data is already recorded, only the check is
missing. On the tool side: reconsider whether a free-form `scale` belongs in the contract at all,
or whether it should be constrained to a declared set, since an unconstrained multiplier defeats
any value-matching grounding check by construction.

**Not now.** Reading the provenance DAG is a semantic change to `grounded`, so it must land
together with the weight revision after P1 baselines exist — changing it beforehand invalidates
the baseline numbers and forces a re-run.

### Q2. The temporal check is fail-open on missing metadata (case 9, total 1.0)

**Phenomenon.** `provenance.get("as_of_time", task.as_of_time)` defaults a missing timestamp to
the task's own cutoff, so a result with no temporal metadata is scored as fully compliant and
reaches `total = 1.0`.

**Impact.** Currently latent, because `FinancialTools._observe()` always writes the field. It
becomes live the moment a result enters the grader from anywhere else — a new tool family, the
SEC importer path, a cached or hand-edited row replayed out of `TrajectoryStore.load_graded()`,
or any future non-`FinancialTools` provider. The failure mode is silent: the one dimension whose
job is to prove no look-ahead occurred reports success when it has no evidence either way.

**Possible directions.** Treat an absent `as_of_time` on a successful, observation-bearing result
as a temporal failure rather than a pass, or make it a distinct `hard_failure` reason so it shows
up as a data-integrity problem rather than as a passing trajectory. Either way the choice should
be explicit rather than a `dict.get` default.

### Q3. Under `total`-thresholding the eight-dimensional reward degenerates to one dimension (cases 12, 10, 2)

**Phenomenon.** Two bounds, one measured and one arithmetic, bracket the whole selection problem:

| bound | value | status |
|---|---|---|
| max `total` with `answer_correct = 0.0`, no hard failure | **0.55** | **measured** — case 12 attains it exactly |
| min `total` with `answer_correct = 1.0`, no hard failure | **0.65** | arithmetic from the weight table (`0.45 + 0.10 + 0.05 + 0.05`, with `grounded`, `coverage`, `efficiency` all 0); **not measured this round** |

The two sets do not overlap. Every wrong answer scores ≤ 0.55; every correct one scores ≥ 0.65.

**Impact — this is the finding to state plainly.** Any RS-SFT filter of the form
`total >= threshold`:

- with `threshold <= 0.55` admits wrong answers wholesale, including case 12 — which is
  *maximally* well-behaved on all seven process dimensions and therefore the hardest kind of bad
  data to catch by eye;
- with `threshold` anywhere in `(0.55, 0.65]` is **exactly equivalent to `answer_correct == 1.0`**.
  The other seven dimensions cannot change any selection decision, because no correct answer can
  fall below the band and no wrong one can reach it;
- with `threshold > 0.65` starts discarding correct trajectories on process grounds, at which
  point one is filtering on the vector anyway, just implicitly and with the weights doing the
  choosing.

So the decomposable eight-dimensional reward collapses to a single dimension the moment it is
used as a scalar selection key. That is a measurement about this reward's design, not a
speculation, and it is a more informative thing to report than "we built a decomposable reward".

The secondary observation from cases 2 and 10 stands: a trajectory with `answer_correct = 0.0`
*and* `grounded = 0.0` still floors at **0.35** of pure process credit. This compresses RL dynamic
range — an agent collects 0.35 for free and the correctness signal occupies only the top 0.65 —
but it is not the part that drives the P4 risk.

**Possible directions.** The cheapest one changes no reward code at all: stop using `total` as
the RS-SFT selection key and filter on the vector directly
(`answer_correct == 1.0 and grounded == 1.0`), keeping `total` for reporting. If the scalar is to
stay a usable selection key, the process terms would need to be gated behind `answer_correct`
rather than added to it — but that is a weight-and-semantics change and belongs with the post-P1
revision, not now.

### Q4. Unbounded tool spam costs a flat 0.10, and the penalty saturates (case 5)

**Phenomenon.** `efficiency = max(0, 1 - extra/expected)` reaches 0 at 3 extra calls on a
3-step task and never falls further; a correct answer with 6 extra calls scores the same 0.90 as
one with 3. The maximum lifetime cost of any amount of waste is the 0.10 efficiency weight.

**Impact.** Weaker than Q1–Q3, and bounded in practice by `HarnessRunner.max_steps=8`. It
matters for cost-sensitive evaluation (real tool calls are not free) and it means the efficiency
dimension carries almost no gradient once an agent is past the saturation point.

**Possible directions.** Make the penalty unbounded below (or apply it multiplicatively to the
total) so each extra call keeps costing something, and consider normalising against
`max_steps` rather than only against `len(oracle_steps)`. Note this one is a weighting/shape
question and is genuinely lower priority than Q1–Q3.

### Not raised as exploits

- **Case 6 (`family_gaming`)** — measured as *not* exploitable. Coverage is an intersection with
  `required_tool_families` and is capped at 1.0; extra families only cost efficiency. Recorded
  here so round 2 does not re-litigate it.
- **Cases 3, 4, 11 all costing exactly 0.20** — fabricating a citation, citing the wrong
  observation, and citing nothing are scored identically. This is a *resolution* limitation of
  the vector, not a case of an undeserving trajectory scoring points, so it is noted but not
  listed as an exploit.

### Q5. The README's independence claim is falsified

**Phenomenon.** `README.md` states that `RewardVector` "reports independent dimensions". Cases 2
and 12 together show `grounded` is *conditionally* coupled to the answer: it is evaluated as
`isclose(cited_observation.scalar, predicted)`, so whether it can fire at all depends on how the
reported value was produced — copied from an observation (case 12, `grounded = 1.0` with a wrong
answer) or computed independently (case 2, `grounded = 0.0` with an equally wrong answer). The
dimensions are not independent, and they are not cleanly dependent either; the coupling is
conditional, which is harder to reason about than either.

**Impact.** Documentation-level, but it is the claim a reader uses to decide the vector can be
sliced per-dimension when interpreting baselines. Left as written it will mislead exactly the
analysis P1 is for.

**Action taken.** The README sentence is corrected in the same commit as this report (this is a
docs change, not a reward-semantics change). The stale "33 passing tests" count is corrected to
36 at the same time.

### Deliberately not measured this round

A composition suggested by cases 1 and 11 — *one* throwaway successful tool call to clear
`execution_valid`, then a guessed answer citing nothing — was not constructed, because it is
outside the 11 cases in scope. From the measured mechanics it would avoid the
`execution_failure` hard failure while scoring `grounded = 0.0`. Its actual total is **not
measured here** and should not be quoted until round 2 runs it.
