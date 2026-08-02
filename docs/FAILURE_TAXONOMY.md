# Failure taxonomy — M1 baseline

Taxonomy version: `m1-baseline-v1`

Each trajectory receives exactly one **primary** label and zero or more **secondary** tags.
Labels are assigned deterministically from the reward vector, terminal reason, and tool errors.
They are intended for baseline diagnosis and reward-hacking audits, not as training targets.

## Primary labels

| Label | Meaning |
|---|---|
| `success` | No hard failure, answer correct, total reward 1.0 |
| `model_call_error` | HTTP/timeout/API payload failure before a valid action |
| `invalid_answer_format` | Unparseable action JSON or malformed final answer object |
| `invalid_arguments` | Tool arguments failed schema validation |
| `temporal_violation` | Requested data after `as_of_time` or after snapshot cutoff |
| `no_tool_use` | Final answer with zero tool calls |
| `max_steps` | Hit step budget without answering |
| `tool_lookup_miss` | Tool executed but fact/period/price was unavailable |
| `execution_failure` | Other non-ok tool results |
| `wrong_unit` | Well-formed answer with incorrect unit |
| `correct_ungrounded` | Numeric answer matches but citations do not ground it |
| `wrong_value` | Well-formed answer with incorrect value |
| `soft_failure` | Residual non-perfect score without a hard failure |

## Secondary tags

| Tag | Meaning |
|---|---|
| `unit_mismatch` | Attached when primary is `wrong_unit` |
| `missing_required_families` | Required diagnostic families were not covered |
| `inefficient` | More tool calls than the oracle path |
| `ungrounded` | Soft failure with grounded=0 |

## How to use in baseline review

1. Run or re-analyze a trajectory store.
2. Inspect `failure_taxonomy.primary` counts before looking at answer accuracy alone.
3. Open `failure_examples` for 1–3 concrete traces per non-success label.
4. Freeze the taxonomy version in the baseline report. Do not silently retune labels after
   comparing models.

`as_of_time` is currently uniform across the SEC task set, so `temporal_violation` may be rare
until temporal challenge tasks are added.
