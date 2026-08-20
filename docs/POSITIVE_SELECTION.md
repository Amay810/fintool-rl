# Positive trajectory selection

`positive-v1` is a training-data selection contract, separate from the reward contract
and strict trajectory success. A trajectory is selected only when all of these hold:

The selection contract is not the reward contract and is not strict trajectory success.

```text
hard_failure is None
answer_correct == 1
grounded == 1
required_family_coverage == 1
execution_valid == 1
argument_valid == 1
temporal_valid == 1
format_valid == 1
```

`efficiency` and `total == 1` are intentionally excluded. Efficiency remains an
evaluation/quality dimension, but is not a correctness gate for imitation-data selection.
The selector records `selected`, `selection_version`, and `failed_conditions` so a
positive decision is auditable.

All exploration and positive trajectory collection for training must use the train split
only. Dev remains evaluation/diagnostic data. Test remains untouched.
