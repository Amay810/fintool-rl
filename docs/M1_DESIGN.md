# M1 design contract

## Objective

Build a deterministic financial tool environment in which every factual answer can be recomputed from a
versioned local snapshot and every trajectory can be regraded without contacting an external service.

## Data contract

Each snapshot must expose:

- `snapshot_id`;
- maximum information cutoff;
- schema version;
- source class and upstream provenance;
- SHA-256 digest;
- table-level row counts.

Tools open the database in SQLite read-only/query-only mode.  Import pipelines may write a new snapshot but a
training/evaluation process never mutates one in place.

## Task construction

Programmatic tasks are generated from database facts and executable oracle templates.  Adapted external
questions retain source-dataset attribution and license metadata.  Manually reviewed challenge tasks remain a
separate split.

Required fields:

```json
{
  "task_id": "...",
  "question": "...",
  "split": "train|dev|test|challenge",
  "as_of_time": "YYYY-MM-DD",
  "difficulty": "single_tool|multi_tool|compositional|held_out_tool",
  "template_family": "...",
  "answer": {"value": 0.0, "unit": "...", "tolerance": 0.0},
  "oracle_steps": [],
  "required_tool_families": [],
  "metadata": {"fact_keys": []}
}
```

The policy-facing view excludes `answer`, `oracle_steps`, `required_tool_families`, `metadata`, and split.

## Split policy

The synthetic smoke set is company-disjoint.  The real M1 dataset will publish three evaluations:

1. IID task generalization;
2. new compositions of seen tools;
3. held-out tools whose public schema is supplied only at evaluation time.

Underlying source fact IDs, company, period, and template-family fingerprints are audited to prevent accidental
cross-split duplication.  Held-out-tool evaluation must not be conflated with unseen-schema evaluation.

## Reward policy

Hard failures:

- invalid arguments;
- data requested after `as_of_time`;
- failed/missing tool execution;
- malformed final answer.

Soft dimensions are stored separately.  Scalarization is versioned and cannot be changed in place after an
experiment is frozen.  Required tool paths are diagnostic unless the task semantics uniquely require one.

## Acceptance gates

Before model training begins:

- all oracle programs execute successfully;
- repeated tool calls produce byte-equivalent semantic results and identical observation IDs;
- snapshot manifest validates;
- fact-leakage audit passes;
- temporal-violation tests pass;
- every reward component has positive and negative unit tests;
- baseline failure taxonomy is frozen;
- the small-model baseline has enough non-zero successes for learning and enough headroom for improvement.

The final baseline difficulty gate will be selected from observed distributions rather than asserted in
advance.

## Immediate real-data work

1. ~~Implement SEC Company Facts/XBRL importer into the existing schema.~~
2. Add frozen daily-price source with explicit licensing and adjustment policy.
3. ~~Select 12–20 tools based on task coverage, not raw tool count.~~
4. ~~Generate the first real task pool and perform source-fact leakage checks.~~
5. Manually audit a stratified challenge sample before running Qwen baselines.
6. Run small / medium / strong-model frozen baselines and freeze the failure taxonomy.
7. Add temporal challenge tasks with per-task cutoffs while retaining post-cutoff facts.

