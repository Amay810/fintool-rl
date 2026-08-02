# SEC real-data validation — 2026-08-01 (rebuild after fiscal-year fix)

## Scope

- Source: SEC Company Facts API and the SEC official ticker/exchange mapping.
- Universe: 15 companies across technology, consumer, financials, healthcare, energy, industrials, and communication services.
- Company split: 9 train / 3 dev / 3 test; companies do not cross splits.
- Information cutoff: 2025-03-31.
- Canonical concepts: revenue, gross profit, net income, total assets, and total liabilities where reported.
- Raw JSON and generated runtime artifacts are retained locally and ignored by Git.
- Request contact information is not persisted in the repository or snapshot.

## Frozen snapshot

| Item | Value |
|---|---:|
| Companies | 15 |
| Annual financial facts | 1,015 |
| Daily prices | 0 |
| Market-index rows | 0 |
| Snapshot SHA-256 | `191e3b0005ab3fb0a0a6f09908372832ffca82d8d25e1a2ffc2ff9ca53ac9927` |
| Independent rebuild hash stable | yes |

### Fiscal-year labeling (post-fix)

The previous importer preferred `frame=CYxxxx` for fiscal-year labels. That mislabeled
non-calendar filers when later 10-K filings restated comparative periods (WMT/NVDA),
creating duplicate `(symbol, metric, period_end)` rows under two fiscal years.

Current rules:

1. Aggregate inside each taxonomy tag by `period_end`, keeping the latest filed revision.
2. Across tags, preserve `METRIC_TAGS` priority; fallback tags only fill uncovered period ends.
3. Drop quarterly noise: reject `CYxxxxQn` frames and duration facts shorter than 300 days.
4. Label fiscal year as `int(period_end[:4])`, except 52/53-week closes in the first week of
   January (day ≤ 7), which map to the prior calendar year (preferring an exact `CYxxxx` frame).
5. Fail closed if one metric still has two different `period_end` values for the same fiscal year.
6. Snapshot integrity asserts uniqueness of `(symbol, metric, period_end)` and
   `(symbol, metric, fiscal_year)`.

The filing-level `fy` field is still not used for labeling, because comparative facts carry the
filing's fiscal year rather than the period's fiscal year.

Coverage varies by industry and filer taxonomy. Banks do not normally expose gross profit, and some
filers do not publish a standalone `Liabilities` concept. Missing concepts are not converted to zero.
One historical BAC `Assets` value equal to zero remains excluded as a ratio denominator.

The initial universe used XOM, but the current SEC ticker file resolves it to a post-cutoff successor
CIK. It was replaced with COP to prevent future entity-mapping leakage.

## Generated task set

| Split | Companies | Tasks |
|---|---:|---:|
| train | 9 | 500 |
| dev | 3 | 100 |
| test | 3 | 200 |

Template distribution:

| Template | Tasks |
|---|---:|
| Financial fact lookup | 285 |
| Year-over-year growth | 282 |
| Gross margin | 74 |
| Liabilities-to-assets | 159 |

Answer units: `USD_million` for lookups; `percent` for growth, gross margin, and
liabilities-to-assets (`calculate_ratio` now takes explicit `output_unit`).

YoY generation requires adjacent fiscal years (`current == previous + 1`). Non-adjacent gaps are
skipped rather than labeled as year-over-year.

All 800 oracle trajectories completed with reward 1.0 and no hard failure. This validates the
rebuilt snapshot, tools, oracles, provenance, and reward integration; it is not a model baseline.

## Engineering checks

- 33 automated tests pass.
- Baseline harness records `invalid_action` / `model_call_error` without aborting the batch;
  reward hard failures respect `terminal_reason` so format errors are not collapsed into
  `execution_failure`.
- Downloads validate JSON, retry transient failures with exponential backoff, and atomically replace files.
- Snapshot reconstruction is deterministic at the byte hash level.
- Generated task metadata identifies `sec_snapshot_v1`.
- Fact-key leakage audits fail closed if a source fact appears in multiple splits.

## Remaining M1 work

- Run frozen baselines for a small local model, a medium local model, and a strong API model.
  Follow [BASELINE_RUNBOOK.md](BASELINE_RUNBOOK.md); label failures with [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md).
- Label baseline failures using the taxonomy in `FAILURE_TAXONOMY.md`.
- Add held-out-tool/compositional protocols; the current benchmark primarily measures held-out-company generalization.
- Add temporal challenge tasks with per-task cutoffs and post-cutoff facts retained in the snapshot.
- Manually audit a stratified sample of normalized XBRL facts and generated questions.
- Add a separately reported human-reviewed challenge set adapted from suitable licensed sources.
- Price/index tools remain implemented but are not populated by this SEC-only snapshot.
