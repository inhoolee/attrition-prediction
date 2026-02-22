# Tableau Verification Workflow

Use this checklist when handling Tableau dashboard requests.

## Gate 1 - Data Profile

- Confirm source file path exists.
- Report row count and column count.
- Report required key columns and missing columns.
- Report encoding for binary fields (e.g., `0/1` vs `Yes/No`).

Pass condition:
- Source file is readable.
- Required columns for requested charts are either `direct` or `derived`.

## Gate 2 - Field Mapping

For each requested chart/KPI/filter, map required fields:

| Requirement | Source Type | Source |
|-------------|-------------|--------|
| Attrition Rate | direct | `Attrition` |
| Department | derived | decode from one-hot |
| JobRole | derived | decode from one-hot + baseline |

Rules:
- Mark as `missing` when no deterministic derivation exists.
- Do not proceed to build steps until every required field is resolved.

## Gate 3 - KPI Reproducibility

- Compute KPI values from source with explicit formulas.
- Show numerator, denominator, and final percent/count.
- Flag any mismatch with spec values.

## Gate 4 - Tableau Build Steps

- Provide sheet-by-sheet instructions:
  - Columns shelf
  - Rows shelf
  - Marks type
  - Label/Color/Sort settings
  - Filter application scope (`All using this data source` vs `Selected worksheets`)
- Include filter conflict handling (for example, exclude `Tenure Band` from tenure distribution sheet).

## Output Contract

Always present:

1. Verified facts (with file path basis)
2. Conflicts/mismatches
3. Next action and explicit confirmation checkpoint
