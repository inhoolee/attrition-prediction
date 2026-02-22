---
name: tableau-attrition-verifier
description: Validate Tableau dashboard requests for this attrition project against local processed data before giving build steps. Use when creating or reviewing Tableau tabs/charts/KPIs/filters derived from attrition_clean.csv and related project outputs.
---

# Tableau Attrition Verifier

Use this skill to prevent unverified Tableau guidance.

## Core Rules

1. Verify data first, then provide dashboard instructions.
2. Back every field/metric claim with local command output.
3. Show verification in gates and pause for user confirmation between major gates.
4. If spec and data conflict, state conflict explicitly and propose exact fix options.

## Workflow

1. Identify target tab and spec file.
2. Profile source CSV with `scripts/validate_attrition_csv.py`.
3. Build a required-field map:
   - `direct`: field exists as-is
   - `derived`: field must be calculated/decoded
   - `missing`: cannot build without another source
4. Reproduce KPI numbers before describing chart builds.
5. Ask for confirmation, then provide Tableau build steps (shelf-level).
6. Validate filter scope and sheet-application rules.
7. End with unresolved risks and missing dependencies.

Use `references/workflow.md` when you need the full checklist and output format.

## Commands

Run from the skill directory:

```bash
python3 scripts/validate_attrition_csv.py --csv /path/to/attrition_clean.csv
python3 scripts/validate_attrition_csv.py --csv /path/to/attrition_clean.csv --spec /path/to/spec.md
```

## Expected Response Pattern

1. `Gate 1 - Data profile`: row count, schema, encoded fields
2. `Gate 2 - Field mapping`: chart requirements vs source fields
3. `Gate 3 - KPI reproducibility`: exact numerator/denominator and values
4. `Gate 4 - Tableau build`: sheet-by-sheet creation instructions
