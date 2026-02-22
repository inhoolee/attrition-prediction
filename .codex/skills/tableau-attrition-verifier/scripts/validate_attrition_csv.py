#!/usr/bin/env python3
"""Profile attrition CSV and reproduce common Tableau KPI checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _exists_cols(df: pd.DataFrame, cols: Iterable[str]) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for col in cols:
        if col in df.columns:
            present.append(col)
        else:
            missing.append(col)
    return present, missing


def _decode_department(row: pd.Series) -> str:
    if row.get("Department_Sales", 0) == 1:
        return "Sales"
    if row.get("Department_Research & Development", 0) == 1:
        return "Research & Development"
    return "Human Resources"


def _decode_jobrole(row: pd.Series, jobrole_cols: list[str], baseline: str) -> str:
    for col in jobrole_cols:
        if row.get(col, 0) == 1:
            return col.replace("JobRole_", "")
    return baseline


def _tenure_band(years: int) -> str:
    if years <= 1:
        return "0-1y"
    if years <= 2:
        return "1-2y"
    if years <= 5:
        return "3-5y"
    if years <= 10:
        return "6-10y"
    return "10y+"


def _print_df(df: pd.DataFrame) -> None:
    """Print table without optional tabulate dependency."""
    print("```text")
    print(df.to_string())
    print("```")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate attrition CSV schema and KPI reproducibility for Tableau."
    )
    parser.add_argument("--csv", required=True, help="Path to attrition CSV.")
    parser.add_argument("--spec", help="Optional spec markdown path.")
    parser.add_argument(
        "--jobrole-baseline",
        default="Healthcare Representative",
        help="Fallback role when all JobRole_* one-hot values are 0.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows, cols = df.shape

    print("# Gate 1 - Data Profile")
    print(f"- CSV: `{csv_path}`")
    if args.spec:
        spec_path = Path(args.spec).expanduser().resolve()
        print(f"- Spec: `{spec_path}` ({'exists' if spec_path.exists() else 'missing'})")
    print(f"- Shape: `{rows} rows x {cols} columns`")

    key_cols = [
        "Attrition",
        "YearsAtCompany",
        "Age",
        "MonthlyIncome",
        "OverTime",
        "Department_Sales",
        "Department_Research & Development",
    ]
    present, missing = _exists_cols(df, key_cols)
    print(f"- Key columns present: `{len(present)}`")
    print(f"- Key columns missing: `{', '.join(missing) if missing else 'none'}`")

    for binary_col in ["Attrition", "OverTime"]:
        if binary_col in df.columns:
            uniq = sorted(df[binary_col].dropna().unique().tolist())
            print(f"- `{binary_col}` unique values: `{uniq}`")

    print("")
    print("# Gate 2 - Field Mapping Hints")
    print("- `Attrition Rate`: direct (`Attrition`)")
    print("- `Attrition Count`: direct (`Attrition` sum when encoded 0/1)")
    print("- `Tenure Band`: derived from `YearsAtCompany`")
    print("- `Department Label`: derived from department one-hot columns")
    print("- `JobRole Label`: derived from `JobRole_*` + baseline fallback")

    if "Attrition" in df.columns:
        print("")
        print("# Gate 3 - KPI Reproducibility")
        headcount = len(df)
        attr_count = int(df["Attrition"].sum()) if pd.api.types.is_numeric_dtype(df["Attrition"]) else None
        if attr_count is not None:
            attr_rate = attr_count / headcount if headcount else float("nan")
            print(f"- Headcount: `{headcount}`")
            print(f"- Attrition Count: `{attr_count}`")
            print(f"- Attrition Rate: `{attr_rate:.4%}`")

        if "YearsAtCompany" in df.columns and attr_count is not None:
            early = df[df["YearsAtCompany"] <= 2]
            early_den = len(early)
            early_num = int(early["Attrition"].sum())
            early_rate = (early_num / early_den) if early_den else float("nan")
            print(f"- Early Attrition (<=2y): `{early_num}/{early_den} = {early_rate:.4%}`")

        if "YearsAtCompany" in df.columns and attr_count is not None:
            tenure = df.copy()
            tenure["Tenure Band"] = tenure["YearsAtCompany"].apply(_tenure_band)
            grouped = tenure.groupby("Tenure Band", dropna=False).agg(
                head=("Attrition", "size"),
                attr=("Attrition", "sum"),
            )
            grouped["rate"] = grouped["attr"] / grouped["head"]
            ordered = ["0-1y", "1-2y", "3-5y", "6-10y", "10y+"]
            print("")
            print("## Tenure Band Stats")
            _print_df(grouped.reindex(ordered))

        dept_cols_needed = {"Department_Sales", "Department_Research & Development"}
        if dept_cols_needed.issubset(df.columns):
            dept = df.copy()
            dept["Department Label"] = dept.apply(_decode_department, axis=1)
            dept_g = dept.groupby("Department Label", dropna=False).agg(
                head=("Attrition", "size"),
                attr=("Attrition", "sum"),
            )
            dept_g["rate"] = dept_g["attr"] / dept_g["head"]
            print("")
            print("## Department Stats")
            _print_df(dept_g.sort_values("rate", ascending=False))

        jobrole_cols = sorted([c for c in df.columns if c.startswith("JobRole_")])
        if jobrole_cols:
            role = df.copy()
            role["JobRole Label"] = role.apply(
                _decode_jobrole,
                axis=1,
                jobrole_cols=jobrole_cols,
                baseline=args.jobrole_baseline,
            )
            role_g = role.groupby("JobRole Label", dropna=False).agg(
                head=("Attrition", "size"),
                attr=("Attrition", "sum"),
            )
            role_g["rate"] = role_g["attr"] / role_g["head"]
            print("")
            print("## JobRole Stats")
            _print_df(role_g.sort_values("rate", ascending=False))

    print("")
    print("# Gate 4 - Ready Check")
    print("- If all required fields are direct/derived and KPIs are reproduced, proceed to Tableau steps.")
    print("- If any required field is missing, stop and request additional source data.")


if __name__ == "__main__":
    main()
