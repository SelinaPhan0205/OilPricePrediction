"""Summarize daily classification training outputs.

Reads the standard result CSVs written by ml/classification steps and writes a
compact Markdown + CSV leaderboard in the same output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULT_FILES = [
    "step1_test_results.csv",
    "step2_test_results.csv",
    "step5_results.csv",
]


def normalize_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["source_file"] = path.name
    if "Model" not in df.columns and "model" in df.columns:
        df["Model"] = df["model"]
    if "Set" not in df.columns:
        df["Set"] = ""
    aliases = {
        "Accuracy": ["Accuracy", "accuracy", "Test_Accuracy"],
        "F1_macro": ["F1_macro", "f1_macro", "F1m", "Test_F1_macro", "Test_F1m"],
        "AUC": ["AUC", "auc", "Test_AUC"],
    }
    for col, candidates in aliases.items():
        if col not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[col] = df[candidate]
                    break
            else:
                df[col] = pd.NA
    keep = ["source_file", "Set", "Model", "Accuracy", "F1_macro", "AUC"]
    return df[[c for c in keep if c in df.columns]]


def format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def build_markdown(out_dir: Path, leaderboard: pd.DataFrame) -> str:
    lines = [
        "# Daily Training Summary",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "## Best Results",
        "",
    ]

    if leaderboard.empty:
        lines.append("No result CSVs were found.")
        return "\n".join(lines) + "\n"

    best_acc = leaderboard.sort_values(["Accuracy", "F1_macro", "AUC"], ascending=False).iloc[0]
    best_auc = leaderboard.sort_values(["AUC", "Accuracy", "F1_macro"], ascending=False).iloc[0]

    lines.extend(
        [
            (
                "- Best Accuracy: "
                f"`{format_float(best_acc['Accuracy'])}` from `{best_acc['source_file']}` "
                f"model=`{best_acc.get('Model', '')}` set=`{best_acc.get('Set', '')}`"
            ),
            (
                "- Best AUC: "
                f"`{format_float(best_auc['AUC'])}` from `{best_auc['source_file']}` "
                f"model=`{best_auc.get('Model', '')}` set=`{best_auc.get('Set', '')}`"
            ),
            "",
            "## Leaderboard",
            "",
            "| Source | Set | Model | Accuracy | F1_macro | AUC |",
            "|---|---|---|---:|---:|---:|",
        ]
    )

    top = leaderboard.sort_values(["Accuracy", "F1_macro", "AUC"], ascending=False).head(20)
    for row in top.itertuples(index=False):
        lines.append(
            "| "
            f"{row.source_file} | {getattr(row, 'Set', '')} | {getattr(row, 'Model', '')} | "
            f"{format_float(row.Accuracy)} | {format_float(row.F1_macro)} | {format_float(row.AUC)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Target remains daily `oil_return_fwd1 > 0`.",
            "- Weekly is not part of this run.",
            "- Treat large jumps as leakage-audit candidates before using them as final claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    frames = []
    for name in RESULT_FILES:
        path = out_dir / name
        if path.exists():
            frames.append(normalize_results(path))

    if frames:
        leaderboard = pd.concat(frames, ignore_index=True)
        leaderboard = leaderboard.dropna(subset=["Accuracy", "AUC"], how="all")
    else:
        leaderboard = pd.DataFrame(columns=["source_file", "Set", "Model", "Accuracy", "F1_macro", "AUC"])

    leaderboard_path = out_dir / "daily_training_leaderboard.csv"
    summary_path = out_dir / "DAILY_TRAINING_SUMMARY.md"
    leaderboard.to_csv(leaderboard_path, index=False)
    summary_path.write_text(build_markdown(out_dir, leaderboard), encoding="utf-8")

    print(f"Saved leaderboard: {leaderboard_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
