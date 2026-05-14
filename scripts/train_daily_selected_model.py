"""Train the selected daily model from the best current probe.

This keeps the main target as daily T -> T+1 (`oil_return_fwd1 > 0`) and uses
the compact feature set that outperformed the all-feature extended pipeline:

  step5b_v2 CLUSTER_POS_10 + top K extended market/regime features

Outputs:
  - selected_daily_model.joblib
  - selected_daily_model_metrics.csv
  - selected_daily_model_features.csv
  - selected_daily_model_predictions.csv
  - selected_daily_model_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "ml"))
sys.path.insert(0, str(BASE_DIR / "ml" / "classification"))

DEFAULT_DATASET = BASE_DIR / "data" / "processed" / "dataset_final_noleak_ext_market_regime_v1.csv"
DEFAULT_PRICE_SOURCE = BASE_DIR / "data" / "processed" / "dataset_step4_noleak.csv"
DEFAULT_SELECTED = BASE_DIR / "ml" / "classification" / "results_step5b_v2" / "step5_selected_features.csv"
DEFAULT_EXT_FEATURES = BASE_DIR / "data" / "processed" / "dataset_final_noleak_ext_market_regime_v1_features.csv"
DEFAULT_OUT_DIR = BASE_DIR / "ml" / "classification" / "results_daily_selected_model_v1"

TARGET = "oil_return_fwd1"
TARGET_DATE_COL = "oil_return_fwd1_date"
SPLIT_DATE = pd.Timestamp("2023-01-01")
RANDOM_STATE = 42


def import_add_technical_features(price_source: Path):
    os.environ["CLASSIFICATION_PRICE_SOURCE_PATH"] = str(price_source)
    os.environ.setdefault("CLASSIFICATION_OUT_DIR", str(DEFAULT_OUT_DIR))
    from step3_technical_improve import add_technical_features

    return add_technical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--price-source", default=str(DEFAULT_PRICE_SOURCE))
    parser.add_argument("--selected-features", default=str(DEFAULT_SELECTED))
    parser.add_argument("--ext-features", default=str(DEFAULT_EXT_FEATURES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-ext-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    add_technical_features = import_add_technical_features(Path(args.price_source))
    df = pd.read_csv(args.dataset, parse_dates=["date", TARGET_DATE_COL]).sort_values("date").reset_index(drop=True)
    df = add_technical_features(df)

    base_features = pd.read_csv(args.selected_features)["feature"].tolist()
    ext_features = pd.read_csv(args.ext_features)["feature"].tolist()
    ext_features = [f for f in ext_features if f in df.columns]

    train_mask = df[TARGET_DATE_COL] < SPLIT_DATE
    test_mask = ~train_mask
    y = (df[TARGET] > 0).astype(int)

    mi = mutual_info_classif(
        df.loc[train_mask, ext_features].fillna(0),
        y.loc[train_mask],
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )
    ext_rank = pd.DataFrame({"feature": ext_features, "mutual_info": mi}).sort_values(
        "mutual_info", ascending=False
    )
    chosen_ext = ext_rank["feature"].head(args.top_ext_k).tolist()
    features = [f for f in base_features + chosen_ext if f in df.columns]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    n_jobs=4,
                ),
            ),
        ]
    )

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "target": TARGET,
        "target_rule": "oil_return_fwd1 > 0",
        "split_date": str(SPLIT_DATE.date()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "top_ext_k": int(args.top_ext_k),
        "n_features": int(len(features)),
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "auc": roc_auc_score(y_test, prob),
        "pred_up_rate": float(np.mean(pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    predictions = df.loc[test_mask, ["date", TARGET_DATE_COL, TARGET]].copy()
    predictions["target_class"] = y_test.values
    predictions["pred_class"] = pred
    predictions["pred_prob_up"] = prob

    pd.DataFrame([metrics]).to_csv(out_dir / "selected_daily_model_metrics.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out_dir / "selected_daily_model_features.csv", index=False)
    ext_rank.to_csv(out_dir / "selected_daily_model_ext_feature_ranking.csv", index=False)
    predictions.to_csv(out_dir / "selected_daily_model_predictions.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "metrics": metrics,
            "dataset": str(Path(args.dataset)),
            "price_source": str(Path(args.price_source)),
            "base_features": base_features,
            "chosen_extended_features": chosen_ext,
        },
        out_dir / "selected_daily_model.joblib",
    )

    summary = [
        "# Selected Daily Model v1",
        "",
        "Target: daily `oil_return_fwd1 > 0`.",
        "",
        f"- Model: `LGBMClassifier`",
        f"- Features: `{len(features)}`",
        f"- Added extended features: `{', '.join(chosen_ext)}`",
        f"- Test Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Test F1_macro: `{metrics['f1_macro']:.4f}`",
        f"- Test AUC: `{metrics['auc']:.4f}`",
        f"- Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}",
        "",
        "Weekly is not part of this model.",
        "",
    ]
    (out_dir / "selected_daily_model_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved selected model artifacts: {out_dir}")


if __name__ == "__main__":
    main()

