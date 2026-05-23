import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

FEATURE_COLS = [
    "delta_power", "theta_power", "alpha_power", "beta_power", "spindle_ratio", "eog_variance", "emg_variance", "signal_entropy"
]

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

def train_and_evaluate(df, test_size=0.4, random_state=42, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    X = df[FEATURE_COLS].values 
    y = df["true_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[ML Baseline] Training on {len(X_train)} epochs, "
          f"testing on {len(X_test)} epochs")

    clf = RandomForestClassifier(
        n_estimators=200,       
        max_depth=12,           
        random_state=random_state,
        n_jobs=-1,        
        class_weight="balanced" 
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    
    labels = sorted(df["true_label"].unique())
    target_names = [STAGE_NAMES[l] for l in labels]

    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True
    )
    per_class_f1 = {
        STAGE_NAMES[l]: report[STAGE_NAMES[l]]["f1-score"]
        for l in labels
    }
    macro_f1 = f1_score(y_test, y_pred, average="macro",
                        labels=labels, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "ml_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(
        [FEATURE_COLS[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue"
    )
    ax2.set_xlabel("Importance")
    ax2.set_title("Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "ml_feature_importance.png")
    plt.savefig(fi_path, dpi=120)
    plt.close()

    print(f" Accuracy:  {acc:.3f}")
    print(f" Macro F1:  {macro_f1:.3f}")
    print(f" Per-class F1: { {k: round(v,3) for k,v in per_class_f1.items()} }")
    print(f" Confusion matrix saved to {cm_path}")

    n_train = len(X_train)
    result_df = df.reset_index(drop=True).iloc[n_train:].copy()
    result_df = result_df.reset_index(drop=True)
    result_df["ml_pred"] = y_pred
    result_df["ml_pred_name"] = [STAGE_NAMES.get(p, "?") for p in y_pred]
    
    pred_path = os.path.join(output_dir, "ml_predictions.csv")
    result_df.to_csv(pred_path, index=False)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(os.path.join(output_dir, "ml_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return clf, metrics, result_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.feature_extractor import load_and_extract
    df = load_and_extract(data_dir="data")
    clf, metrics, result_df = train_and_evaluate(df, output_dir="outputs")
    print("Done.")