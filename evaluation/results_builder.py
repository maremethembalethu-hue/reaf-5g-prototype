# Computes the standard evaluation metrics: confusion matrix, accuracy,
# per-class precision/recall/F1, detection rate, false positive rate, false
# negative rate

import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DEFAULT_RESULTS = "joined_results.csv"
DEFAULT_OUT = "evaluation_report.json"


def load_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate(path=DEFAULT_RESULTS, out_path=DEFAULT_OUT, benign_label="Benign"):
    rows = load_results(path)
    if not rows:
        print("No joined results to evaluate.")
        return None

    y_true = [r["expected_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    labels = sorted(set(y_true))

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    # Binary attack-vs-benign framing for detection rate / FPR / FNR
    tp = sum(1 for t, p in zip(y_true, y_pred) if t != benign_label and p == t)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t != benign_label and p != t)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == benign_label and p != benign_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == benign_label and p == benign_label)

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (tp + fn) if (tp + fn) > 0 else None

    result = {
        "n_flows": len(rows),
        "accuracy": acc,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
        "detection_rate": detection_rate,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = {k: v for k, v in result.items() if k not in ("confusion_matrix", "per_class_report")}
    print(json.dumps(summary, indent=2))
    print(f"Full report (incl. confusion matrix, per-class metrics) written to {out_path}")
    return result



if __name__ == "__main__":
    evaluate(DEFAULT_RESULTS, DEFAULT_OUT, "Benign")