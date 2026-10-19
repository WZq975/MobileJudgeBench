"""Binary classification metrics for judge predictions (positive class = successful trajectory)."""
from __future__ import annotations

from typing import Callable, Optional


def compute_classification_metrics(results: list[dict], filter_fn: Optional[Callable[[dict], bool]] = None) -> dict:
    """Accuracy, precision, recall, F1 and balanced accuracy of ``judge_result`` against
    ``annotation_success`` over the results that have both (optionally filtered)."""
    rows = [r for r in results if r.get("judge_result") is not None and r.get("annotation_success") is not None
            and (filter_fn is None or filter_fn(r))]
    tp = sum(1 for r in rows if r["judge_result"] == 1 and r["annotation_success"] == 1)
    tn = sum(1 for r in rows if r["judge_result"] == 0 and r["annotation_success"] == 0)
    fp = sum(1 for r in rows if r["judge_result"] == 1 and r["annotation_success"] == 0)
    fn = sum(1 for r in rows if r["judge_result"] == 0 and r["annotation_success"] == 1)
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if total and precision is not None and recall is not None else None),
        "balanced_accuracy": (recall + tnr) / 2 if recall is not None and tnr is not None else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total, "n_positive": tp + fn, "n_negative": tn + fp,
    }


def summarize(results: list[dict]) -> dict:
    """Overall, per-benchmark and per-agent metrics of a results list."""
    ok = [r for r in results if r.get("judge_result") is not None]
    out = {"evaluated": len(ok), "errors": sum(1 for r in results if r.get("error") is not None),
           "total_entries": len(results), "overall": compute_classification_metrics(ok),
           "by_benchmark": {}, "by_agent": {}}
    for b in sorted({r.get("benchmark") for r in ok if r.get("benchmark")}):
        out["by_benchmark"][b] = compute_classification_metrics(ok, lambda r, b=b: r.get("benchmark") == b)
    for a in sorted({r.get("agent") for r in ok if r.get("agent")}):
        out["by_agent"][a] = compute_classification_metrics(ok, lambda r, a=a: r.get("agent") == a)
    return out


def format_metrics_summary(m: dict, indent: int = 0) -> str:
    p = " " * indent
    f = lambda v: "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    return "\n".join([
        f"{p}Accuracy:          {f(m.get('accuracy'))}",
        f"{p}Precision:         {f(m.get('precision'))}",
        f"{p}Recall:            {f(m.get('recall'))}",
        f"{p}F1:                {f(m.get('f1'))}",
        f"{p}Balanced Accuracy: {f(m.get('balanced_accuracy'))}",
        f"{p}Confusion Matrix:  TP={m.get('tp', 0)}, TN={m.get('tn', 0)}, FP={m.get('fp', 0)}, FN={m.get('fn', 0)}",
        f"{p}Class Distribution: Positive(success)={m.get('n_positive', 0)}, Negative(failure)={m.get('n_negative', 0)}",
    ])


def print_summary(summary: dict) -> None:
    print(f"\nOverall ({summary['overall']['total']} trajectories with labels):")
    print(format_metrics_summary(summary["overall"], 2))
    for title, key in (("Benchmark", "by_benchmark"), ("Agent", "by_agent")):
        if summary.get(key):
            print(f"\nMetrics by {title}:")
            for name, m in summary[key].items():
                print(f"\n  [{name}] ({m['total']} trajectories):")
                print(format_metrics_summary(m, 4))
