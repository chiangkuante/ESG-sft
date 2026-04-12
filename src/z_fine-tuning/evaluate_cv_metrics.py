import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

LABELS = [
    "Climate Change",
    "Natural Capital",
    "Pollution & Waste",
    "Human Capital",
    "Product Liability",
    "Community Relations",
    "Corporate Governance",
    "Business Ethics & Values",
    "Non-ESG"
]
INVALID_LABEL = "__INVALID__"

def calc_fold_metrics(y_true, y_pred):
    y_true_clean = [t if t in LABELS else INVALID_LABEL for t in y_true]
    y_pred_clean = [p if p in LABELS else INVALID_LABEL for p in y_pred]

    acc = accuracy_score(y_true_clean, y_pred_clean)
    kappa = cohen_kappa_score(y_true_clean, y_pred_clean, labels=LABELS)
    
    # Macro & Micro
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=LABELS, average="macro", zero_division=0
    )
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=LABELS, average="micro", zero_division=0
    )
    
    # Per class
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=LABELS, average=None, zero_division=0
    )
    
    metrics = {
        "Accuracy": acc,
        "Kappa": kappa,
        "Macro P": p_mac,
        "Macro R": r_mac,
        "Macro F1": f1_mac,
        "Micro F1": f1_mic,
    }
    
    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "Precision": p_cls[i],
            "Recall": r_cls[i],
            "F1": f1_cls[i]
        }
        
    return metrics, per_class

def format_mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

def main():
    cv_dir = Path("results/cv")
    output_dir = Path("results/evaluate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not cv_dir.exists():
        logger.error(f"Directory {cv_dir} does not exist.")
        return
        
    all_overall_stats = []
    all_class_stats = []
    
    for model_path in sorted(cv_dir.iterdir()):
        if not model_path.is_dir():
            continue
            
        model_name = model_path.name
        
        fold_metrics = []
        fold_class_metrics = {label: {"Precision": [], "Recall": [], "F1": []} for label in LABELS}
        
        matched_files = list(model_path.glob("fold_*.json"))
        if not matched_files:
            logger.info(f"Skipping empty directory: {model_name}")
            continue
            
        logger.info(f"Processing model: {model_name} with {len(matched_files)} fold files.")
        
        for fold_file in matched_files:
            with open(fold_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            y_true = []
            y_pred = []
            for item in data:
                y_true.append(item.get("ground_truth_label"))
                y_pred.append(item.get("parsed_label"))
                
            metrics, per_class = calc_fold_metrics(y_true, y_pred)
            fold_metrics.append(metrics)
            
            for label in LABELS:
                fold_class_metrics[label]["Precision"].append(per_class[label]["Precision"])
                fold_class_metrics[label]["Recall"].append(per_class[label]["Recall"])
                fold_class_metrics[label]["F1"].append(per_class[label]["F1"])
                
        num_folds = len(fold_metrics)
        aggregated = {"Model": model_name, "Validation Folds": num_folds}
        
        for key in ["Accuracy", "Macro F1", "Micro F1", "Kappa"]:
            vals = [m[key] for m in fold_metrics]
            aggregated[key] = format_mean_std(vals)
            
        all_overall_stats.append(aggregated)
        
        for label in LABELS:
            row = {"Model": model_name, "Category": label}
            for metric in ["Precision", "Recall", "F1"]:
                vals = fold_class_metrics[label][metric]
                row[metric] = format_mean_std(vals)
            all_class_stats.append(row)
            
    if not all_overall_stats:
        logger.warning("No metrics found. Make sure JSON files exist in results/cv/<model>/.")
        return
        
    df_overall = pd.DataFrame(all_overall_stats)
    df_class = pd.DataFrame(all_class_stats)
    
    df_overall.to_csv(output_dir / "cv_overall_metrics.csv", index=False)
    df_overall.to_markdown(output_dir / "cv_overall_metrics.md", index=False)
    
    df_class.to_csv(output_dir / "cv_per_class_metrics.csv", index=False)
    df_class.to_markdown(output_dir / "cv_per_class_metrics.md", index=False)
    
    print("\n=== Overall Metrics (Mean ± Std) ===")
    print(df_overall.to_markdown(index=False))
    
    print("\n=== Per-class Metrics (Mean ± Std) ===")
    print(df_class.to_markdown(index=False))
    
    logger.info(f"Saved evaluation reports to {output_dir}/")

if __name__ == "__main__":
    main()
