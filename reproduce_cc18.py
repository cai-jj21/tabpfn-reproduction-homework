from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

try:
    import torch
except Exception:
    torch = None


CC18_DATASETS = {
    "balance-scale": 11,
    "mfeat-fourier": 14,
    "breast-w": 15,
    "mfeat-karhunen": 16,
    "mfeat-morphological": 18,
    "mfeat-zernike": 22,
    "credit-g": 31,
    "diabetes": 37,
    "tic-tac-toe": 50,
    "vehicle": 54,
    "eucalyptus": 188,
    "pc4": 1049,
    "pc3": 1050,
    "kc2": 1063,
    "pc1": 1068,
    "wdbc": 1510,
    "car": 40975,
    "steel-plates-fault": 40982,
}

PAPER_AUC = {
    "balance-scale": 0.9973,
    "mfeat-fourier": 0.9811,
    "breast-w": 0.9934,
    "mfeat-karhunen": 0.9978,
    "mfeat-morphological": 0.9669,
    "mfeat-zernike": 0.9823,
    "credit-g": 0.7894,
    "diabetes": 0.8410,
    "tic-tac-toe": 0.9759,
    "vehicle": 0.9589,
    "eucalyptus": 0.9245,
    "pc4": 0.9383,
    "pc3": 0.8373,
    "kc2": 0.8346,
    "pc1": 0.8761,
    "wdbc": 0.9964,
    "car": 0.9950,
    "steel-plates-fault": 0.9655,
}

CSV_COLUMNS = [
    "Dataset",
    "Mean_ROC_AUC",
    "Std_ROC_AUC",
    "Mean_Accuracy",
    "Std_Accuracy",
    "Mean_Time_s",
    "Paper_ROC_AUC",
    "Diff",
]


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    with output_path.open("r", encoding="utf-8", newline="") as f:
        return {row["Dataset"] for row in csv.DictReader(f)}


def load_openml_dataset(dataset_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    x_df = data.data.copy()
    y_raw = pd.Series(data.target)

    for col in x_df.columns:
        if x_df[col].dtype == "category" or x_df[col].dtype == "object":
            x_df[col] = LabelEncoder().fit_transform(x_df[col].astype(str))
        else:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

    valid_mask = ~x_df.isna().any(axis=1) & ~y_raw.isna()
    x = x_df.loc[valid_mask].to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(y_raw.loc[valid_mask].astype(str))

    if len(y) == 0:
        raise RuntimeError("empty dataset after filtering missing values")

    return x, y


def evaluate_dataset(
    dataset_id: int,
    device: str,
    splits: int,
    auc_multiclass: str,
) -> dict[str, float]:
    x, y = load_openml_dataset(dataset_id)
    aucs: list[float] = []
    accuracies: list[float] = []
    times: list[float] = []

    for seed in range(splits):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.5,
            random_state=seed,
            stratify=y,
        )

        model = TabPFNClassifier(device=device)
        start = time.time()
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)
        elapsed = time.time() - start

        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class=auc_multiclass)

        aucs.append(float(auc))
        accuracies.append(float(accuracy_score(y_test, np.argmax(proba, axis=1))))
        times.append(float(elapsed))

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mean_roc_auc": float(np.mean(aucs)),
        "std_roc_auc": float(np.std(aucs)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_time": float(np.mean(times)),
    }


def append_result(output_path: Path, dataset_name: str, result: dict[str, float]) -> None:
    is_new_file = not output_path.exists()
    paper_auc = PAPER_AUC[dataset_name]
    diff = result["mean_roc_auc"] - paper_auc

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(
            [
                dataset_name,
                f"{result['mean_roc_auc']:.4f}",
                f"{result['std_roc_auc']:.4f}",
                f"{result['mean_accuracy']:.4f}",
                f"{result['std_accuracy']:.4f}",
                f"{result['mean_time']:.4f}",
                f"{paper_auc:.4f}",
                f"{diff:+.4f}",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce TabPFN on 18 OpenML-CC18 datasets.")
    parser.add_argument("--device", default=default_device(), help="TabPFN device, e.g. cuda or cpu.")
    parser.add_argument("--splits", type=int, default=5, help="Number of stratified 50/50 splits.")
    parser.add_argument(
        "--auc-multiclass",
        choices=["ovo", "ovr"],
        default="ovo",
        help="Multiclass ROC AUC mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tabpfn_cc18_safe_corrected_ovo_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the output CSV before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    done = load_done(args.output)
    print(f"device={args.device}, splits={args.splits}, done={len(done)}")

    for index, (dataset_name, dataset_id) in enumerate(CC18_DATASETS.items(), start=1):
        if dataset_name in done:
            print(f"[{index:02d}/18] {dataset_name:<25} skip")
            continue

        print(f"[{index:02d}/18] {dataset_name:<25} running")
        try:
            result = evaluate_dataset(
                dataset_id=dataset_id,
                device=args.device,
                splits=args.splits,
                auc_multiclass=args.auc_multiclass,
            )
        except Exception as exc:
            print(f"[{index:02d}/18] {dataset_name:<25} failed: {exc}")
            continue

        append_result(args.output, dataset_name, result)
        print(f"[{index:02d}/18] {dataset_name:<25} auc={result['mean_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
