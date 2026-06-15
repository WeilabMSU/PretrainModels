#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


SMI_SYMBOLS = (
    r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|"
    r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"
)
SMI_REGEX = re.compile(SMI_SYMBOLS)

TARGETS = [
    "c", "C", "(", ")", "1", "O", "=", "2", "N", "n",
    "3", "[", "]", "@", "H", "F", "-", "4", "S", "Cl",
    "/", "s", "o", ".", "Br", "5", "+", "#", "\\", "6",
    "I", "P", "Si", "7", "8", "B", "%", "Na", "9", "0",
    "K", "Sn", "Se", "Li", "Zn", "Al", "b", "As", "Mg", "p",
    "Ca", "se", "Ag", "Te", "Ba", "Bi", "Rb", "Cs", "Sr", "te",
    "Be", "length", "symbol_type",
]

MODEL_LABELS = {
    0: ("chembl27_512", "model_chembl27"),
    1: ("chembl27_pubchem_512", "model_chembl27_pubchem"),
    2: ("chembl27_pubchem_zinc_512", "model_chembl27_pubchem_zinc"),
}

PRETRAIN_PREFIXES = ["c", "cp", "cpz"]


def tokenize_smiles(smiles: str) -> List[str]:
    return SMI_REGEX.findall(smiles.strip())


def iter_smiles(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def empty_distribution(target: str) -> Dict[float, int]:
    if target == "length":
        return {k: 0 for k in range(1, 257)}
    if target == "symbol_type":
        return {k: 0 for k in range(1, 62)}
    return {float(np.round(k, 2)): 0 for k in np.arange(0.0, 1.001, 0.01)}


def bucket_value(target: str, tokens: List[str]):
    if target == "length":
        return min(max(len(tokens), 1), 256)
    if target == "symbol_type":
        return min(max(len(set(tokens)), 1), 61)
    if not tokens:
        return 0.0
    return float(np.round(tokens.count(target) / len(tokens), 2))


def dataset_distribution(smiles_path: str, target: str) -> Dict[float, int]:
    distribution = empty_distribution(target)
    for smiles in iter_smiles(smiles_path):
        key = bucket_value(target, tokenize_smiles(smiles))
        if key not in distribution:
            key = float(np.round(key, 2))
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def wasserstein_from_counts(left: Dict, right: Dict) -> float:
    left_keys = np.sort(list(left.keys()))
    right_keys = np.sort(list(right.keys()))
    left_values = np.array([left[k] for k in left_keys], dtype=np.float64)
    right_values = np.array([right[k] for k in right_keys], dtype=np.float64)

    left_max = max(float(left_values.max()), 1.0)
    right_max = max(float(right_values.max()), 1.0)
    left_values = 1e-15 + left_values / left_max
    right_values = 1e-15 + right_values / right_max

    if len(left_values) == len(right_values):
        distance = np.mean(np.abs(np.sort(left_values) - np.sort(right_values)))
    else:
        distance = empirical_wasserstein(left_values, right_values)
    return float(np.round(distance, 4))


def empirical_wasserstein(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_values = np.sort(left_values)
    right_values = np.sort(right_values)
    all_values = np.concatenate([left_values, right_values])
    all_values.sort(kind="mergesort")
    deltas = np.diff(all_values)
    if len(deltas) == 0:
        return 0.0
    left_cdf = np.searchsorted(left_values, all_values[:-1], side="right") / len(left_values)
    right_cdf = np.searchsorted(right_values, all_values[:-1], side="right") / len(right_values)
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def build_wasserstein_features(smiles_path: str, pretrain_distribution_path: Path) -> np.ndarray:
    pretrained = np.load(pretrain_distribution_path, allow_pickle=True).item()
    target_distributions = [
        dataset_distribution(smiles_path, target)
        for target in TARGETS
    ]

    features: List[float] = []
    for prefix in PRETRAIN_PREFIXES:
        for idx, target_distribution in enumerate(target_distributions):
            features.append(
                wasserstein_from_counts(
                    target_distribution,
                    pretrained[f"{prefix}-{idx}"],
                )
            )
    return np.asarray(features, dtype=np.float64)


def load_training_table(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    names: List[str] = []
    labels: List[int] = []
    rows: List[List[float]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        feature_columns = [str(i) for i in range(189)]
        for row in reader:
            names.append(row.get("", ""))
            labels.append(int(row["label"]))
            rows.append([float(row[column]) for column in feature_columns])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), names


def standardize_all(training: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_features = np.vstack([training, query[None, :]])
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (all_features - mean) / std
    return normalized[:-1], normalized[-1]


def pca_fit_transform(training: np.ndarray, query: np.ndarray, n_components: int):
    mean = training.mean(axis=0)
    centered = training - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T, (query - mean) @ components.T


def ridge_scores(training: np.ndarray, labels: np.ndarray, query: np.ndarray, alpha: float = 1.0):
    classes = np.unique(labels)
    targets = -np.ones((len(labels), len(classes)), dtype=np.float64)
    for class_idx, class_label in enumerate(classes):
        targets[labels == class_label, class_idx] = 1.0

    design = np.column_stack([np.ones(len(training)), training])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    train_scores = design @ weights
    train_pred = classes[np.argmax(train_scores, axis=1)]

    query_design = np.concatenate([[1.0], query])
    query_scores = query_design @ weights
    return classes, query_scores, float(np.mean(train_pred == labels))


def select_model(smiles_path: str, data_dir: Path):
    training, labels, _ = load_training_table(data_dir / "wasserstein.csv")
    query = build_wasserstein_features(
        smiles_path,
        data_dir / "PretrainedSetsDistribution.npy",
    )
    training, query = standardize_all(training, query)

    best = None
    max_components = min(training.shape)
    for n_components in range(1, max_components + 1):
        reduced_training, reduced_query = pca_fit_transform(training, query, n_components)
        classes, scores, train_accuracy = ridge_scores(reduced_training, labels, reduced_query)
        result = (train_accuracy, n_components, classes, scores)
        if best is None or train_accuracy > best[0]:
            best = result
        if train_accuracy == 1.0:
            best = result
            break

    train_accuracy, n_components, classes, scores = best
    score_by_label = {int(label): float(score) for label, score in zip(classes, scores)}
    selected_label = int(classes[np.argmax(scores)])
    folder, model_name = MODEL_LABELS[selected_label]
    return {
        "selected_label": selected_label,
        "selected_model": model_name,
        "selected_folder": folder,
        "scores": {
            MODEL_LABELS[int(label)][0]: float(score_by_label[int(label)])
            for label in classes
        },
        "training_accuracy": float(train_accuracy),
        "pca_components": int(n_components),
    }


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description="Select a pretrained BT-FP model for a SMILES dataset.")
    parser.add_argument("--input-dataset", "--input_dataset", required=True, help="Input .smi file.")
    parser.add_argument("--data-dir", default=str(default_data), help="Folder with model-selection data files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_model(args.input_dataset, Path(args.data_dir))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Selected pretrained model: {result['selected_model']}")
    print(f"Model folder: {result['selected_folder']}")
    print(f"Training-set selector accuracy: {result['training_accuracy']:.3f}")
    print(f"PCA components used: {result['pca_components']}")
    print("Scores:")
    for folder, score in result["scores"].items():
        print(f"  {folder}: {score:.6f}")


if __name__ == "__main__":
    main()

