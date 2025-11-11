#!/usr/bin/env python3
"""Analyze similarity between embedding sets via orthogonal Procrustes."""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA


def load_embeddings(path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    df = pd.read_parquet(path)
    if "prompt_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"{path} must contain 'prompt_id' and 'embedding' columns.")

    df = df.dropna(subset=["prompt_id"])
    mapping: Dict[str, np.ndarray] = {}
    for pid, emb in zip(df["prompt_id"], df["embedding"]):
        vec = np.asarray(emb, dtype=np.float64)
        mapping[str(pid)] = vec
    label = path.stem
    return mapping, label


def stack_common(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids = sorted(set(a.keys()).intersection(b.keys()))
    if not ids:
        raise ValueError("No shared prompt IDs between two embedding sets.")
    xa = np.vstack([a[i] for i in ids])
    xb = np.vstack([b[i] for i in ids])
    return xa, xb, ids


def center_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def maybe_reduce(x: np.ndarray, y: np.ndarray, pca_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if pca_dim and pca_dim < x.shape[1]:
        concat = np.vstack([x, y])
        pca = PCA(n_components=pca_dim, random_state=42)
        reduced = pca.fit_transform(concat)
        return reduced[: x.shape[0]], reduced[x.shape[0] :]
    return x, y


def procrustes_disparity(x: np.ndarray, y: np.ndarray, pca_dim: int) -> float:
    xr, yr = maybe_reduce(x, y, pca_dim)
    xc = center_rows(xr)
    yc = center_rows(yr)
    r, _ = orthogonal_procrustes(yc, xc)
    y_aligned = yc @ r
    numerator = np.linalg.norm(xc - y_aligned, ord="fro")
    denominator = np.linalg.norm(xc, ord="fro") + 1e-12
    return float(numerator / denominator)


def build_distance_matrix(models: List[Dict[str, np.ndarray]], labels: List[str], pca_dim: int) -> np.ndarray:
    n = len(models)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, j in itertools.combinations(range(n), 2):
        xa, xb, _ = stack_common(models[i], models[j])
        d = procrustes_disparity(xa, xb, pca_dim=pca_dim)
        matrix[i, j] = matrix[j, i] = d
    return matrix


def enumerate_pairs(matrix: np.ndarray, labels: List[str]) -> List[Tuple[float, str, str]]:
    triples: List[Tuple[float, str, str]] = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        triples.append((matrix[i, j], labels[i], labels[j]))
    triples.sort(key=lambda x: x[0])
    return triples


def print_closest_pairs(triples: List[Tuple[float, str, str]], topk: int) -> None:
    print("\n=== Closest pairs ===")
    for d, a, b in triples[:topk]:
        print(f"{a} ↔ {b}  disparity={d:.6f}")


def analyze_unknown(matrix: np.ndarray, labels: List[str], unknown_label: str) -> None:
    if unknown_label not in labels:
        raise ValueError(f"Unknown label '{unknown_label}' not found.")
    idx = labels.index(unknown_label)
    distances = [(matrix[idx, j], labels[j]) for j in range(len(labels)) if j != idx]
    distances.sort(key=lambda x: x[0])
    best_distance, best_label = distances[0]

    print(f"\n=== Unknown model: {unknown_label} ===")
    print(f"Closest neighbor: {best_label} (disparity={best_distance:.6f})")

    all_pairs = [(matrix[i, j], labels[i], labels[j]) for i, j in itertools.combinations(range(len(labels)), 2)]
    all_pairs.sort(key=lambda x: x[0])
    rank = next(
        k
        for k, (d, a, b) in enumerate(all_pairs, start=1)
        if {a, b} == {unknown_label, best_label}
    )
    print(f"Global rank of ({unknown_label}, {best_label}) = #{rank} out of {len(all_pairs)} pairs.")

    dvals = np.array([d for d, _, _ in all_pairs], dtype=np.float64)
    mean = float(dvals.mean())
    std = float(dvals.std(ddof=1)) if len(dvals) > 1 else 1.0
    z_score = (best_distance - mean) / (std + 1e-12)
    print(f"Mean disparity={mean:.6f}, std={std:.6f}, z-score={z_score:.3f}")


def plot_heatmap(matrix: np.ndarray, labels: List[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 2, 0.6 * len(labels) + 2))
    im = ax.imshow(matrix, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Procrustes disparity (lower = more similar)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Procrustes analysis over embedding parquet files.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="Embedding parquet files to compare.")
    parser.add_argument(
        "--unknown",
        type=Path,
        required=True,
        help="Path to the embedding parquet to treat as the unknown model.",
    )
    parser.add_argument("--pca-dim", type=int, default=0, help="Optional PCA dimension (0 = none).")
    parser.add_argument("--topk", type=int, default=10, help="Number of closest pairs to print.")
    parser.add_argument("--heatmap", type=Path, help="Optional path to save a heatmap image.")
    parser.add_argument("--csv", type=Path, help="Optional path to write pairwise disparities as CSV.")
    args = parser.parse_args()

    paths: List[Path] = list(dict.fromkeys(args.inputs))  # preserve order, drop duplicates
    if args.unknown not in paths:
        paths.append(args.unknown)

    models: List[Dict[str, np.ndarray]] = []
    labels: List[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        emb_dict, label = load_embeddings(path)
        models.append(emb_dict)
        labels.append(label)

    matrix = build_distance_matrix(models, labels, pca_dim=args.pca_dim)
    print(f"Loaded {len(labels)} embedding sets.")
    triples = enumerate_pairs(matrix, labels)
    print_closest_pairs(triples, args.topk)
    analyze_unknown(matrix, labels, unknown_label=args.unknown.stem)

    if args.csv:
        df = pd.DataFrame(
            [(a, b, d) for d, a, b in triples],
            columns=["model_a", "model_b", "disparity"],
        )
        df.to_csv(args.csv, index=False)
        print(f"Wrote CSV to {args.csv}")

    if args.heatmap:
        plot_heatmap(matrix, labels, args.heatmap)


if __name__ == "__main__":
    main()
