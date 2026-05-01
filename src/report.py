"""Painel CLI + gráficos de segmentação."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from .clustering import ResultadoSegmentacao

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

PALETA = plt.colormaps["tab10"]


def imprimir(resultado: ResultadoSegmentacao, n_clientes: int, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    rfm = resultado.rfm_com_clusters
    perfis = rfm.groupby("Cluster_GMM")[["Recency", "Frequency", "Monetary"]].mean()

    print(f"\n{'═'*54}")
    print("  SEGMENTAÇÃO DE CONSUMIDORES — RFM + GMM")
    print(f"  Clientes: {n_clientes:,}")
    print(f"{'═'*54}")
    print(f"  K-Means k={resultado.k_otimo}  Silhouette: {resultado.silhouette_kmeans:.3f}")
    print(f"  GMM     k={len(perfis)}       Silhouette: {resultado.silhouette_gmm:.3f}")
    print(f"{'─'*54}")
    print(f"  {'Cluster':<10} {'Clientes':>9} {'Recency':>9} {'Freq':>7} {'Monetary':>11}")
    for cl, row in perfis.iterrows():
        n = (rfm["Cluster_GMM"] == cl).sum()
        print(f"  {cl:<10} {n:>9,} {row['Recency']:>9.0f} {row['Frequency']:>7.1f} £{row['Monetary']:>9,.0f}")
    print(f"{'═'*54}\n")

    rfm.to_csv(output_dir / "rfm_clusters.csv", index=False)

    _elbow_bic(resultado, output_dir)
    _scatter_pca(resultado, output_dir)
    _rfm_boxplots(rfm, output_dir)
    _profile_heatmap(perfis, output_dir)


def _elbow_bic(resultado: ResultadoSegmentacao, out: Path):
    from . import clustering as cl_mod
    ks = list(cl_mod.K_RANGE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, resultado.inertias, "o-", color="steelblue")
    ax1.set_title("K-Means — Elbow (Inertia)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, resultado.bic_scores, "o-", color="darkorange")
    ax2.set_title("GMM — BIC Score")
    ax2.set_xlabel("k")
    ax2.set_ylabel("BIC")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "elbow_bic.png", dpi=150, bbox_inches="tight")
    plt.close()


def _scatter_pca(resultado: ResultadoSegmentacao, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    coords = resultado.pca_coords

    for k, labels, ax, title in [
        (resultado.k_otimo, resultado.labels_kmeans, ax1, "K-Means"),
        (resultado.probs_gmm.shape[1], resultado.labels_gmm, ax2, "GMM"),
    ]:
        colors = [PALETA(i / max(k - 1, 1)) for i in labels]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6)
        ax.set_title(f"Clusters PCA 2D — {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out / "clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.close()


def _rfm_boxplots(rfm: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    dims = ["Recency", "Frequency", "Monetary"]
    for ax, dim in zip(axes, dims):
        groups = [rfm.loc[rfm["Cluster_GMM"] == c, dim].values
                  for c in sorted(rfm["Cluster_GMM"].unique())]
        ax.boxplot(groups, labels=sorted(rfm["Cluster_GMM"].unique()))
        ax.set_title(f"{dim} por Cluster (GMM)")
        ax.set_xlabel("Cluster")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "rfm_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()


def _profile_heatmap(perfis: pd.DataFrame, out: Path):
    normed = (perfis - perfis.min()) / (perfis.max() - perfis.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(normed.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(normed.columns)))
    ax.set_xticklabels(normed.columns)
    ax.set_yticks(range(len(normed)))
    ax.set_yticklabels([f"Cluster {i}" for i in normed.index])
    plt.colorbar(im, ax=ax, label="Valor normalizado")
    ax.set_title("Perfil Médio por Cluster — RFM (GMM)")
    for i in range(len(normed)):
        for j, col in enumerate(normed.columns):
            ax.text(j, i, f"{perfis.loc[normed.index[i], col]:.1f}",
                    ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(out / "profile_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
