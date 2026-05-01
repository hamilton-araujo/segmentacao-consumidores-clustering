"""
Segmentação: K-Means (determinístico) + GMM (estocástico/probabilístico).

K-Means: elbow method (inertia) + silhouette para escolher k
GMM    : BIC para escolha de n_components; responsabilidades suaves por cluster
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

K_RANGE = range(2, 9)
LABELS = {
    0: "Champions",
    1: "Fiéis",
    2: "Em Risco",
    3: "Novos",
    4: "Hibernando",
    5: "Perdidos",
    6: "Promissores",
    7: "Potenciais",
}


@dataclass
class ResultadoSegmentacao:
    k_otimo: int
    silhouette_kmeans: float
    silhouette_gmm: float
    bic_scores: list
    inertias: list
    labels_kmeans: np.ndarray
    labels_gmm: np.ndarray
    probs_gmm: np.ndarray
    pca_coords: np.ndarray
    rfm_com_clusters: pd.DataFrame


def segmentar(rfm_raw: pd.DataFrame, rfm_scaled: pd.DataFrame,
              random_state: int = 42) -> ResultadoSegmentacao:
    X = rfm_scaled.values

    # ── Elbow + Silhouette para escolher k ──────────────────────────────
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lbl))

    k_otimo = list(K_RANGE)[int(np.argmax(silhouettes))]
    logger.info("K-Means — k ótimo: %d (silhouette=%.3f)", k_otimo, max(silhouettes))

    # ── K-Means final ────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=k_otimo, random_state=random_state, n_init=20)
    labels_kmeans = km_final.fit_predict(X)
    sil_km = float(silhouette_score(X, labels_kmeans))

    # ── GMM (estocástico) ────────────────────────────────────────────────
    bic_scores = []
    for k in K_RANGE:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state, n_init=5)
        gm.fit(X)
        bic_scores.append(gm.bic(X))

    k_gmm = list(K_RANGE)[int(np.argmin(bic_scores))]
    logger.info("GMM — k ótimo: %d (BIC=%.1f)", k_gmm, min(bic_scores))

    gm_final = GaussianMixture(n_components=k_gmm, covariance_type="full",
                                random_state=random_state, n_init=20)
    gm_final.fit(X)
    labels_gmm  = gm_final.predict(X)
    probs_gmm   = gm_final.predict_proba(X)
    sil_gmm = float(silhouette_score(X, labels_gmm))

    # ── PCA 2D para visualização ─────────────────────────────────────────
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X)

    rfm_result = rfm_raw.copy()
    rfm_result["Cluster_KMeans"] = labels_kmeans
    rfm_result["Cluster_GMM"]    = labels_gmm
    rfm_result["Prob_max_GMM"]   = probs_gmm.max(axis=1)

    return ResultadoSegmentacao(
        k_otimo=k_otimo,
        silhouette_kmeans=sil_km,
        silhouette_gmm=sil_gmm,
        bic_scores=bic_scores,
        inertias=inertias,
        labels_kmeans=labels_kmeans,
        labels_gmm=labels_gmm,
        probs_gmm=probs_gmm,
        pca_coords=pca_coords,
        rfm_com_clusters=rfm_result,
    )
