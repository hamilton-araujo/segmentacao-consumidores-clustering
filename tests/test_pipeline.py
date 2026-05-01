"""Testes unitários do pipeline de segmentação."""

import numpy as np
import pandas as pd
import pytest

from src.ingest import _limpar
from src.features import calcular_rfm
from src.clustering import segmentar


def _df_sintetico(n=500, seed=42):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2010-01-01", "2011-12-31", periods=n)
    return pd.DataFrame({
        "InvoiceNo":   [f"5{i:05d}" for i in range(n)],
        "StockCode":   rng.choice(["A", "B", "C", "D"], n),
        "Description": ["item"] * n,
        "Quantity":    rng.integers(1, 20, n),
        "InvoiceDate": datas,
        "UnitPrice":   rng.uniform(0.5, 50, n),
        "CustomerID":  rng.integers(1000, 1050, n).astype(float),
        "Country":     rng.choice(["United Kingdom", "France", "Germany"], n),
    })


class TestIngest:
    def test_remove_negativos(self):
        df = _df_sintetico()
        df.loc[0, "Quantity"] = -5
        df_clean = _limpar(df)
        assert (df_clean["Quantity"] > 0).all()

    def test_remove_nulos_customer(self):
        df = _df_sintetico()
        df.loc[0, "CustomerID"] = np.nan
        df_clean = _limpar(df)
        assert df_clean["CustomerID"].isna().sum() == 0

    def test_remove_cancelamentos(self):
        df = _df_sintetico()
        df.loc[0, "InvoiceNo"] = "C12345"
        df_clean = _limpar(df)
        assert not df_clean["InvoiceNo"].astype(str).str.startswith("C").any()

    def test_revenue_criada(self):
        df = _limpar(_df_sintetico())
        assert "Revenue" in df.columns
        assert (df["Revenue"] > 0).all()


class TestFeatures:
    def setup_method(self):
        self.df = _limpar(_df_sintetico())

    def test_rfm_colunas(self):
        rfm, _, _ = calcular_rfm(self.df)
        assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)

    def test_rfm_positivos(self):
        rfm, _, _ = calcular_rfm(self.df)
        assert (rfm["Recency"] >= 0).all()
        assert (rfm["Frequency"] >= 1).all()
        assert (rfm["Monetary"] > 0).all()

    def test_scaled_media_zero(self):
        _, rfm_scaled, _ = calcular_rfm(self.df)
        assert abs(rfm_scaled["Recency"].mean()) < 0.1


class TestClustering:
    def setup_method(self):
        df = _limpar(_df_sintetico())
        self.rfm_raw, self.rfm_scaled, _ = calcular_rfm(df)

    def test_labels_range(self):
        resultado = segmentar(self.rfm_raw, self.rfm_scaled)
        assert resultado.labels_kmeans.min() >= 0
        assert resultado.labels_gmm.min() >= 0

    def test_probs_somam_um(self):
        resultado = segmentar(self.rfm_raw, self.rfm_scaled)
        assert np.allclose(resultado.probs_gmm.sum(axis=1), 1.0, atol=1e-5)

    def test_pca_shape(self):
        resultado = segmentar(self.rfm_raw, self.rfm_scaled)
        assert resultado.pca_coords.shape == (len(self.rfm_raw), 2)
