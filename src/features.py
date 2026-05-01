"""
RFM (Recency, Frequency, Monetary) feature engineering.

Recency  : dias desde a última compra (relativo à data de referência)
Frequency: número de pedidos únicos
Monetary : receita total (£)

Pós-processamento:
  - Log1p transform para reduzir skewness
  - StandardScaler
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def calcular_rfm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Returns:
        rfm_raw    — DataFrame com R/F/M originais por CustomerID
        rfm_scaled — DataFrame com R/F/M log+scaled para clustering
        scaler     — StandardScaler ajustado
    """
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency  =("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("InvoiceNo",   "nunique"),
        Monetary =("Revenue",     "sum"),
    ).reset_index()

    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].copy()
    rfm_log["Recency"]   = np.log1p(rfm_log["Recency"])
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"]  = np.log1p(rfm_log["Monetary"])

    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(
        scaler.fit_transform(rfm_log),
        columns=["Recency", "Frequency", "Monetary"],
        index=rfm.index,
    )

    logger.info(
        "RFM — %d clientes | Recency [%d, %d] dias | Monetary [£%.0f, £%.0f]",
        len(rfm),
        rfm["Recency"].min(), rfm["Recency"].max(),
        rfm["Monetary"].min(), rfm["Monetary"].max(),
    )
    return rfm, rfm_scaled, scaler
