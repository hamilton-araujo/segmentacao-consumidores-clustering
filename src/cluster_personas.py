"""
Mapeia clusters em personas operacionais de marketing.

Por que existe:
    Cluster 0/1/2... é matemática. Marketing precisa de nomes:
    Champions, Loyal, Big Spenders, At Risk, Hibernating, Lost.

Heurística baseada em quartis dos R, F, M:
    Champions    : R baixo (recente), F alto, M alto
    Loyal        : R baixo, F alto, M médio
    Big Spenders : R baixo, F médio, M alto
    Promising    : R baixo, F baixo, M baixo  (novos)
    At Risk      : R alto, F alto, M alto      (ex-VIPs sumindo)
    Hibernating  : R alto, F médio, M médio
    Lost         : R alto, F baixo, M baixo
"""

import numpy as np
import pandas as pd


def classificar_personas(rfm: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'persona' ao dataframe RFM."""
    df = rfm.copy()

    df["R_score"] = pd.qcut(-df["Recency"], q=4, labels=[1, 2, 3, 4]).astype(int)
    df["F_score"] = pd.qcut(df["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    df["M_score"] = pd.qcut(df["Monetary"], q=4, labels=[1, 2, 3, 4]).astype(int)

    def _persona(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal"
        if r >= 3 and m >= 3:
            return "Big Spenders"
        if r >= 3:
            return "Promising"
        if f >= 3 and m >= 3:
            return "At Risk"
        if f >= 2 or m >= 2:
            return "Hibernating"
        return "Lost"

    df["persona"] = df.apply(_persona, axis=1)
    return df


def estatisticas_persona(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas RFM + CLV (se houver) por persona."""
    cols = ["Recency", "Frequency", "Monetary"]
    if "clv" in df.columns:
        cols.append("clv")

    agg = df.groupby("persona")[cols].agg(["count", "mean", "sum"])
    agg.columns = ["_".join(c) for c in agg.columns]

    out = pd.DataFrame({
        "n_clientes":    agg["Recency_count"].astype(int),
        "recency_medio": agg["Recency_mean"].round(0),
        "freq_media":    agg["Frequency_mean"].round(1),
        "monetary_total": agg["Monetary_sum"].round(0),
        "monetary_medio": agg["Monetary_mean"].round(0),
    })
    if "clv_sum" in agg.columns:
        out["clv_total"] = agg["clv_sum"].round(0)
        out["clv_medio"] = agg["clv_mean"].round(0)

    return out.sort_values("monetary_total", ascending=False)
