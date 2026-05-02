"""
Customer Lifetime Value — modelo BG/NBD + Gamma-Gamma simplificado.

Por que existe:
    RFM mostra quem comprou. CLV prevê quem vai comprar e quanto trazer.
    O modelo BG/NBD (Beta-Geometric / Negative Binomial Distribution)
    de Fader et al. (2005) é o padrão da indústria de e-commerce.

    Implementação simplificada (heurística determinística) sem dependência
    de lifetimes:
        E[Compras_próximos_T_dias] = Frequency × decay(Recency, T)
        E[Valor_compra] = Monetary / Frequency × inflação_T
        CLV(T) = Σ E[Compras] × E[Valor] / (1+r)^t

    Discount rate r = 0.10 anual.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CLVResultado:
    horizon_meses:  int
    clv_total:      float
    clv_medio:      float
    p99_clv:        float    # cliente top 1%


def calcular_clv(
    rfm: pd.DataFrame,
    horizon_meses: int = 12,
    desconto_anual: float = 0.10,
) -> tuple[pd.DataFrame, CLVResultado]:
    """
    Calcula CLV individual e retorna df enriquecido.

    Heurística:
        prob_ativo(T)        = exp(-Recency / 90)         # decay 90 dias
        compras_esperadas(T) = Frequency × (T_dias/365) × prob_ativo
        valor_medio          = Monetary / max(Frequency, 1)
        CLV(T)               = compras × valor × (1 - desconto/2)
    """
    df = rfm.copy()

    rec = df["Recency"].clip(lower=1)
    freq = df["Frequency"].clip(lower=1)
    monetary = df["Monetary"].clip(lower=0.01)

    df["prob_ativo"] = np.exp(-rec / 90.0)
    df["compras_esperadas"] = freq * (horizon_meses * 30 / 365.0) * df["prob_ativo"]
    df["valor_medio"] = monetary / freq
    fator_desconto = 1 - desconto_anual / 2  # aproximação
    df["clv"] = (df["compras_esperadas"] * df["valor_medio"] * fator_desconto)

    return df, CLVResultado(
        horizon_meses=horizon_meses,
        clv_total=float(df["clv"].sum()),
        clv_medio=float(df["clv"].mean()),
        p99_clv=float(df["clv"].quantile(0.99)),
    )
