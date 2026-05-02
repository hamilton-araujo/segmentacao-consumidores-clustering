"""
Alocação ótima de budget de campanhas por persona.

Por que existe:
    R$ 1M de budget Q1. 6 personas. Cada uma responde diferente a
    canais (email, SMS, push, ad). Alocação ingênua (% pop) destrói ROI.

Heurística:
    Custo por contato: { Champions: R$ 8 (preferencial),
                          Loyal: R$ 5,
                          At Risk: R$ 4 (urgência),
                          Hibernating: R$ 2,
                          Lost: R$ 1,
                          Promising: R$ 3 }
    Lift esperado vs base: Champions +5%, Loyal +8%, At Risk +25%,
                            Hibernating +15%, Promising +12%, Lost +3%

    ROI(persona) = (CLV × lift × n_alvo) - (custo_contato × n_alvo)
    Aloca budget proporcionalmente ao ROI esperado descrescente,
    respeitando o cap por persona (n_clientes × custo_contato).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


CUSTO_POR_CONTATO = {
    "Champions":    8.0,
    "Loyal":        5.0,
    "Big Spenders": 6.0,
    "At Risk":      4.0,
    "Hibernating":  2.0,
    "Promising":    3.0,
    "Lost":         1.0,
}

LIFT_ESPERADO = {       # uplift de receita esperado em CLV
    "Champions":    0.05,
    "Loyal":        0.08,
    "Big Spenders": 0.06,
    "At Risk":      0.25,
    "Hibernating":  0.15,
    "Promising":    0.12,
    "Lost":         0.03,
}


@dataclass
class AlocacaoCampanha:
    persona:           str
    n_clientes:        int
    custo_unitario:    float
    lift:              float
    clv_medio:         float
    receita_marginal:  float
    custo_total:       float
    roi:               float
    aloca_budget:      float
    n_alvo:            int


def otimizar(
    stats_persona: pd.DataFrame,    # output de estatisticas_persona (com clv_medio)
    budget_total: float = 1_000_000.0,
) -> tuple[pd.DataFrame, dict]:
    """Aloca budget priorizando ROI marginal por persona."""
    rows = []
    for persona, row in stats_persona.iterrows():
        custo = CUSTO_POR_CONTATO.get(persona, 3.0)
        lift = LIFT_ESPERADO.get(persona, 0.10)
        clv_medio = float(row.get("clv_medio", row.get("monetary_medio", 0.0)))
        n = int(row["n_clientes"])

        receita_marg_unitaria = clv_medio * lift
        roi_unitario = (receita_marg_unitaria - custo) / custo if custo > 0 else 0
        rows.append({
            "persona": persona,
            "n_clientes": n,
            "custo_unitario": custo,
            "lift": lift,
            "clv_medio": clv_medio,
            "receita_marg_unitaria": receita_marg_unitaria,
            "roi_unitario": roi_unitario,
        })

    df = pd.DataFrame(rows).sort_values("roi_unitario", ascending=False)

    # Aloca budget gulosamente até esgotar
    budget_restante = budget_total
    df["n_alvo"] = 0
    df["custo_total"] = 0.0
    for idx, row in df.iterrows():
        cap = row["n_clientes"] * row["custo_unitario"]
        aloca = min(cap, budget_restante)
        df.at[idx, "custo_total"] = aloca
        df.at[idx, "n_alvo"] = int(aloca / row["custo_unitario"]) if row["custo_unitario"] > 0 else 0
        budget_restante -= aloca
        if budget_restante <= 0:
            break

    df["receita_marginal_total"] = df["n_alvo"] * df["receita_marg_unitaria"]
    df["roi_pct"] = np.where(df["custo_total"] > 0,
                             (df["receita_marginal_total"] - df["custo_total"]) / df["custo_total"] * 100,
                             0)

    resumo = {
        "budget_total":        budget_total,
        "budget_alocado":      float(df["custo_total"].sum()),
        "receita_total":       float(df["receita_marginal_total"].sum()),
        "roi_global_pct":      float((df["receita_marginal_total"].sum() - df["custo_total"].sum()) /
                                     df["custo_total"].sum() * 100) if df["custo_total"].sum() > 0 else 0,
        "personas_atendidas":  int((df["n_alvo"] > 0).sum()),
    }
    return df, resumo
