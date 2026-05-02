"""
Relatório executivo CMO — Segmentação + CLV + Alocação de Campanhas.
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import carregar
from src.features import calcular_rfm
from src.clv_model import calcular_clv
from src.cluster_personas import classificar_personas, estatisticas_persona
from src.campaign_allocation import otimizar

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CORES_PERSONA = {
    "Champions": "#27ae60",  "Loyal": "#3498db",  "Big Spenders": "#9b59b6",
    "At Risk": "#e67e22",    "Hibernating": "#95a5a6",
    "Promising": "#f39c12",  "Lost": "#7f8c8d",
}


def _grafico_alocacao(df_aloc, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cs = [CORES_PERSONA.get(p, "#bdc3c7") for p in df_aloc["persona"]]

    axes[0].bar(df_aloc["persona"], df_aloc["custo_total"] / 1e3, color=cs, alpha=0.85)
    axes[0].set_ylabel("Budget alocado (R$ k)")
    axes[0].set_title("Alocação de Budget por Persona")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(df_aloc["persona"], df_aloc["roi_pct"], color=cs, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("ROI esperado (%)")
    axes[1].set_title("ROI por Persona")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_personas(stats: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    cs = [CORES_PERSONA.get(p, "#bdc3c7") for p in stats.index]
    bars = ax.bar(stats.index, stats["clv_total"] / 1e3, color=cs, alpha=0.85)
    for bar, n in zip(bars, stats["n_clientes"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={int(n)}", ha="center", fontsize=9)
    ax.set_ylabel("CLV total (R$ k)")
    ax.set_title("Customer Lifetime Value Total por Persona (12 meses)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Carregando transações...")
    df = carregar()

    rfm_raw, rfm_scaled, _ = calcular_rfm(df)

    logger.info("Calculando CLV (12 meses)...")
    rfm_clv, clv_res = calcular_clv(rfm_raw, horizon_meses=12)

    logger.info("Classificando personas...")
    rfm_p = classificar_personas(rfm_clv)
    rfm_p.to_csv(OUTPUT_DIR / "rfm_clv_personas.csv", index=False)

    stats = estatisticas_persona(rfm_p)
    stats.to_csv(OUTPUT_DIR / "personas_stats.csv")

    logger.info("Otimizando alocação de R$ 1M...")
    df_aloc, resumo = otimizar(stats, budget_total=1_000_000.0)
    df_aloc.to_csv(OUTPUT_DIR / "alocacao_campanhas.csv", index=False)

    _grafico_personas(stats, OUTPUT_DIR / "personas_clv.png")
    _grafico_alocacao(df_aloc, OUTPUT_DIR / "alocacao_campanhas.png")

    # Markdown
    lines = [
        "# Segmentação de Consumidores — Decisão CMO",
        "",
        "## Sumário Executivo",
        "",
        f"- **Clientes analisados:** {len(rfm_p):,}",
        f"- **Receita histórica:** £ {rfm_raw['Monetary'].sum()/1e6:.2f}M",
        f"- **CLV total esperado (12m):** £ {clv_res.clv_total/1e6:.2f}M",
        f"- **CLV médio por cliente:** £ {clv_res.clv_medio:,.0f}",
        f"- **Budget Q1:** R$ 1.000.000",
        f"- **ROI global esperado:** {resumo['roi_global_pct']:.0f}%",
        f"- **Receita marginal esperada:** R$ {resumo['receita_total']/1e3:,.0f}k",
        "",
        "![CLV](personas_clv.png)",
        "",
        "---",
        "",
        "## 1. Personas Operacionais",
        "",
        "| Persona | n | Recency | Freq | Monetary Total | CLV Total | CLV Médio |",
        "|---|---|---|---|---|---|---|",
    ]
    for p, r in stats.iterrows():
        lines.append(
            f"| {p} | {int(r['n_clientes'])} | {int(r['recency_medio'])}d | "
            f"{r['freq_media']:.1f} | £ {r['monetary_total']/1e3:,.0f}k | "
            f"£ {r.get('clv_total', 0)/1e3:,.0f}k | £ {r.get('clv_medio', 0):,.0f} |"
        )

    lines += [
        "",
        "## 2. Alocação Ótima de Budget Q1",
        "",
        "![Alocação](alocacao_campanhas.png)",
        "",
        "| Persona | Custo/contato | Lift | n_alvo | Budget | Receita Marg. | ROI |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df_aloc.iterrows():
        lines.append(
            f"| {r['persona']} | R$ {r['custo_unitario']:.2f} | "
            f"{r['lift']*100:.0f}% | {int(r['n_alvo'])} | "
            f"R$ {r['custo_total']/1e3:,.0f}k | "
            f"R$ {r['receita_marginal_total']/1e3:,.0f}k | {r['roi_pct']:.0f}% |"
        )

    lines += [
        "",
        f"**Budget alocado:** R$ {resumo['budget_alocado']/1e3:,.0f}k de R$ 1.000k",
        f"**Personas atendidas:** {resumo['personas_atendidas']}",
        f"**ROI global:** {resumo['roi_global_pct']:.0f}%",
        "",
        "---",
        "",
        "## Metodologia",
        "",
        "- **RFM**: Recency (dias), Frequency (pedidos únicos), Monetary (£ acumulado).",
        "- **Personas**: heurística determinística sobre R/F/M-scores quartilizados.",
        "- **CLV**: heurística BG/NBD-like — prob_ativo·exp(-Recency/90) × freq × valor médio × desconto.",
        "- **Alocação ótima**: greedy por ROI unitário decrescente, respeitando cap por persona.",
    ]
    (OUTPUT_DIR / "relatorio_cmo.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'═'*60}")
    print("  SEGMENTAÇÃO + CLV + CAMPANHAS — DECISÃO CMO")
    print(f"{'═'*60}")
    print(f"  Clientes analisados        {len(rfm_p):,}")
    print(f"  CLV total (12m)            £ {clv_res.clv_total/1e6:.2f}M")
    print(f"  Budget alocado             R$ {resumo['budget_alocado']/1e3:,.0f}k")
    print(f"  Receita marginal esperada  R$ {resumo['receita_total']/1e3:,.0f}k")
    print(f"  ROI global                 {resumo['roi_global_pct']:.0f}%")
    print(f"  Personas atendidas         {resumo['personas_atendidas']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
