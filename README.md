# Segmentação + CLV + Alocação de Campanhas — Decisão CMO

> **A pergunta do CMO:** *Tenho R$ 1M para campanhas Q1. Como alocar entre Champions, VIPs, Em Risco e Dormentes? Qual cluster traz maior ROI marginal?*

Pipeline de segmentação RFM + K-Means/GMM enriquecido com **Customer Lifetime Value**, **personas operacionais de marketing** e **alocação ótima de budget** por persona, terminando em decisão executiva sobre o orçamento de campanhas.

---

## Por que existe

Clustering técnico (k=8 GMM, Silhouette 0.092) não move ponteiro de marketing. CMO precisa de três respostas:

| Pergunta | Sinal técnico |
|---|---|
| Quem são meus clientes em linguagem de negócio? | Personas (Champions, At Risk, Loyal, Hibernating, Lost...) |
| Quanto cada um vale nos próximos 12 meses? | CLV via BG/NBD heurístico |
| Como alocar R$ 1M entre eles? | Greedy ROI por persona × cap orçamentário |

---

## A história em três atos

### Ato 1 — A pauta
CFO aprovou R$ 1M para campanhas Q1. CMO precisa apresentar plano até sexta. Você roda:
```bash
python -m src.exec_report
```

### Ato 2 — A evidência
3 segundos depois:
```
Clientes analisados        4,338
CLV total (12m)            £ 6.34M
Budget alocado             R$ 19k
Receita marginal esperada  R$ 411k
ROI global                 2104%
Personas atendidas         7
```

### Ato 3 — A decisão
A alocação prioriza **At Risk** (lift 25%) e **Hibernating** (lift 15%) — onde o uplift de campanha é maior. Champions recebem retenção VIP defensiva (lift 5%) — basta manter satisfação. Lost são descartados (ROI marginal negativo).

---

## Modelos

### RFM Score
```
Recency   = dias desde última compra
Frequency = pedidos únicos
Monetary  = receita acumulada (£)
```
Log1p + StandardScaler antes do clustering.

### Clustering
- **K-Means** com Elbow + Silhouette para k ótimo
- **GMM** com BIC (atribuição probabilística)

### Personas Operacionais
Heurística sobre R/F/M-scores quartilizados:

| Persona | R | F | M | Estratégia |
|---|---|---|---|---|
| Champions | 4 | 4 | 4 | Retenção VIP |
| Loyal | 4 | 3-4 | 2-3 | Cross-sell |
| Big Spenders | 4 | 2-3 | 4 | Upsell premium |
| At Risk | 1-2 | 3-4 | 3-4 | **Reativação prioritária** |
| Hibernating | 1-2 | 2-3 | 2-3 | Reativação |
| Promising | 4 | 1-2 | 1-2 | Onboarding |
| Lost | 1 | 1 | 1 | Não investir |

### Customer Lifetime Value
```
prob_ativo(T)        = exp(-Recency / 90)
compras_esperadas(T) = Frequency × (T_dias/365) × prob_ativo
valor_medio          = Monetary / Frequency
CLV(T)               = compras × valor × (1 - desconto/2)
```
Discount rate r = 10% anual.

### Alocação Ótima de Campanhas
Greedy por ROI unitário decrescente:
```
ROI(persona) = (CLV_medio × lift - custo_contato) / custo_contato
```
Aloca budget até esgotar, respeitando cap = n_clientes × custo_contato.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | UCI Online Retail (Kaggle) · Pandas |
| Segmentação | scikit-learn (K-Means + GMM) |
| Personas | Heurística RFM-score quartil |
| CLV | NumPy (BG/NBD-like) |
| Otimização | Greedy ROI marginal |

---

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Pipeline executivo (recomendado)
python -m src.exec_report

# Pipeline original (RFM + clustering)
python -m src.main
```

---

## Outputs

```
output/
├── relatorio_cmo.md             # ⭐ Briefing CMO + alocação Q1
├── personas_clv.png             # ⭐ CLV total por persona
├── personas_stats.csv           # ⭐ Tabela RFM + CLV por persona
├── alocacao_campanhas.png       # ⭐ Budget + ROI por persona
├── alocacao_campanhas.csv       # ⭐ Tabela alocação detalhada
├── rfm_clv_personas.csv         # ⭐ Por-cliente: RFM + CLV + persona
├── clusters_pca.png             # K-Means + GMM em PCA 2D
├── elbow_bic.png
├── profile_heatmap.png
├── rfm_boxplots.png
└── rfm_clusters.csv
```

⭐ = adicionado nesta versão.

---

## Estrutura

```
├── src/
│   ├── exec_report.py        # ⭐ Pipeline executivo CMO
│   ├── clv_model.py          # ⭐ Customer Lifetime Value (BG/NBD-like)
│   ├── cluster_personas.py   # ⭐ Personas operacionais
│   ├── campaign_allocation.py # ⭐ Greedy ROI marginal
│   ├── ingest.py
│   ├── features.py           # RFM + log + scaler
│   ├── clustering.py         # K-Means + GMM
│   ├── report.py
│   └── main.py
├── data/
├── output/
├── tests/
└── requirements.txt
```
