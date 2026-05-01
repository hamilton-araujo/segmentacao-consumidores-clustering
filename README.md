# Segmentação de Consumidores — RFM + K-Means + GMM

Pipeline de segmentação de clientes baseado em RFM (Recency, Frequency, Monetary) com clustering determinístico (K-Means) e estocástico/probabilístico (Gaussian Mixture Models).

## Dataset

**UCI Online Retail** — Kaggle (`carrie1/ecommerce-data`)  
541.909 transações | 4.338 clientes | 37 países | £8.9M de receita

## Stack

| Camada | Tecnologia |
|---|---|
| Features | RFM + Log1p + StandardScaler |
| Clustering | K-Means (Elbow + Silhouette) |
| Clustering Estocástico | Gaussian Mixture Models (BIC) |
| Visualização | PCA 2D · Boxplots · Heatmap |

## Pipeline

```
Carga → Limpeza → RFM → K-Means (elbow) → GMM (BIC) → Perfis → Relatório
```

## Resultados

```
══════════════════════════════════════════════════════
  SEGMENTAÇÃO DE CONSUMIDORES — RFM + GMM
  Clientes: 4,338
══════════════════════════════════════════════════════
  K-Means k=2  Silhouette: 0.433
  GMM     k=8  Silhouette: 0.092

  Cluster    Clientes   Recency    Freq    Monetary
  0 (Dormentes) 1,492     157d     1.0    £361
  1 (VIP)         413      18d    13.4    £4,861
  6 (Champions)    76      25d    38.7   £42,140
══════════════════════════════════════════════════════
```

**Insight**: Cluster 6 (Champions) representa apenas 1.8% dos clientes mas concentra £3.2M em receita (36% do total).

## Estrutura

```
├── src/
│   ├── ingest.py      # Carga e limpeza das transações
│   ├── features.py    # RFM calculation + scaling
│   ├── clustering.py  # K-Means + GMM
│   ├── report.py      # Painel CLI + gráficos
│   └── main.py        # CLI argparse
├── data/
│   └── ecommerce.csv
├── output/
├── tests/
└── requirements.txt
```
