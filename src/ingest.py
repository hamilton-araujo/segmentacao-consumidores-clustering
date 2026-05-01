"""Carga e limpeza do UCI Online Retail dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ecommerce.csv"


def carregar() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {CSV_PATH}.\n"
            "Baixe via: kaggle datasets download -d carrie1/ecommerce-data"
        )
    df = pd.read_csv(CSV_PATH, encoding="latin1")
    df = _limpar(df)
    logger.info("Dataset carregado: %d transações | %d clientes",
                len(df), df["CustomerID"].nunique())
    return df


def _limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    logger.info("Após limpeza: %d transações | %d clientes",
                len(df), df["CustomerID"].nunique())
    return df.reset_index(drop=True)


def resumo(df: pd.DataFrame) -> None:
    print(f"\n{'─'*52}")
    print(f"  UCI Online Retail — {df['CustomerID'].nunique():,} clientes")
    print(f"{'─'*52}")
    print(f"  Transações     : {len(df):,}")
    print(f"  Período        : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    print(f"  Países         : {df['Country'].nunique()}")
    print(f"  Receita total  : £{df['Revenue'].sum():,.0f}")
    print(f"{'─'*52}\n")
