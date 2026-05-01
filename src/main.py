"""CLI — Segmentação de Consumidores (RFM + K-Means + GMM)."""

import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _parse():
    p = argparse.ArgumentParser(description="Customer Segmentation — RFM + K-Means + GMM")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = _parse()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from .ingest import carregar, resumo
    from .features import calcular_rfm
    from .clustering import segmentar
    from .report import imprimir

    df = carregar()
    resumo(df)

    rfm_raw, rfm_scaled, _ = calcular_rfm(df)
    resultado = segmentar(rfm_raw, rfm_scaled)

    imprimir(resultado, len(rfm_raw), args.output_dir)


if __name__ == "__main__":
    main()
