"""Paso 3: evaluar predicciones de la Subtask 2.

Ejemplo:
    python scripts/evaluate_subtask2.py --pred outputs/preds/SV.tartunlp_ru.tsv --threshold 0.35 --sweep

Lee la tabla con `prob` y `label`, binariza con el umbral y muestra:
  - resumen (F1 macro por palabra y F1 global) y baseline "siempre 1"
  - informe por palabra
  - (con --sweep) F1 para distintos umbrales
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semeval27.metrics import baselines, per_word_report, summary, threshold_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="tsv con columnas prob y label")
    ap.add_argument("--threshold", type=float, default=0.35, help="0.35 = umbral de TartuNLP")
    ap.add_argument("--sweep", action="store_true", help="probar umbrales de 0.05 a 0.95")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.pred, sep="\t", dtype={"sentence_id": str})
    df = df[df["label"].notna()].copy()
    if df.empty:
        sys.exit("No hay etiquetas en el fichero: no se puede evaluar (¿es test?).")
    df["pred"] = (df["prob"] >= args.threshold).astype(int)

    name = Path(args.pred).name.replace(".tsv", "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res = {"file": name, "threshold": args.threshold, **summary(df), "baselines": baselines(df)}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (out / f"{name}.summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    report = per_word_report(df)
    print("\nPor palabra:\n" + report.to_string(index=False))
    report.to_csv(out / f"{name}.per_word.tsv", sep="\t", index=False)

    if args.sweep:
        sweep = threshold_sweep(df)
        print("\nBarrido de umbral:\n" + sweep.to_string(index=False))
        sweep.to_csv(out / f"{name}.sweep.tsv", sep="\t", index=False)
        best = sweep.loc[sweep["f1_macro_words"].idxmax()]
        print(f"\nMejor umbral (F1 macro por palabra): {best['threshold']} → {best['f1_macro_words']}"
              "\n¡Ojo! Elegido sobre estos mismos datos: es optimista.")


if __name__ == "__main__":
    main()
