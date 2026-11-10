"""Paso 4: predicciones (prob) → fichero de envío de la Subtask 2 para Codabench.

Ejemplo:
    python scripts/make_submission.py \
        --pred SV=outputs/preds/SV.tartunlp_ru.tsv \
        --threshold 0.35 --data_dir data/dev --out submissions/st2_tartunlp

Formato oficial (página "Submission Format"): JSONL, una línea por uso,
    {"word": "...", "sentence_id": "...", "label": 1}
con label entero 0 o 1.

Genera <out>/<LANG>/subtask2.jsonl y <out>.zip con esa misma estructura.
Comprueba el nombre del fichero y las carpetas con el ejemplo de envío que
dan los organizadores; si difieren, usa --filename y/o --flat.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semeval27.data import read_jsonl  # noqa: E402


def parse_pred_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("usa LANG=ruta, p. ej. SV=outputs/preds/SV.tsv")
    lang, path = value.split("=", 1)
    return lang, Path(path)


def build_rows(pred_path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(pred_path, sep="\t", dtype={"sentence_id": str})
    missing = {"word", "sentence_id", "prob"} - set(df.columns)
    if missing:
        sys.exit(f"{pred_path}: faltan columnas {sorted(missing)}")
    if df["prob"].isna().any():
        sys.exit(f"{pred_path}: hay filas sin prob")
    df["label"] = (df["prob"] >= threshold).astype(int)
    if df.duplicated(["word", "sentence_id"]).any():
        sys.exit(f"{pred_path}: hay (word, sentence_id) repetidos")
    return df[["word", "sentence_id", "label"]]


def check_coverage(rows: pd.DataFrame, lang_dir: Path) -> None:
    """Avisa si falta o sobra algún uso respecto a usages.jsonl."""
    usages = read_jsonl(lang_dir / "usages.jsonl")
    expected = set(zip(usages["word"], usages["sentence_id"].astype(str)))
    got = set(zip(rows["word"], rows["sentence_id"]))
    if expected != got:
        sys.exit(f"{lang_dir.name}: faltan {len(expected - got)} usos y sobran "
                 f"{len(got - expected)} respecto a usages.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", type=parse_pred_arg, required=True,
                    help="LANG=ruta.tsv (salida de predict_tartunlp.py), uno por idioma")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--data_dir", help="carpeta con <LANG>/usages.jsonl para comprobar que están todos los usos")
    ap.add_argument("--out", default="submissions/subtask2")
    ap.add_argument("--filename", default="subtask2.jsonl")
    ap.add_argument("--flat", action="store_true",
                    help="sin carpetas por idioma (solo vale con un idioma)")
    args = ap.parse_args()

    if args.flat and len(args.pred) > 1:
        sys.exit("--flat solo vale con un idioma")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for lang, path in args.pred:
        rows = build_rows(path, args.threshold)
        if args.data_dir:
            check_coverage(rows, Path(args.data_dir) / lang)
        target = out / args.filename if args.flat else out / lang / args.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for r in rows.itertuples(index=False):
                f.write(json.dumps({"word": r.word, "sentence_id": r.sentence_id,
                                    "label": int(r.label)}, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(rows)} usos, {rows['label'].mean():.1%} con label 1 → {target}")

    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"zip para subir: {zip_path}")


if __name__ == "__main__":
    main()
