"""Paso 1: datos oficiales de SemEval-2027 → pares (definición, uso).

Ejemplos:
    python scripts/convert_data.py --data_dir data/dev --out_dir outputs/pairs                 # datos tal cual
    python scripts/convert_data.py --data_dir data/dev --steps core --tag core                 # pipeline de entrada
    python scripts/convert_data.py --data_dir data/dev --steps core --mark_target --tag core_mark

Genera, por idioma (con --tag X: <LANG>.X.pairs.tsv, para no pisar otras variantes):
    outputs/pairs/<LANG>.pairs.tsv   tabla con metadatos (para predecir y evaluar)
    outputs/pairs/<LANG>.jsonl       formato de entrenamiento de TartuNLP (si hay etiquetas)

--steps: none (por defecto) | core | hist | all | lista de pasos (ver semeval27/preprocess.py)
"""
import argparse
import sys
from pathlib import Path

from semeval27.data import list_languages, load_language
from semeval27.preprocess import ORDER, PRESETS, change_report, preprocess_usages, resolve_steps
from semeval27.pairs import make_pairs, write_pairs_tsv, write_tartunlp_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="carpeta con subcarpetas por idioma (SV, EN, ...)")
    ap.add_argument("--out_dir", default="outputs/pairs")
    ap.add_argument("--langs", nargs="*", help="idiomas a convertir (por defecto, todos)")
    ap.add_argument("--mark_target", action="store_true", help="marcar la palabra con <t> </t>")
    ap.add_argument("--steps", nargs="*", default=["none"],
                    help=f"pasos de limpieza: un preset {list(PRESETS)} o nombres de {ORDER}")
    ap.add_argument("--tag", default="", help="sufijo de los ficheros de salida, p. ej. core")
    args = ap.parse_args()

    langs = args.langs or list_languages(args.data_dir)
    if not langs:
        sys.exit(f"No hay carpetas de idioma con usages.jsonl en {args.data_dir}")

    steps = resolve_steps(args.steps)
    suffix = f".{args.tag}" if args.tag else ""

    out = Path(args.out_dir)
    for lang in langs:
        usages = load_language(Path(args.data_dir) / lang)
        if steps:
            clean = preprocess_usages(usages, steps)
            print(f"{lang} limpieza {steps}: {change_report(usages, clean)}")
            usages = clean
        pairs = make_pairs(usages, mark=args.mark_target)
        write_pairs_tsv(pairs, out / f"{lang}{suffix}.pairs.tsv")
        n_lab = int(pairs["label"].notna().sum())
        if n_lab:
            write_tartunlp_jsonl(pairs, out / f"{lang}{suffix}.jsonl")
        pos = pairs["label"].mean() if n_lab else float("nan")
        print(f"{lang}: {usages['word'].nunique()} palabras, {len(pairs)} pares, "
              f"{n_lab} etiquetados, {pos:.1%} positivos → {out}/{lang}{suffix}.pairs.tsv")


if __name__ == "__main__":
    main()
