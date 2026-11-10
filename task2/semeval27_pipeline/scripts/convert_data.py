"""Paso 1: datos oficiales de SemEval-2027 → pares (definición, uso).

Ejemplo:
    python scripts/convert_data.py --data_dir data/dev --out_dir outputs/pairs

Genera, por idioma:
    outputs/pairs/<LANG>.pairs.tsv   tabla con metadatos (para predecir y evaluar)
    outputs/pairs/<LANG>.jsonl       formato de entrenamiento de TartuNLP (si hay etiquetas)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semeval27.data import list_languages, load_language  # noqa: E402
from semeval27.pairs import make_pairs, write_pairs_tsv, write_tartunlp_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="carpeta con subcarpetas por idioma (SV, EN, ...)")
    ap.add_argument("--out_dir", default="outputs/pairs")
    ap.add_argument("--langs", nargs="*", help="idiomas a convertir (por defecto, todos)")
    ap.add_argument("--mark_target", action="store_true", help="marcar la palabra con <t> </t>")
    args = ap.parse_args()

    langs = args.langs or list_languages(args.data_dir)
    if not langs:
        sys.exit(f"No hay carpetas de idioma con usages.jsonl en {args.data_dir}")

    out = Path(args.out_dir)
    for lang in langs:
        usages = load_language(Path(args.data_dir) / lang)
        pairs = make_pairs(usages, mark=args.mark_target)
        write_pairs_tsv(pairs, out / f"{lang}.pairs.tsv")
        n_lab = int(pairs["label"].notna().sum())
        if n_lab:
            write_tartunlp_jsonl(pairs, out / f"{lang}.jsonl")
        pos = pairs["label"].mean() if n_lab else float("nan")
        print(f"{lang}: {usages['word'].nunique()} palabras, {len(pairs)} pares, "
              f"{n_lab} etiquetados, {pos:.1%} positivos → {out}/{lang}.pairs.tsv")


if __name__ == "__main__":
    main()
