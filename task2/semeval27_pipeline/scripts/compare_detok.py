"""Compara nuestro destokenizador (detok) con el de Moses (detok_moses, paquete sacremoses).

Ejemplo:
    pip install sacremoses
    python scripts/compare_detok.py --data_dir data/dev --out outputs/compare_detok

Para cada idioma aplica, sobre los mismos usos:
    A = fix_span + detok       + ws   (nuestras reglas)
    B = fix_span + detok_moses + ws   (Moses)
y muestra SOLO las frases donde el resultado es distinto, con el trozo que cambia
resaltado. Guarda todas las diferencias en <out>/<LANG>.detok_diff.tsv para revisarlas
a mano (columna `mejor`: A, B, igual o ninguno).

Comprueba además que la palabra objetivo sigue bien marcada en las dos versiones.
"""
import argparse
import difflib
from pathlib import Path

import pandas as pd

from semeval27.data import list_languages, load_language
from semeval27.preprocess import transform_text, fix_span

A_STEPS = ["fix_span", "detok", "ws"]
B_STEPS = ["fix_span", "detok_moses", "ws"]


def apply(text, start, end, word, lang, steps):
    s, e = fix_span(text, start, end, word)
    try:
        t, s, e = transform_text(text, s, e, steps, lang)
        return t, t[s:e], ""
    except ValueError as err:           # se perdió un centinela: el paso rompió la marca
        return text, "", f"centinela perdido: {err}"


def highlight(a: str, b: str, context: int = 25) -> tuple[str, str]:
    """Devuelve los trozos de a y b alrededor de la primera diferencia, con [[ ]]."""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            lo_a, lo_b = max(0, i1 - context), max(0, j1 - context)
            return (f"…{a[lo_a:i1]}[[{a[i1:i2]}]]{a[i2:i2 + context]}…",
                    f"…{b[lo_b:j1]}[[{b[j1:j2]}]]{b[j2:j2 + context]}…")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--out", default="outputs/compare_detok")
    ap.add_argument("--show", type=int, default=15, help="nº de diferencias a imprimir por idioma")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang in args.langs or list_languages(args.data_dir):
        df = load_language(Path(args.data_dir) / lang)
        rows = []
        for r in df.itertuples(index=False):
            ta, tga, erra = apply(r.text, int(r.start), int(r.end), r.word, lang, A_STEPS)
            tb, tgb, errb = apply(r.text, int(r.start), int(r.end), r.word, lang, B_STEPS)
            if ta != tb or tga != tgb or erra or errb:
                ha, hb = highlight(ta, tb)
                rows.append({"word": r.word, "sentence_id": r.sentence_id,
                             "target_A": tga, "target_B": tgb,
                             "diff_A_ours": ha, "diff_B_moses": hb,
                             "error": erra or errb, "mejor": "",
                             "text_raw": r.text, "A_ours": ta, "B_moses": tb})
        diff = pd.DataFrame(rows)
        path = out / f"{lang}.detok_diff.tsv"
        diff.to_csv(path, sep="\t", index=False)

        n = len(df)
        changed_target = int((diff["target_A"] != diff["target_B"]).sum()) if len(diff) else 0
        errors = int((diff["error"] != "").sum()) if len(diff) else 0
        print(f"\n=== {lang}: {len(diff)} de {n} frases distintas ({len(diff) / n:.1%}) | "
              f"palabra objetivo distinta: {changed_target} | centinelas perdidos: {errors}")
        print(f"    todas las diferencias → {path}")
        for r in diff.head(args.show).itertuples(index=False):
            print(f"\n  [{r.word}] {r.sentence_id}")
            print(f"    A nuestro: {r.diff_A_ours}")
            print(f"    B Moses  : {r.diff_B_moses}")
            if r.error:
                print(f"    ERROR    : {r.error}")


if __name__ == "__main__":
    main()
