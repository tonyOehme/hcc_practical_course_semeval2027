"""Carga de los datos oficiales de SemEval-2027 Task 3.

Estructura esperada (una carpeta por idioma):

    dev/
      SV/
        usages.jsonl        word, sentence_id, text, pos, period_label, start, end, year
        definitions.jsonl   word, definition   (la definición/hipótesis de la Subtask 2)
        subtask1.jsonl      word, period_label, sentence_id, label: [ids de sentido]
        subtask2.jsonl      word, sentence_id, label: 0/1
        usages_count.tsv    nº de usos por palabra y periodo

En test, subtask1/subtask2 no traerán etiquetas (o no existirán): el cargador
funciona igual y simplemente deja esas columnas vacías.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def list_languages(data_dir: str | Path) -> list[str]:
    """Carpetas de idioma (SV, EN, ...) dentro de data_dir."""
    data_dir = Path(data_dir)
    return sorted(p.name for p in data_dir.iterdir()
                  if p.is_dir() and (p / "usages.jsonl").exists())


def load_language(lang_dir: str | Path) -> pd.DataFrame:
    """Un DataFrame con una fila por uso, uniendo usos, definición y etiquetas.

    Columnas: lang, word, sentence_id, text, start, end, target, pos,
    period_label, year, definition, label_st2 (0/1 o NaN), label_st1 (lista o NaN).
    """
    lang_dir = Path(lang_dir)
    usages = read_jsonl(lang_dir / "usages.jsonl")
    usages["lang"] = lang_dir.name
    # La forma concreta de la palabra en la frase (p. ej. "kvinnor" para "kvinna")
    usages["target"] = [t[s:e] for t, s, e in zip(usages["text"], usages["start"], usages["end"])]

    defs_path = lang_dir / "definitions.jsonl"
    if defs_path.exists():
        defs = read_jsonl(defs_path)
        if defs["word"].duplicated().any():
            raise ValueError(f"{defs_path}: hay más de una definición por palabra; "
                             "revisa el formato antes de seguir.")
        usages = usages.merge(defs[["word", "definition"]], on="word", how="left")

    st2_path = lang_dir / "subtask2.jsonl"
    if st2_path.exists():
        st2 = read_jsonl(st2_path)
        if "label" in st2:
            st2 = st2.rename(columns={"label": "label_st2"})
            usages = usages.merge(st2[["word", "sentence_id", "label_st2"]],
                                  on=["word", "sentence_id"], how="left")

    st1_path = lang_dir / "subtask1.jsonl"
    if st1_path.exists():
        st1 = read_jsonl(st1_path)
        if "label" in st1:
            st1 = st1.rename(columns={"label": "label_st1"})
            usages = usages.merge(st1[["word", "sentence_id", "label_st1"]],
                                  on=["word", "sentence_id"], how="left")

    return usages


def load_all(data_dir: str | Path, langs: list[str] | None = None) -> pd.DataFrame:
    """Todos los idiomas de data_dir (o solo los indicados) en un único DataFrame."""
    langs = langs or list_languages(data_dir)
    return pd.concat([load_language(Path(data_dir) / l) for l in langs], ignore_index=True)
