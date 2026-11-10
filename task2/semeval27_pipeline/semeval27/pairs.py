"""Conversión de usos a pares (definición, uso), el formato que usa TartuNLP.

TartuNLP alimenta el modelo con un único texto "definición<TAB>uso" (sin marcar
la palabra objetivo), así que por defecto hacemos lo mismo para que el
adaptador entrenado en AXOLOTL vea la entrada que espera.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PAIR_COLUMNS = ["lang", "word", "sentence_id", "period_label", "target",
                "definition", "text", "pair_text", "label"]


def mark_target(text: str, start: int, end: int, left: str = "<t>", right: str = "</t>") -> str:
    """Rodea la palabra objetivo con marcadores (opcional, como hace XL-LEXEME)."""
    return f"{text[:start]}{left}{text[start:end]}{right}{text[end:]}"


def make_pairs(usages: pd.DataFrame, mark: bool = False) -> pd.DataFrame:
    """Un par por uso: la definición de su palabra + la frase.

    label = etiqueta de la Subtask 2 si existe (dev/train), si no queda vacía (test).
    """
    df = usages[usages["definition"].notna()].copy()
    usage_text = ([mark_target(t, s, e) for t, s, e in zip(df["text"], df["start"], df["end"])]
                  if mark else df["text"])
    df["pair_text"] = df["definition"].str.cat(pd.Series(usage_text, index=df.index), sep="\t")
    df["label"] = df["label_st2"] if "label_st2" in df else pd.NA
    return df[PAIR_COLUMNS].reset_index(drop=True)


def write_pairs_tsv(pairs: pd.DataFrame, path: str | Path) -> None:
    """Tabla completa (con metadatos) para predecir y evaluar después."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(path, sep="\t", index=False)


def read_pairs_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=True,
                       dtype={"sentence_id": str})


def write_tartunlp_jsonl(pairs: pd.DataFrame, path: str | Path) -> None:
    """Formato de entrenamiento de TartuNLP: {"text": "def\\tuso", "label": 0/1}.

    Sirve para entrenar/afinar el adaptador con nuestros datos más adelante.
    Solo se escriben filas con etiqueta.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    labelled = pairs[pairs["label"].notna()]
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(labelled["pair_text"], labelled["label"]):
            f.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=False) + "\n")
