"""Métricas para la Subtask 2 (¿expresa este uso el sentido definido?).

La tarea anuncia "Average F1". Hasta confirmar en Codabench cómo se promedia,
calculamos las dos lecturas razonables:
  - f1_macro_words: F1 de la clase positiva por palabra, promediado entre palabras
  - f1_global: F1 de la clase positiva con todos los usos juntos
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=0)


def per_word_report(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Una fila por (idioma, palabra) con nº de usos, % positivos, P, R y F1."""
    rows = []
    for (lang, word), g in df.groupby(["lang", "word"], sort=True):
        y, p = g["label"].astype(int), g[pred_col].astype(int)
        rows.append({
            "lang": lang, "word": word, "n": len(g),
            "pos_rate": round(y.mean(), 3),
            "pred_rate": round(p.mean(), 3),
            "precision": round(precision_score(y, p, zero_division=0), 3),
            "recall": round(recall_score(y, p, zero_division=0), 3),
            "f1": round(_f1(y, p), 3),
        })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    report = per_word_report(df, pred_col)
    return {
        "f1_macro_words": round(report["f1"].mean(), 4),
        "f1_global": round(_f1(df["label"].astype(int), df[pred_col].astype(int)), 4),
        "n_usages": int(len(df)),
        "n_words": int(len(report)),
    }


def threshold_sweep(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """F1 para distintos umbrales sobre la columna prob."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        tmp = df.assign(pred=(df["prob"] >= t).astype(int))
        rows.append({"threshold": float(t), **summary(tmp)})
    return pd.DataFrame(rows)


def baselines(df: pd.DataFrame) -> dict:
    """Referencias triviales: decir siempre 1 (todos los usos expresan el sentido)."""
    return {"always_1": summary(df.assign(pred=1))}
