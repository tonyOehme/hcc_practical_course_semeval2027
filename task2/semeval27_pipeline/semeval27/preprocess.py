"""Pipeline de entrada: limpiar usos y definiciones antes de construir los pares.

Núcleo independiente del idioma. Cada paso es una función pequeña y se activa por
nombre, para poder medir su efecto por ablación (con y sin cada paso).

Pasos (en este orden):
    fix_span    repara offsets start/end mal alineados (el span incluye espacios)
    dehyphen    une palabras cortadas por OCR: "mån¬ ga" → "många"
    join_hyph   une "accep- terat" → "accepterat" (salvo "upp- och ..."); más agresivo
    detok       quita los espacios que la tokenización puso antes de la puntuación:
                "intrång , mr" → "intrång, mr"
    ws          colapsa espacios repetidos y recorta los extremos

Todos los pasos que tocan el texto mantienen la posición de la palabra objetivo:
se rodea con dos caracteres centinela de uso privado, se transforma el texto y se
recuperan start/end buscando los centinelas.

Los usos de la definición solo reciben limpieza de caracteres invisibles (clean_definition).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

L, R = "", ""  # centinelas (Unicode de uso privado: no aparecen en los datos)

CORE_STEPS = ["fix_span", "dehyphen", "detok", "ws"]
ALL_STEPS = ["fix_span", "dehyphen", "join_hyph", "detok", "ws"]

# Tras "palabra- " estas palabras indican elipsis de compuesto ("upp- och nedgång"), no un corte de línea.
_KEEP_AFTER_HYPHEN = {"och", "eller", "samt", "resp", "respektive", "till", "som", "and", "or",
                      "и", "или", "ja", "tai"}


# ---------------------------------------------------------------- pasos sobre el span

def fix_span(text: str, start: int, end: int, lemma: str, window: int = 80) -> tuple[int, int]:
    """Si el span tiene espacios en los bordes, está desplazado: buscamos la forma de la palabra.

    Candidatos: palabras del texto que empiezan como el lema (3 primeras letras, sin
    mayúsculas); se elige la más cercana al start original. Si no hay candidato, se
    deja el span recortado de espacios.
    """
    span = text[start:end]
    if span == span.strip() and span:
        return start, end
    prefix = re.escape(lemma[:3].lower())
    cands = [m for m in re.finditer(rf"\b{prefix}\w*", text, flags=re.IGNORECASE)
             if abs(m.start() - start) <= window]
    if cands:
        best = min(cands, key=lambda m: abs(m.start() - start))
        return best.start(), best.end()
    # sin candidato: recortar espacios del span
    s = start + (len(span) - len(span.lstrip()))
    e = end - (len(span) - len(span.rstrip()))
    return (s, e) if e > s else (start, end)


# ---------------------------------------------------------------- pasos sobre el texto

def _dehyphen(t: str) -> str:
    # "¬" es el guion de fin de línea del OCR: siempre une
    return re.sub(r"(\w)¬\s*", r"\1", t)


def _join_hyph(t: str) -> str:
    def repl(m: re.Match) -> str:
        nxt = m.group(2)
        if nxt.lower() in _KEEP_AFTER_HYPHEN or nxt[:1].isupper():
            return m.group(0)
        return m.group(1) + nxt
    return re.sub(r"(\w)- (\w+)", repl, t)


def _detok(t: str) -> str:
    t = re.sub(r" +([,.;:!?)\]»%])", r"\1", t)   # "palabra ," → "palabra,"
    t = re.sub(r"([(\[«]) +", r"\1", t)           # "( palabra" → "(palabra"
    return t


def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


TEXT_STEPS = {"dehyphen": _dehyphen, "join_hyph": _join_hyph, "detok": _detok, "ws": _ws}


def transform_text(text: str, start: int, end: int, steps: list[str]) -> tuple[str, int, int]:
    """Aplica los pasos de texto manteniendo la posición de la palabra objetivo."""
    t = f"{text[:start]}{L}{text[start:end]}{R}{text[end:]}"
    for name in steps:
        if name in TEXT_STEPS:
            t = TEXT_STEPS[name](t)
    s = t.index(L)
    t = t.replace(L, "", 1)
    e = t.index(R)
    t = t.replace(R, "", 1)
    return t, s, e


def clean_definition(d: str) -> str:
    """Quita guiones blandos y caracteres invisibles, normaliza a NFC y colapsa espacios."""
    d = unicodedata.normalize("NFC", str(d))
    d = re.sub(r"[­​-‍﻿]", "", d)
    return _ws(d)


# ---------------------------------------------------------------- sobre el DataFrame

def preprocess_usages(df: pd.DataFrame, steps: list[str] | None = None) -> pd.DataFrame:
    """Devuelve una copia con text/start/end/target limpios y el original en text_raw.

    df: salida de semeval27.data.load_language (columnas word, text, start, end, ...).
    steps: nombres de ALL_STEPS; None = CORE_STEPS; [] = no tocar nada.
    """
    steps = CORE_STEPS if steps is None else steps
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"Pasos desconocidos: {sorted(unknown)}. Válidos: {ALL_STEPS}")

    out = df.copy()
    out["text_raw"] = out["text"]
    texts, starts, ends = [], [], []
    for text, s, e, w in zip(out["text"], out["start"].astype(int), out["end"].astype(int), out["word"]):
        if "fix_span" in steps:
            s, e = fix_span(text, s, e, w)
        text, s, e = transform_text(text, s, e, steps)
        texts.append(text); starts.append(s); ends.append(e)
    out["text"], out["start"], out["end"] = texts, starts, ends
    out["target"] = [t[s:e] for t, s, e in zip(out["text"], out["start"], out["end"])]
    if steps and "definition" in out:   # steps=[] → datos tal cual (baseline)
        out["definition"] = out["definition"].map(clean_definition, na_action="ignore")
    return out


def change_report(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Cuántos usos cambió el preprocesado (para imprimir y documentar)."""
    return {
        "usos": len(after),
        "texto_cambiado": int((before["text"].values != after["text"].values).sum()),
        "spans_con_espacios_antes": int(sum(t != t.strip() for t in before["target"])),
        "spans_con_espacios_despues": int(sum(t != t.strip() for t in after["target"])),
        "definiciones_cambiadas": int((before.groupby("word")["definition"].first()
                                       != after.groupby("word")["definition"].first()).sum())
                                  if "definition" in before else 0,
    }
