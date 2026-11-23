"""Pipeline de entrada: limpiar usos y definiciones antes de construir los pares.

Tres capas:
  1. Núcleo común (todos los idiomas): fix_span, dehyphen, join_hyph, detok, ws.
  2. Configuración por idioma (LANG_CONFIG): las reglas del núcleo que cambian
     según el idioma (conjunciones, apóstrofos, signos de apertura).
  3. Módulos históricos por idioma (ru_prereform, ...): solo se aplican a su idioma;
     en los demás no hacen nada.

Cada paso se activa por nombre para medir su efecto por ablación. Todos los pasos
que tocan el texto conservan la posición de la palabra objetivo con centinelas
(ver transform_text).

Idiomas de la tarea: SV, EN, IT, ES, NL, RU (+ FI para datos de entrenamiento AXOLOTL).
OJO: solo las reglas del sueco están comprobadas con datos reales (dev SV). Las de
los demás idiomas siguen convenciones conocidas de tokenización y hay que
verificarlas en cuanto tengamos sus datos.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

L, R = "", ""  # centinelas (Unicode de uso privado: no aparecen en los datos)

# ---------------------------------------------------------------- configuración por idioma

# Signos que se pegan a la palabra ANTERIOR / SIGUIENTE en todos los idiomas.
_CLOSE = ",.;:!?)]%…»"
_OPEN = "([«"

LANG_CONFIG: dict[str, dict] = {
    # conj: palabras que, tras "palabra- ", indican elipsis de compuesto ("upp- och nedgång")
    # sym_quotes: comillas que abren Y cierran con el mismo signo (sueco: »doktor»).
    # No se mueven: sin ver toda la cita no se sabe si cada una abre o cierra.
    "sv": {"conj": {"och", "eller", "samt", "resp", "respektive", "till", "som"},
           "sym_quotes": "»"},
    "en": {"conj": {"and", "or", "nor", "to"},
           "contractions": True},          # do n't → don't, it 's → it's
    "it": {"conj": {"e", "ed", "o", "od"},
           "elision": True},               # l' uomo → l'uomo, dell' anno → dell'anno
    "es": {"conj": {"y", "e", "o", "u", "ni"},
           "open_extra": "¿¡"},            # ¿ Qué → ¿Qué
    "nl": {"conj": {"en", "of", "tot"}},
    "ru": {"conj": {"и", "или", "а", "да"}},
    "fi": {"conj": {"ja", "tai", "sekä"}},
}
_DEFAULT_CFG = {"conj": set()}


def lang_key(lang: str | None) -> str:
    """'SV', 'sv', 'swe' → 'sv'. Desconocido → '' (solo reglas comunes)."""
    l = (lang or "").lower()
    aliases = {"swe": "sv", "eng": "en", "ita": "it", "spa": "es", "nld": "nl",
               "rus": "ru", "fin": "fi"}
    return aliases.get(l, l if l in LANG_CONFIG else "")


# ---------------------------------------------------------------- span

def fix_span(text: str, start: int, end: int, lemma: str, window: int = 80) -> tuple[int, int]:
    """Si el span tiene espacios en los bordes, está desplazado: buscamos la forma de la palabra.

    Candidatos: palabras que empiezan como el lema (3 primeras letras, sin mayúsculas);
    se elige la más cercana al start original. Sin candidato: se recorta el span.
    Independiente del idioma (\\w reconoce cualquier alfabeto).
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
    s = start + (len(span) - len(span.lstrip()))
    e = end - (len(span) - len(span.rstrip()))
    return (s, e) if e > s else (start, end)


# ---------------------------------------------------------------- pasos de texto
# Todos tienen la firma (texto, cfg) -> texto.

def _dehyphen(t: str, cfg: dict) -> str:
    """"¬" = guion de fin de línea del OCR: siempre une. "mån¬ ga" → "många"."""
    return re.sub(r"(\w)¬\s*", r"\1", t)


def _join_hyph(t: str, cfg: dict) -> str:
    """"accep- terat" → "accepterat", salvo elipsis de compuesto ("upp- och") o mayúscula."""
    conj = cfg.get("conj", set())

    def repl(m: re.Match) -> str:
        nxt = m.group(2)
        if nxt.lower() in conj or nxt[:1].isupper():
            return m.group(0)
        return m.group(1) + nxt
    return re.sub(r"(\w)- (\w+)", repl, t)


def _attach_right(t: str, i: int) -> str:
    """Quita los espacios justo DESPUÉS de la posición i (comilla que abre)."""
    j = i + 1
    while j < len(t) and t[j] == " ":
        j += 1
    return t[:i + 1] + t[j:]


def _attach_left(t: str, i: int) -> str:
    """Quita los espacios justo ANTES de la posición i (comilla que cierra)."""
    j = i
    while j > 0 and t[j - 1] == " ":
        j -= 1
    return t[:j] + t[i:]


def _quotes(t: str, cfg: dict) -> str:
    """Comillas (lo que Moses hacía mejor que nosotros; ver compare_detok.py).

    - „ siempre abre. Si la frase usa „ … “ (estilo alemán), “ cierra.
    - Si no, “ abre, y la cierra la siguiente " recta que haya detrás (el OCR suele
      convertir la ” de cierre en ": “Donna Juanita ") o, si no hay, la siguiente “
      (en textos antiguos se usa “ … “).
    - Las " rectas que sobran se emparejan entre sí (1.ª abre, 2.ª cierra, ...) solo si
      son un número PAR. Si es impar, la frase empieza o acaba a mitad de una cita
      y no se tocan.
    Las comillas de cfg["sym_quotes"] (sueco ») no se tocan nunca.
    """
    sym = cfg.get("sym_quotes", "")
    german = "„" in t
    # 1) decidir qué hace cada comilla: lista de (posición, "R" pegar a la derecha / "L" a la izquierda)
    acts: list[tuple[int, str]] = []
    straight = [m.start() for m in re.finditer('"', t)] if '"' not in sym else []
    used: set[int] = set()
    curly_open = False                                      # ¿hay una “ abierta sin cerrar?
    for m in re.finditer("[„“]", t):
        c, i = m.group(), m.start()
        if c in sym:
            continue
        if c == "“" and not german and curly_open:          # “ … “ : la segunda cierra
            acts.append((i, "L"))
            curly_open = False
            continue
        if c == "„" or not german:
            acts.append((i, "R"))
            if c == "“":                                    # buscar su cierre recto
                nxt = [p for p in straight if p > i and p not in used]
                nxt_curly = [q.start() for q in re.finditer("“", t) if q.start() > i]
                if nxt and (not nxt_curly or nxt[0] < nxt_curly[0]):
                    used.add(nxt[0])
                    acts.append((nxt[0], "L"))
                else:
                    curly_open = True
        else:                                               # “ que cierra un „
            acts.append((i, "L"))
    rest = [p for p in straight if p not in used]
    if len(rest) % 2 == 0:
        acts += [(p, "R" if k % 2 == 0 else "L") for k, p in enumerate(rest)]
    # 2) aplicar de derecha a izquierda para que las posiciones no se muevan
    for i, side in sorted(acts, reverse=True):
        t = _attach_right(t, i) if side == "R" else _attach_left(t, i)
    return t


def _detok(t: str, cfg: dict) -> str:
    """Quita los espacios que puso la tokenización alrededor de la puntuación."""
    sym = cfg.get("sym_quotes", "")
    closing = "".join(c for c in _CLOSE if c not in sym)
    opening = "".join(c for c in _OPEN + cfg.get("open_extra", "") if c not in sym)
    t = re.sub(rf" +([{re.escape(closing)}])", r"\1", t)      # "palabra ," → "palabra,"
    t = re.sub(rf"([{re.escape(opening)}]) +", r"\1", t)      # "( palabra" → "(palabra"
    t = _quotes(t, cfg)
    if cfg.get("contractions"):                               # inglés, estilo Penn Treebank
        t = re.sub(r"(\w) +(n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1\2", t, flags=re.IGNORECASE)
    if cfg.get("elision"):                                    # italiano: l' uomo → l'uomo
        # solo formas elididas cortas; "po'" (= poco) va seguido de espacio y no se toca
        t = re.sub(rf"\b(?!po')(\w{{1,6}}') +(?=[\w{L}])", r"\1", t, flags=re.IGNORECASE)
    return t


_MOSES_CACHE: dict[str, object] = {}


def _detok_moses(t: str, cfg: dict) -> str:
    """Alternativa a _detok: el destokenizador de Moses (paquete sacremoses).

    Se usa para comparar con nuestras reglas (scripts/compare_detok.py). Necesita
    `pip install sacremoses`. Los centinelas van pegados a la palabra objetivo, así
    que viajan dentro de su token.
    """
    try:
        from sacremoses import MosesDetokenizer
    except ImportError as e:
        raise ImportError("detok_moses necesita sacremoses: pip install sacremoses") from e
    lang = cfg.get("_lang") or "en"
    if lang not in _MOSES_CACHE:
        _MOSES_CACHE[lang] = MosesDetokenizer(lang=lang)
    return _MOSES_CACHE[lang].detokenize(t.split(), return_str=True, unescape=False)


def _ws(t: str, cfg: dict) -> str:
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- módulos históricos por idioma

_RU_PREREFORM = str.maketrans({"ѣ": "е", "Ѣ": "Е", "і": "и", "І": "И",
                               "ѳ": "ф", "Ѳ": "Ф", "ѵ": "и", "Ѵ": "И"})


def _ru_prereform(t: str, cfg: dict) -> str:
    """Ortografía rusa anterior a la reforma de 1918 → moderna (solo reglas seguras).

    миръ → мир, вѣра → вера, Россія → Россия, ѳеатръ → феатр.
    No toca terminaciones como -аго/-яго (новаго → нового): son frecuentes pero
    tienen excepciones; probarlo aparte si los datos lo piden.
    """
    t = t.translate(_RU_PREREFORM)
    return re.sub(r"(?<=[бвгджзклмнпрстфхцчшщ])ъ\b", "", t, flags=re.IGNORECASE)


# nombre → (función, idiomas en los que se aplica; None = todos)
STEPS = {
    "dehyphen": (_dehyphen, None),
    "join_hyph": (_join_hyph, None),
    "detok": (_detok, None),
    "detok_moses": (_detok_moses, None),   # alternativa a detok (no usar los dos a la vez)
    "ws": (_ws, None),
    "ru_prereform": (_ru_prereform, {"ru"}),
}
ORDER = ["fix_span", "dehyphen", "join_hyph", "ru_prereform", "detok", "detok_moses", "ws"]

CORE_STEPS = ["fix_span", "dehyphen", "detok", "ws"]
HIST_STEPS = CORE_STEPS + ["ru_prereform"]
ALL_STEPS = [s for s in ORDER if s != "detok_moses"]
PRESETS = {"none": [], "core": CORE_STEPS, "hist": HIST_STEPS, "all": ALL_STEPS}


# ---------------------------------------------------------------- aplicar

def transform_text(text: str, start: int, end: int, steps: list[str],
                   lang: str | None = None) -> tuple[str, int, int]:
    """Aplica los pasos de texto conservando la posición de la palabra objetivo.

    Se rodea la palabra con centinelas, se transforma el texto completo y se
    recuperan start/end buscando los centinelas.
    """
    key = lang_key(lang)
    cfg = {**LANG_CONFIG.get(key, _DEFAULT_CFG), "_lang": key}
    t = f"{text[:start]}{L}{text[start:end]}{R}{text[end:]}"
    for name in ORDER:                       # orden fijo, sea cual sea el de `steps`
        if name not in steps or name not in STEPS:
            continue
        fn, langs = STEPS[name]
        if langs is None or key in langs:
            t = fn(t, cfg)
    s = t.index(L)
    t = t.replace(L, "", 1)
    e = t.index(R)
    t = t.replace(R, "", 1)
    return t, s, e


def clean_definition(d: str, lang: str | None = None, steps: list[str] | None = None) -> str:
    """Definición: quita caracteres invisibles, NFC, espacios; y la ortografía histórica si toca."""
    d = unicodedata.normalize("NFC", str(d))
    d = re.sub(r"[­​-‍﻿]", "", d)
    if steps and "ru_prereform" in steps and lang_key(lang) == "ru":
        d = _ru_prereform(d, {})
    return _ws(d, {})


def resolve_steps(steps: list[str] | str | None) -> list[str]:
    """None → core; "core"/"hist"/"all"/"none" → preset; lista → la lista (validada)."""
    if steps is None:
        return CORE_STEPS
    if isinstance(steps, str):
        steps = [steps]
    if len(steps) == 1 and steps[0] in PRESETS:
        return PRESETS[steps[0]]
    unknown = set(steps) - set(ORDER)
    if {"detok", "detok_moses"} <= set(steps):
        raise ValueError("detok y detok_moses son alternativas: usa solo uno")
    if unknown:
        raise ValueError(f"Pasos desconocidos: {sorted(unknown)}. Válidos: {ORDER} o {list(PRESETS)}")
    return list(steps)


def preprocess_usages(df: pd.DataFrame, steps: list[str] | str | None = None) -> pd.DataFrame:
    """Copia con text/start/end/target limpios y el original en text_raw.

    df: salida de semeval27.data.load_language (word, text, start, end, lang, ...).
    steps: preset ("core", "hist", "all", "none"), lista de pasos, o None (= core).
    El idioma de cada fila sale de la columna `lang` (SV, EN, ...).
    """
    steps = resolve_steps(steps)
    out = df.copy()
    out["text_raw"] = out["text"]
    langs = out["lang"] if "lang" in out else pd.Series([None] * len(out), index=out.index)
    texts, starts, ends = [], [], []
    for text, s, e, w, lang in zip(out["text"], out["start"].astype(int), out["end"].astype(int),
                                   out["word"], langs):
        if "fix_span" in steps:
            s, e = fix_span(text, s, e, w)
        text, s, e = transform_text(text, s, e, steps, lang)
        texts.append(text); starts.append(s); ends.append(e)
    out["text"], out["start"], out["end"] = texts, starts, ends
    out["target"] = [t[s:e] for t, s, e in zip(out["text"], out["start"], out["end"])]
    if steps and "definition" in out:   # steps=[] → datos tal cual (baseline)
        out["definition"] = [clean_definition(d, l, steps) if pd.notna(d) else d
                             for d, l in zip(out["definition"], langs)]
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
