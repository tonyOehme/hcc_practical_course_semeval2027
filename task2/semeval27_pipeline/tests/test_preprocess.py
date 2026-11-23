"""Tests del pipeline de entrada. Ejecutar con:  python -m pytest tests

Los casos de sueco salen del dev real. Los de los demás idiomas son ejemplos de
las convenciones de tokenización conocidas: cuando tengamos sus datos, sustituirlos
por frases reales.
"""
import pandas as pd
import pytest

from semeval27.preprocess import (clean_definition, fix_span, lang_key, preprocess_usages,
                                  resolve_steps, transform_text)


def check(text, target, steps, lang, expected):
    """Aplica los pasos y comprueba el texto y que la palabra objetivo sigue bien marcada."""
    start = text.index(target)
    t, s, e = transform_text(text, start, start + len(target), steps, lang)
    assert t == expected
    return t[s:e]


# ------------------------------------------------------------------ núcleo (sueco, dev real)

def test_detok_keeps_target():
    assert check("en skär rodnad , och mer", "skär", ["detok"], "SV",
                 "en skär rodnad, och mer") == "skär"


def test_target_next_to_punctuation():
    assert check("hon var skär .", "skär", ["detok"], "SV", "hon var skär.") == "skär"


def test_fix_span_shifted_offset():
    text = "anmärkte hon ljuft , och derpå spred sig en skär rodnad"
    s, e = fix_span(text, 8, 12, "skär")          # ' hon' en vez de 'skär'
    assert text[s:e] == "skär"


def test_fix_span_leaves_good_spans_alone():
    assert fix_span("hon sög på karamellen", 4, 7, "suga") == (4, 7)   # forma irregular


def test_dehyphen_ocr():
    assert check("det var mån¬ ga kvinnor", "kvinnor", ["dehyphen"], "SV",
                 "det var många kvinnor") == "kvinnor"


def test_join_hyph_keeps_compound_ellipsis_sv():
    check("accep- terat upp- och nedgång", "nedgång", ["join_hyph"], "SV",
          "accepterat upp- och nedgång")


def test_sentinel_limitation_documented():
    # Limitación conocida: si la palabra objetivo es justo la mitad tras el guion,
    # el centinela impide unirla. Se deja así a propósito (la palabra sigue bien marcada).
    assert check("accep- terat", "terat", ["join_hyph"], "SV", "accep- terat") == "terat"


def test_clean_definition_soft_hyphen():
    assert clean_definition("fram­ställning  inom") == "framställning inom"


def test_no_steps_is_identity():
    df = pd.DataFrame({"word": ["skär"], "text": ["en skär  dag ,"], "start": [3], "end": [7],
                       "definition": ["fram­ställning"], "lang": ["SV"]})
    out = preprocess_usages(df, [])
    assert out["text"][0] == df["text"][0] and out["definition"][0] == df["definition"][0]


# ------------------------------------------------------------------ otros idiomas

def test_en_contractions():
    assert check("I do n't think it 's a gay party .", "gay", ["detok"], "EN",
                 "I don't think it's a gay party.") == "gay"


def test_en_rules_not_applied_to_sv():
    # la regla de contracciones es solo del inglés
    t, _, _ = transform_text("ett do n't här", 0, 3, ["detok"], "SV")
    assert t == "ett do n't här"


def test_it_elision():
    assert check("dell' anno e l' uomo , un po' di pane", "uomo", ["detok"], "IT",
                 "dell'anno e l'uomo, un po' di pane") == "uomo"


def test_it_elision_target_right_after_apostrophe():
    assert check("l' uomo", "uomo", ["detok"], "IT", "l'uomo") == "uomo"


def test_es_opening_marks():
    assert check("¿ Qué es la ciencia ? ¡ Nada !", "ciencia", ["detok"], "ES",
                 "¿Qué es la ciencia? ¡Nada!") == "ciencia"


def test_ru_guillemets():
    assert check("он сказал « мир » .", "мир", ["detok"], "RU", "он сказал «мир».") == "мир"


def test_ru_prereform():
    assert check("Миръ и вѣра въ Россіи", "Миръ", ["ru_prereform"], "RU",
                 "Мир и вера в России") == "Мир"


def test_ru_prereform_only_for_russian():
    t, _, _ = transform_text("миръ", 0, 4, ["ru_prereform"], "SV")
    assert t == "миръ"


def test_ru_prereform_in_definition():
    assert clean_definition("Миръ", "RU", ["ru_prereform"]) == "Мир"


def test_join_hyph_conjunctions_per_language():
    check("radio- e televisione", "radio", ["join_hyph"], "IT", "radio- e televisione")
    check("radio- en televisie", "radio", ["join_hyph"], "NL", "radio- en televisie")


# ------------------------------------------------------------------ utilidades

def test_lang_key():
    assert lang_key("SV") == lang_key("swe") == "sv"
    assert lang_key("XX") == ""


def test_resolve_steps():
    assert resolve_steps("none") == []
    assert "ru_prereform" in resolve_steps(["hist"])
    with pytest.raises(ValueError):
        resolve_steps(["no_existe"])


# ------------------------------------------------------------------ Moses (opcional)

def test_detok_and_detok_moses_are_exclusive():
    with pytest.raises(ValueError):
        resolve_steps(["detok", "detok_moses"])


def test_detok_moses_keeps_target():
    # exc_type=ImportError: saltar también si sacremoses está instalado pero no carga
    # (p. ej. Windows bloquea la DLL de su dependencia `regex`)
    pytest.importorskip("sacremoses", exc_type=ImportError)
    t, s, e = transform_text("en skär rodnad , och mer .", 3, 7, ["detok_moses"], "SV")
    assert t[s:e] == "skär" and " ," not in t


def test_sv_symmetric_quotes_untouched():
    # dev real: en sueco » abre y cierra; no se mueve (sí la coma)
    assert check("tvifla på att » doktor » Lee , och", "Lee", ["detok"], "SV",
                 "tvifla på att » doktor » Lee, och") == "Lee"


# ------------------------------------------------------------------ comillas (dev SV real)

def test_straight_quotes_paired():
    assert check('En ny upplaga af " Orleanska jungfrun " , fru', "fru", ["detok"], "SV",
                 'En ny upplaga af "Orleanska jungfrun", fru') == "fru"


def test_straight_quotes_odd_count_untouched():
    # impar: la cita empezó antes del fragmento → no se toca
    assert check('anlete " . Hon var skär', "skär", ["detok"], "SV",
                 'anlete ". Hon var skär') == "skär"


def test_curly_opening_quote():
    assert check("uppsats om “ Familjeband och familjeplikter", "Familjeband", ["detok"], "SV",
                 "uppsats om “Familjeband och familjeplikter") == "Familjeband"


def test_german_style_quotes():
    assert check("fallit i „ mordarnes “ händer", "händer", ["detok"], "SV",
                 "fallit i „mordarnes“ händer") == "händer"


def test_quote_next_to_target():
    assert check('kallad " skär " av alla', "skär", ["detok"], "SV",
                 'kallad "skär" av alla') == "skär"


def test_curly_open_straight_close_ocr():
    # dev real: la ” de cierre llega como " recta
    assert check('operetten “ Donna Juanita " , hvari fröken', "fröken", ["detok"], "SV",
                 'operetten “Donna Juanita", hvari fröken') == "fröken"


def test_curly_quote_used_for_both():
    # dev real: “ … “ (la misma comilla abre y cierra)
    assert check("ett “ orkesterstycke i marschform “ af icke", "af", ["detok"], "SV",
                 "ett “orkesterstycke i marschform“ af icke") == "af"
