"""Tests del pipeline de entrada. Ejecutar con:  python -m pytest tests"""
import pandas as pd

from semeval27.preprocess import fix_span, preprocess_usages, transform_text, clean_definition


def test_detok_keeps_target():
    t, s, e = transform_text("en skär rodnad , och mer", 3, 7, ["detok"])
    assert t == "en skär rodnad, och mer" and t[s:e] == "skär"


def test_target_next_to_punctuation():
    t, s, e = transform_text("hon var skär .", 8, 12, ["detok"])
    assert t == "hon var skär." and t[s:e] == "skär"


def test_fix_span_shifted_offset():
    text = "anmärkte hon ljuft , och derpå spred sig en skär rodnad"
    s, e = fix_span(text, 8, 12, "skär")          # ' hon' en vez de 'skär'
    assert text[s:e] == "skär"


def test_fix_span_leaves_good_spans_alone():
    text = "hon sög på karamellen"                   # forma irregular: no tocar
    assert fix_span(text, 4, 7, "suga") == (4, 7)


def test_dehyphen_ocr():
    t, s, e = transform_text("det var mån¬ ga kvinnor", 16, 23, ["dehyphen"])
    assert t == "det var många kvinnor" and t[s:e] == "kvinnor"


def test_join_hyph_keeps_compound_ellipsis():
    t, _, _ = transform_text("accep- terat upp- och nedgång", 0, 1, ["join_hyph"])
    assert t == "accepterat upp- och nedgång"


def test_clean_definition_soft_hyphen():
    assert clean_definition("fram­ställning  inom") == "framställning inom"


def test_no_steps_is_identity():
    df = pd.DataFrame({"word": ["skär"], "text": ["en skär  dag ,"], "start": [3], "end": [7],
                       "definition": ["fram­ställning"]})
    out = preprocess_usages(df, [])
    assert out["text"][0] == df["text"][0] and out["definition"][0] == df["definition"][0]
