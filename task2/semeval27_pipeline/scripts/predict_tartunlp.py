"""Paso 2: puntuar cada par (definición, uso) con el adaptador de TartuNLP.

Se ejecuta con el Python del entorno de TartuNLP (adapters==0.1.0), p. ej.:
    /content/venv/bin/python scripts/predict_tartunlp.py \
        --pairs outputs/pairs/SV.pairs.tsv \
        --adapter_dir /content/drive/MyDrive/axolotl_models/ru/classification \
        --out outputs/preds/SV.tartunlp_ru.tsv

Añade una columna `prob` = P(la definición describe el uso). La carga del
modelo y la entrada "definición<TAB>uso" son las mismas que en predict_sense.py.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from adapters import AdapterConfig, AutoAdapterModel
from tqdm import trange
from transformers import AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="<LANG>.pairs.tsv de convert_data.py")
    ap.add_argument("--adapter_dir", required=True,
                    help="carpeta del adaptador, p. ej. trained_models/ru/classification")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="xlm-roberta-base")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No encuentro adapter_config.json en {adapter_dir}. "
                         "Debe ser la subcarpeta 'classification' del modelo entrenado.")

    device = pick_device()
    print("dispositivo:", device)
    model = AutoAdapterModel.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = AdapterConfig.load((adapter_dir / "adapter_config.json").as_posix())
    model.load_adapter(adapter_dir.as_posix(), config=config)
    model.set_active_adapters("classification")
    model.eval()
    model.to(device)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype={"sentence_id": str})
    texts = pairs["pair_text"].tolist()
    probs = []
    for i in trange(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], return_tensors="pt",
                          padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())

    pairs["prob"] = probs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, sep="\t", index=False)
    print(f"{len(pairs)} pares puntuados → {args.out}")


if __name__ == "__main__":
    main()
