# Pipeline Subtask 2 (SemEval-2027 Task 3)

```
semeval27/            código reutilizable (se importa desde los scripts)
  data.py             carga los datos oficiales (usages, definitions, subtask1/2) en un DataFrame
  pairs.py            usos → pares "definición<TAB>uso" (formato TartuNLP)
  metrics.py          F1 por palabra, F1 global, barrido de umbral, baseline "siempre 1"
scripts/
  convert_data.py     paso 1: data/dev → outputs/pairs/<LANG>.pairs.tsv (+ .jsonl para entrenar)
  predict_tartunlp.py paso 2: añade la columna prob con el adaptador de TartuNLP
  evaluate_subtask2.py paso 3: métricas → results/
  make_submission.py  paso 4: prob → <LANG>/subtask2.jsonl + .zip para Codabench
notebooks/
  02_tartunlp_dev_subtask2.ipynb   los tres pasos en Colab
```

## Uso

```bash
python scripts/convert_data.py --data_dir data/dev --out_dir outputs/pairs
/content/venv/bin/python scripts/predict_tartunlp.py --pairs outputs/pairs/SV.pairs.tsv \
    --adapter_dir <ruta>/ru/classification --out outputs/preds/SV.tartunlp_ru.tsv
python scripts/evaluate_subtask2.py --pred outputs/preds/SV.tartunlp_ru.tsv --threshold 0.35 --sweep
python scripts/make_submission.py --pred SV=outputs/preds/SV.tartunlp_ru.tsv --threshold 0.35 \
    --data_dir data/dev --out submissions/st2_tartunlp
```

`convert_data.py` y `evaluate_subtask2.py` solo necesitan pandas y scikit-learn.
`predict_tartunlp.py` necesita el entorno de TartuNLP (torch 2.1.2, transformers 4.35.2, adapters 0.1.0).

## Datos dev (SV)

10 palabras, 1983 usos, 35,6 % positivos. Cada palabra trae **una** definición (la hipótesis) y cada uso una etiqueta 0/1.
El % de positivos varía mucho entre palabras (fröken 6 %, exportera 95 %), por eso se reporta el F1 por palabra.

Baseline "siempre 1" en dev SV: F1 macro por palabra 0.472, F1 global 0.525.

## Notas

- "Average F1": falta confirmar en Codabench si se promedia por palabra, por idioma o globalmente; se calculan ambas lecturas.
- El adaptador está entrenado en ruso (AXOLOTL-24) y se aplica a sueco sin reentrenar: transferencia cross-lingual zero-shot.
- `--mark_target` en `convert_data.py` marca la palabra con `<t> </t>`. El adaptador de TartuNLP no se entrenó así; usarlo solo al reentrenar.
- Código de TartuNLP: github.com/slowwavesleep/ancient-lang-adapters (rama `axolotl`, licencia MIT). `predict_tartunlp.py` replica su forma de cargar el adaptador y de construir la entrada.
