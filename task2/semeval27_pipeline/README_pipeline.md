# Pipeline Subtask 2 (SemEval-2027 Task 3)

```
pyproject.toml        hace instalable el paquete semeval27 (pip install -e .)
semeval27/            código reutilizable (se importa desde scripts y notebooks)
  data.py             carga los datos oficiales (usages, definitions, subtask1/2) en un DataFrame
  pairs.py            usos → pares "definición<TAB>uso" (formato TartuNLP)
  metrics.py          F1 por palabra, F1 global, barrido de umbral, baseline "siempre 1"
scripts/
  colab_setup.sh      prepara el entorno en Colab/Kaggle (/content/venv, Python 3.11)
  convert_data.py     paso 1: datos → <LANG>.pairs.tsv (+ .jsonl para entrenar)
  predict_tartunlp.py paso 2: añade la columna prob con el adaptador de TartuNLP
  evaluate_subtask2.py paso 3: métricas → results/
  make_submission.py  paso 4: prob → <LANG>/subtask2.jsonl + .zip para Codabench
notebooks/
  00_colab_template.ipynb          plantilla para cada experimento nuevo en Colab
  02_tartunlp_dev_subtask2.ipynb   TartuNLP-ru zero-shot sobre dev SV
  03_error_analysis.ipynb          análisis de errores (CPU)
```

## Cómo se trabaja

- **El código vive en el repo.** Los notebooks solo preparan el entorno y llaman a los scripts o funciones de `semeval27`.
- **Colab/Kaggle para lo pesado (GPU)**, el PC para escribir código y lo ligero (CPU).
- **Datos, modelos y resultados en Drive** (`MyDrive/semeval27/...`), nunca en git (`data/`, `outputs/`, `trained_models/` están en `.gitignore`).
- Si arreglas algo directamente en Colab, pásalo al repo el mismo día.

### Entorno local (Windows, VS Code)

Desde `task2/semeval27_pipeline`:

```
python -m venv .venv            # solo la primera vez
.venv\Scripts\activate
pip install -e .                # núcleo: datos, pares, métricas
pip install -e ".[tartunlp]"    # opcional: torch/transformers/adapters fijados (CPU, lento)
```

En VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

### Colab / Kaggle

Copia `notebooks/00_colab_template.ipynb`. Hace `git pull`, ejecuta `bash scripts/colab_setup.sh`
y deja `PY = '/content/venv/bin/python'` para lanzar los scripts:

```bash
$PY scripts/convert_data.py --data_dir $DATA/dev --out_dir outputs/pairs
$PY scripts/predict_tartunlp.py --pairs outputs/pairs/SV.pairs.tsv \
    --adapter_dir $DRIVE/axolotl_models/ru/classification --out $DATA/preds/SV.tartunlp_ru.tsv
$PY scripts/evaluate_subtask2.py --pred $DATA/preds/SV.tartunlp_ru.tsv --threshold 0.35 --sweep
$PY scripts/make_submission.py --pred SV=$DATA/preds/SV.tartunlp_ru.tsv --threshold 0.35 \
    --data_dir $DATA/dev --out submissions/st2_tartunlp
```

¿Por qué un venv aparte en Colab? TartuNLP necesita torch 2.1.2 + transformers 4.35.2 + adapters 0.1.0,
y torch 2.1.2 no existe para el Python 3.12 de Colab.

## Datos dev (SV)

10 palabras, 1983 usos, 35,6 % positivos. Cada palabra trae **una** definición (la hipótesis) y cada uso una etiqueta 0/1.
El % de positivos varía mucho entre palabras (fröken 6 %, exportera 95 %), por eso se reporta el F1 por palabra.

Baseline "siempre 1" en dev SV: F1 macro por palabra 0.472, F1 global 0.525.

## Notas

- "Average F1": falta confirmar en Codabench si se promedia por palabra, por idioma o globalmente; se calculan ambas lecturas.
- El adaptador está entrenado en ruso (AXOLOTL-24) y se aplica a sueco sin reentrenar: transferencia cross-lingual zero-shot.
- `--mark_target` en `convert_data.py` marca la palabra con `<t> </t>`. El adaptador de TartuNLP no se entrenó así; usarlo solo al reentrenar.
- Código de TartuNLP: github.com/slowwavesleep/ancient-lang-adapters (rama `axolotl`, licencia MIT). `predict_tartunlp.py` replica su forma de cargar el adaptador y de construir la entrada.
