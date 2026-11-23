#!/usr/bin/env bash
# Prepara el entorno en Colab o Kaggle. Se puede ejecutar varias veces: si ya existe, no reinstala nada.
#
#   bash scripts/colab_setup.sh            -> núcleo + entorno TartuNLP (GPU)
#   bash scripts/colab_setup.sh --core     -> solo el núcleo (análisis en CPU, más rápido)
#
# Crea /content/venv con Python 3.11 (torch 2.1.2 no existe para el 3.12 de Colab)
# e instala este paquete en modo editable. Después, ejecuta los scripts con:
#   /content/venv/bin/python scripts/...
set -euo pipefail

VENV=${VENV:-/content/venv}
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA="[tartunlp]"
[[ "${1:-}" == "--core" ]] && EXTRA=""

command -v uv >/dev/null || pip install -q uv

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creando $VENV (Python 3.11)..."
  uv venv -q --python 3.11 "$VENV"
fi

echo "Instalando semeval27$EXTRA desde $PKG_DIR ..."
uv pip install -q --python "$VENV/bin/python" -e "$PKG_DIR$EXTRA"

"$VENV/bin/python" - <<'EOF'
import semeval27, pandas
msg = f"OK: semeval27 + pandas {pandas.__version__}"
try:
    import torch, transformers, adapters
    msg += f" | torch {torch.__version__}, transformers {transformers.__version__}, GPU: {torch.cuda.is_available()}"
except ImportError:
    msg += " | (sin entorno TartuNLP)"
print(msg)
EOF
