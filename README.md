# Amharic Handwriting OCR

OCR-based recognition of handwritten Amharic (Ethiopic) characters and short
text lines, returning editable Amharic Unicode text through a web application.

**Scope (initial deliverable):** scanned/photographed images of handwritten
Amharic are preprocessed, split into text lines when needed, and recognized
with either a CNN baseline (isolated characters) or a CRNN with CTC loss
(text lines). Results are editable, copyable Amharic Unicode text.

**Not included:** interactive 3D neural-network visualization, Grad-CAM
explainability, demo/auto-pilot features, or any fabricated dataset, labels,
checkpoint, or training results.

```
C:\Projects\Amharic-Handwriting-OCR
├── README.md
├── setup.ps1          # PowerShell: create venv + install dependencies
├── run.ps1            # PowerShell: start API + serve frontend
├── requirements.txt
├── config.yaml        # paths, image size, normalization, training hyperparams
├── .gitignore
│
├── data/
│   ├── uni_dataset/        # 237 Amharic character classes, 37,652 images
│   │                       # (copied from source clone; not moved/deleted there)
│   ├── lines/              # (you add labeled line images here)
│   ├── lines.jsonl         # (you create: path<TAB>Amharic text)
│   └── .gitkeep
│
├── models/
│   ├── amharic_ocr_v2.pth   # supplied CNN checkpoint (copied from source clone)
│   └── .gitkeep
│
├── docs/
│   ├── DATASET.md   # annotation format + collection/labeling instructions
│   ├── MODELS.md    # CNN baseline, CRNN+CTC, checkpoint format, config
│   └── LIMITS.md    # scope and limitations
│
├── src/
│   ├── __init__.py
│   ├── config.py            # loads config.yaml; all paths configurable
│   ├── vocabulary.py        # explicit Ethiopic vocab (never filesystem order)
│   ├── preprocessing.py     # grayscale, denoise, threshold, deskew, crop, normalize,
│   │                         # line-region detection; identical train/inference
│   ├── augmentation.py      # rotation, scale, shear, brightness, elastic deformation
│   ├── dataset.py           # char-folder + line JSONL datasets, validation, splits
│   ├── metrics.py           # acc/prec/rec/F1/CM, CER/WER (computed only from real runs)
│   ├── inference.py         # OCRPredictor: char mode / line mode / graceful no-model
│   ├── train_cnn.py         # (moved to scripts/)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py           # AHCNN baseline (checkpoint-compatible)
│   │   └── crnn.py          # CRNN + CTC loss, greedy/beam decode
│   └── api/
│       ├── __init__.py
│       └── app.py           # FastAPI: /health, /ocr, validation, errors
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_dataset.py   # dataset health + train/val split manifest
│   ├── train_cnn.py         # train CNN baseline, report real val metrics
│   ├── train_crnn.py        # validates line dataset (train loop TBD with your data)
│   └── evaluate.py          # real accuracy/precision/recall/F1 + confusion matrix
│
├── frontend/
│   ├── index.html   # dependency-free SPA (no Three.js / 3D visualization)
│   ├── style.css
│   └── app.js       # upload -> /ocr -> recognized text, copy/edit, errors
│
└── tests/             # stdlib unittest suite
```

## Quick start (Windows PowerShell)

```powershell
# 1. One-time setup (creates .venv and installs dependencies)
.\setup.ps1

# 2. Start the backend + frontend
.\run.ps1

# 3. Open http://127.0.0.1:8000
```

Notes:
- This environment's `py -3` is Python 3.15 **without pip**; `setup.ps1` uses
  `python` (3.12, which has pip) when available and falls back to `py -3`.
  If you only have `py -3`, bootstrap pip before running setup.
- Do not assume `python` is on PATH outside the activated virtual environment.

## Endpoints

- `GET  /health` — service readiness and model status (CNN/CRNN ready, vocab).
- `POST /ocr` — upload an image (`file`), optional `?mode=char|line`.
  Returns recognized Amharic Unicode text, confidence, top-k predictions
  (char mode) or per-line results (line mode), and clear errors for bad files
  or missing models.
  - `?decoder=greedy|beam` — CTC decode strategy for line mode (default: `greedy`).
  - `?beam_width=N` — beam size when `decoder=beam` (default: 10).

## Training & evaluation

```powershell
# Inspect & split the dataset
python -m scripts.prepare_dataset

# Train the CNN baseline (reports real val accuracy/precision/recall/F1)
python -m scripts.train_cnn

# Evaluate a checkpoint (default: models/amharic_ocr_v2_trained.pth)
python -m scripts.evaluate
```

The CRNN requires labeled line data you supply (`data/lines.jsonl`, see
`docs/DATASET.md`); `scripts/train_crnn.py` validates that data and explains
what is missing until you provide it.

## Tests

```powershell
python -m unittest discover -s tests -v
# or with pytest:
python -m pytest tests -q
```

### Linting

```powershell
python -m pip install -r requirements-dev.txt
`ruff check .`
```

### Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check .` and the test
suite on every push and pull request to `main` and `amharic-ocr-upgrades`.

## Data & models you still need to supply

- **CNN checkpoint:** `models/amharic_ocr_v2.pth` is supplied (copied from the
  source clone; upstream recorded accuracy 85.62%, not recomputed here).
- **CRNN checkpoint + labeled line data:** NOT supplied. Collect line images
  and `data/lines.jsonl` per `docs/DATASET.md`, then train the CRNN.
- Any claimed accuracy/CER/WER figures must come from a real run of
  `scripts/evaluate.py` (or your own CRNN evaluation); this project does not
  fabricate metrics.

## Source

This project was adapted from `BemnetMussa/DeepLearning---Amharic-handwriting-detection-Project`.
The source clone is left unchanged; only the relevant parts (dataset,
checkpoint, classifier, preprocessing, API ideas) were copied and reshaped
around this OCR proposal. The 3D visualization, Grad-CAM, Streamlit UI, and
demo features were deliberately excluded.
