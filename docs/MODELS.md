# Models — Amharic Handwriting OCR

Two models are supported: a CNN baseline for isolated characters and a CRNN
with CTC loss for short text lines.

## Label vocabulary (explicit, never filesystem order)

Character labels come from an explicit, ordered Ethiopic vocabulary in
`src/vocabulary.py`. The mapping `index <-> character` is loaded from the
checkpoint's `class_to_idx` (e.g. `models/amharic_ocr_v2.pth`), which stores 237
classes in Ethiopian syllabary order (`ሀ` = 0 … `ፖ` = 236). A hard-coded
fallback list is used only when no checkpoint exists. **Directory/filesystem
order is never used for label-to-character mapping.**

## 1. CNN baseline — isolated characters (`src/models/cnn.py`)

Architecture (rewritten from the upstream AHCNN with standard `nn.Conv2d` /
`nn.BatchNorm2d` modules):

```
Input (1, 1, 28, 28)
Block1: Conv2d(1->32, 3x3) -> BN -> LeakyReLU -> MaxPool2d -> Dropout(0.1)
Block2: Conv2d(32->64, 3x3) -> BN -> LeakyReLU -> MaxPool2d -> Dropout(0.15)
Block3: Conv2d(64->128, 3x3) -> BN -> LeakyReLU -> MaxPool2d -> Dropout(0.20)
Block4: Conv2d(128->256, 3x3) -> BN -> LeakyReLU -> Dropout(0.25)
GlobalAvgPool -> Linear(256 -> num_classes)
```

- Checkpoint compatibility: the upstream state-dict keys (`blockN.weight`,
  `blockN.bias`) are remapped to this module's keys (`blockN.conv.weight`,
  `blockN.conv.bias`) in `load_checkpoint`, so the supplied checkpoint loads
  verbatim.
- Supplied checkpoint: `models/amharic_ocr_v2.pth` (copied from source clone;
  upstream recorded `final_accuracy: 85.62`, an upstream number not
  recomputed here).
- Configurable via `config.yaml`: `models.cnn_checkpoint`,
  `preprocessing.img_height/img_width`, `preprocessing.mean/std`.

## 2. CRNN — text lines with CTC (`src/models/crnn.py`)

Architecture:

```
CNN feature extractor (same 4 conv blocks as the baseline)
-> reshape columns as time steps
-> bidirectional LSTM (2 layers, hidden 256)
-> Linear(hidden*2 -> num_classes + 1)
-> log_softmax over extended alphabet (index 0 = CTC blank)
```

- Loss: `nn.CTCLoss(blank=0, zero_infinity=True)`.
- Decoding: greedy best-path (`greedy_decode`) and prefix beam search
  (`beam_search_decode`, no language model).
- Connected handwriting: the line is fed whole to the CRNN; characters are NOT
  cut apart, since segmentation of cursive Amharic is unreliable.
- No pretrained CRNN checkpoint is supplied. Train once labeled line data is
  available (see docs/DATASET.md).

## 3. Checkpoint format

Checkpoints saved by `scripts/train_cnn.py` contain:
- `model_state_dict`
- `class_to_idx` (char -> index, explicit vocabulary)
- `input_shape`: (1, H, W)
- `model_architecture`
- `final_accuracy` (from the real validation run)
- `val_split`, `split_strategy`, `writer_separated`, `epoch`

Model choice and checkpoint paths are configurable in `config.yaml` under
`models.cnn_checkpoint` / `models.crnn_checkpoint`, and the API exposes model
readiness via `/health`.
