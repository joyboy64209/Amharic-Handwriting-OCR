# Dataset — Amharic Handwriting OCR

This document describes the two data flavors used by the project, how the
existing data was obtained, and how to collect and label additional samples.

## 1. Isolated-character dataset (CNN baseline)

**Location:** `data/uni_dataset/`

One folder per Amharic character class; the folder name *is* the Ethiopic
Unicode character (e.g. `data/uni_dataset/ሀ/001he.1.jpg`). Images are grayscale
28×28 JPEG. This dataset was **copied from the source clone**
(`BemnetMussa/DeepLearning---Amharic-handwriting-detection-Project`) and is
included in this project (37,652 images across 237 classes, ~37 MB).

It is **not** fabricated: it is the actual data present in the upstream clone.
The upstream README claimed an advertised dataset/trained checkpoint; we
confirmed both the dataset folder and the checkpoint `models/amharic_ocr_v2.pth`
are present in the clone and have been copied here.

### Writer identity
Many filenames begin with a numeric prefix (e.g. `001he.1.jpg`). The prefix is
treated as a writer ID so that train/validation splits can be **writer-separated**
(no writer appears in both splits), avoiding data leakage.

## 2. Text-line dataset (CRNN + CTC)

**Location:** `data/lines/` + `data/lines.jsonl`

This dataset is **not included**. The upstream project does not ship labeled
text-line images, so there is nothing to copy. You must collect and label them
yourself before the CRNN can be trained or line recognition used.

### Annotation format (`data/lines.jsonl`)

One JSON-lines record per line image. Each line is two tab-separated fields:

```
<relative_image_path>\t<Amharic Unicode text>
```

Example:

```
lines/001.png	ሰላም
lines/002.png	ኢትዮጵያ
lines/003.png	ልዩ ትምህርት
```

- Field 1: path to the line image, relative to `data/lines/` (or an absolute
  path).
- Field 2: the ground-truth Amharic Unicode text for that line.
- Encoding: UTF-8.

### Collecting and labeling instructions

1. Scan or photograph handwritten Amharic lines on paper. Prefer clean,
   well-lit, straight-on captures; skewed or folded pages hurt line detection.
2. Crop each image to a single text line and save it into `data/lines/`.
3. Transcribe the line exactly as written, using standard Ethiopic Unicode
   characters (do not transliterate to Latin).
4. Append one `path\ttext` record per line to `data/lines.jsonl`.
5. Run `python -m scripts.prepare_dataset` to validate, then
   `python -m scripts.train_crnn` to train the CRNN (wiring still to be
   completed once labeled data exists — see scripts/train_crnn.py).

## 3. Validation tool

`python -m scripts.prepare_dataset` inspects both datasets and writes a split
manifest to `data/split_manifest.json`. It reports class counts, per-class
sample counts, missing classes, and whether writer-separated splitting is
possible.

## 4. What is NOT provided

- No synthetic or fabricated labels.
- No pretrained CRNN checkpoint.
- No pretrained CNN beyond the upstream `amharic_ocr_v2.pth` (recorded upstream
  accuracy 85.62% — an upstream number, not recomputed by us).
