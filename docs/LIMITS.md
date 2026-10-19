# Limits — Amharic Handwriting OCR

## Initial deliverable scope

- Handwritten Amharic images (scans or photos).
- Isolated-character recognition (CNN baseline, 28×28 input).
- Short handwritten text-line recognition (CRNN + CTC) — once a CRNN
  checkpoint and labeled line data are provided.
- Web UI + REST API (`/ocr`, `/health`) for upload, recognition, copy/edit of
  extracted Amharic Unicode text.

## Known limitations

1. **Severely degraded documents.** Faint ink, heavy noise, low resolution,
   motion blur, shadows, or uneven lighting can break preprocessing
   (binarization / deskew / line detection). Results may be missing or wrong;
   the API returns low-confidence warnings and errors rather than fabricated
   text.

2. **Highly connected handwriting.** Line-mode recognition avoids character
   segmentation, but very tightly cursive or overlapping strokes can still
   exceed what a CTC sequence model can resolve without a language model or
   larger training data.

3. **No pretrained CRNN.** The only supplied checkpoint is the isolated
   character CNN (`models/amharic_ocr_v2.pth`). Line recognition falls back to
   the CNN on the whole image unless you train a CRNN with your own labeled
   line data (docs/DATASET.md).

4. **Vocabulary coverage.** The CNN covers the 237 core Ethiopic syllabary
   classes present in the checkpoint's `class_to_idx`. Characters outside that
   set (rare variants, punctuation, numerals) are mapped to `?`.

5. **No claimed metrics unless computed.** The upstream checkpoint's recorded
   `final_accuracy` (85.62%) is an upstream number, not recomputed here. Any
   new accuracy / precision / recall / F1 / CER / WER figures are produced only
   by running `scripts/evaluate.py` on a real validation split.

6. **CPU-only baseline.** The project runs on CPU by default and will use CUDA
   only if a CUDA-capable torch and GPU are available.
