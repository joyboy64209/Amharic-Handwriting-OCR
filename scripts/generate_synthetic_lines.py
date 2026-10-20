"""Bootstrap synthetic Amharic text lines from isolated character images.

Reads random glyphs from ``data/uni_dataset/<char>/``, stitches them together
horizontally with slight random spacing and vertical jitter to form realistic
Amharic words and short sentences, and writes:

* line images to ``data/lines/line_NNNNN.png``
* annotations to ``data/lines.jsonl`` as one JSON object per line:
  ``{"path": "data/lines/line_NNNNN.png", "text": "ሰላም ..."}``

This is clearly synthetic training data (not real human handwriting) and is
intended to unblock CRNN + CTC line recognition until labeled line data is
collected per docs/DATASET.md.

Usage (PowerShell, from the project root):
    python scripts/generate_synthetic_lines.py --num-lines 1000 --img-height 64 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running both as `python scripts/generate_synthetic_lines.py` and from
# any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# A small built-in list of common Amharic words for "realistic" line text.
# Words are filtered at runtime to only those whose characters all exist in
# the character dataset; otherwise random characters are used.
# ---------------------------------------------------------------------------

AMHARIC_WORDS = [
    "ሰላም",
    "ኢትዮጵያ",
    "ቤት",
    "ልጅ",
    "ስራ",
    "ወጣት",
    "መግቢያ",
    "ገበሬ",
    "ምስራቅ",
    "ምዕራብ",
    "ሰው",
    "እናት",
    "አባት",
    "ወንድም",
    "እህት",
    "ስልክ",
    "መስኮት",
    "ምግብ",
    "ውሃ",
    "ቡና",
    "ጥበብ",
    "ትምህርት",
    "መጽሐፍ",
    "ጽሑፍ",
    "መልአክ",
    "ሁሉ",
    "አለ",
    "አለም",
    "ሀገር",
    "ጋበዝኳት",
    "ወደ",
    "እነ",
    "ይለዋል",
    "ነው",
    "እንደ",
    "በ",
    "ከ",
    "ለ",
    "እንዲ",
    "አንድ",
    "ሁለት",
    "ሶስት",
    "አራት",
    "አምስት",
]


def load_glyphs(char_dataset_dir: Path) -> dict[str, list[Path]]:
    """Map each character folder to the list of glyph image paths inside it."""
    glyphs: dict[str, list[Path]] = {}
    root = Path(char_dataset_dir)
    if not root.exists():
        return glyphs
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ]
        if images:
            glyphs[class_dir.name] = images
    return glyphs


def load_and_resize_glyph(path: Path, img_height: int) -> Image.Image:
    """Load a glyph image, ensure dark ink on light background, resize to height."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    # Ensure dark ink (low values) on light background (high values).
    if arr.mean() > 127:
        arr = 255 - arr
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    w, h = img.size
    new_h = img_height
    new_w = max(1, int(w * new_h / max(h, 1)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def glyph_to_rgba(img: Image.Image) -> Image.Image:
    """Convert a grayscale glyph to RGBA: dark ink opaque black, bg transparent."""
    arr = np.array(img.convert("L"))
    alpha = (255 - arr).astype(np.uint8)  # dark ink -> high alpha
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)  # black ink
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")



def add_minimal_noise(canvas: Image.Image, level: float, rng: random.Random) -> Image.Image:
    """Add light salt/pepper noise to the canvas background/ink."""
    if level <= 0:
        return canvas
    arr = np.array(canvas.convert("L")).astype(np.float32)
    # numpy RNG is seeded in main() for reproducibility.
    mask = np.random.random(arr.shape) < level
    noise = np.random.choice([0, 255], size=arr.shape).astype(np.float32)
    arr = np.where(mask, noise, arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def crop_to_content(canvas: Image.Image, pad: int = 6) -> Image.Image:
    """Crop canvas to the bounding box of non-white pixels, with padding."""
    arr = np.array(canvas.convert("L"))
    nonwhite = arr < 245
    if not nonwhite.any():
        return canvas
    rows = np.any(nonwhite, axis=1)
    cols = np.any(nonwhite, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(h - 1, bottom + pad)
    right = min(w - 1, right + pad)
    return canvas.crop((left, top, right + 1, bottom + 1))


def generate_text(
    rng: random.Random,
    glyphs: dict[str, list[Path]],
    min_chars: int,
    max_chars: int,
    use_words: bool,
) -> str:
    """Generate a random Amharic text string of glyphs separated by spaces.

    When ``use_words`` is True, samples from the built-in Amharic word list,
    keeping only words whose characters all exist in the glyph set. Falls back
    to random characters if no usable words remain or ``use_words`` is False.
    """
    available_chars = set(glyphs.keys())
    if not available_chars:
        return ""

    def random_char() -> str:
        return rng.choice(sorted(available_chars))

    words: list[str] = []
    if use_words:
        usable_words = [w for w in AMHARIC_WORDS if set(w) <= available_chars]
        if usable_words:
            n_words = rng.randint(2, max(2, min(6, max(1, max_chars // 2))))
            for _ in range(n_words):
                w = rng.choice(usable_words)
                if sum(len(wd) for wd in words) + len(w) + len(words) > max_chars:
                    break
                words.append(w)

    if words:
        text = " ".join(words)
        if min_chars <= len(text.replace(" ", "")) <= max_chars:
            return text

    # Fallback: random character sequence with occasional spaces.
    n_chars = rng.randint(min_chars, max_chars)
    chars: list[str] = []
    space_every = rng.randint(3, 6)  # insert a space every few chars
    for i in range(n_chars):
        if i > 0 and i % space_every == 0:
            chars.append(" ")
        else:
            chars.append(random_char())
    text = "".join(chars)
    return text


def render_line(
    text: str,
    glyphs: dict[str, list[Path]],
    img_height: int,
    min_gap: int,
    max_gap: int,
    max_jitter: int,
    noise_level: float,
    rng: random.Random,
) -> Image.Image | None:
    """Stitch the glyphs for ``text`` into a single line image.

    Spaces become wide gaps with no glyph. Returns None if nothing was rendered.
    """
    if not text.strip():
        return None

    # Estimate an upper-bound canvas width; we crop to content afterwards.
    est_width = len(text.replace(" ", "")) * (img_height + max_gap) + max_gap * 2 + 40
    canvas = Image.new("RGB", (max(est_width, 100), img_height), (255, 255, 255))

    x = 0
    placed_any = False
    for ch in text:
        if ch == " ":
            x += rng.randint(min_gap, max_gap) * 2 + 8  # wider space gap
            continue
        options = glyphs.get(ch)
        if not options:
            # Skip characters we have no glyphs for.
            continue
        path = rng.choice(options)
        glyph = load_and_resize_glyph(path, img_height)
        glyph_rgba = glyph_to_rgba(glyph)
        gw, gh = glyph.size
        y = rng.randint(-max_jitter, max_jitter)
        y = max(0, min(y, img_height - gh))
        canvas.paste(glyph_rgba, (x, y), glyph_rgba)
        x += gw + rng.randint(min_gap, max_gap)
        placed_any = True

    if not placed_any:
        return None

    canvas = crop_to_content(canvas, pad=8)
    canvas = add_minimal_noise(canvas, noise_level, rng)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Amharic text lines from isolated character images.",
    )
    parser.add_argument(
        "--char-dataset-dir",
        default="data/uni_dataset",
        help="Source folder with one sub-folder per Amharic character class.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lines",
        help="Directory to write generated line images.",
    )
    parser.add_argument(
        "--annotations-file",
        default="data/lines.jsonl",
        help="JSON Lines annotation file path.",
    )
    parser.add_argument(
        "--num-lines",
        type=int,
        default=1000,
        help="Number of synthetic lines to generate.",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=64,
        help="Target line height in pixels (glyphs scaled to this height).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=5,
        help="Minimum number of glyphs per line (excluding spaces).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=30,
        help="Maximum number of glyphs per line (excluding spaces).",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=4,
        help="Minimum horizontal gap between glyphs (pixels).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=14,
        help="Maximum horizontal gap between glyphs (pixels).",
    )
    parser.add_argument(
        "--max-jitter",
        type=int,
        default=4,
        help="Maximum vertical jitter per glyph (pixels).",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.02,
        help="Fraction of pixels perturbed by salt/pepper noise (0 = none).",
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Sample from built-in Amharic word list instead of random chars.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    glyphs = load_glyphs(Path(args.char_dataset_dir))
    if not glyphs:
        print(f"[generate_synthetic_lines] No glyphs found in {args.char_dataset_dir}")
        print("   Make sure data/uni_dataset/ contains character class folders.")
        return 1

    print(f"[generate_synthetic_lines] Loaded {len(glyphs)} character classes "
          f"from {args.char_dataset_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = Path(args.annotations_file)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with annotations_path.open("w", encoding="utf-8") as af:
        for i in range(args.num_lines):
            text = generate_text(
                rng,
                glyphs,
                args.min_chars,
                args.max_chars,
                args.words,
            )
            if not text.strip():
                skipped += 1
                continue

            line_img = render_line(
                text,
                glyphs,
                args.img_height,
                args.min_gap,
                args.max_gap,
                args.max_jitter,
                args.noise_level,
                rng,
            )
            if line_img is None:
                skipped += 1
                continue

            file_name = f"line_{i:05d}.png"
            out_path = output_dir / file_name
            # Store path relative to project root for portability.
            rel_path = f"data/lines/{file_name}"
            line_img.save(out_path)
            af.write(json.dumps({"path": rel_path, "text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"[generate_synthetic_lines] Wrote {written} line images to {output_dir}")
    print(f"[generate_synthetic_lines] Wrote {written} annotations to {annotations_path}")
    if skipped:
        print(f"[generate_synthetic_lines] Skipped {skipped} empty/unrenderable lines")
    print("[generate_synthetic_lines] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())