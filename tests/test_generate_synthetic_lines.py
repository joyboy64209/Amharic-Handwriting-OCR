"""Focused tests for scripts.generate_synthetic_lines."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import unittest

from scripts.generate_synthetic_lines import (
    load_glyphs,
    generate_text,
    render_line,
    AMHARIC_WORDS,
)


class TestGenerateSyntheticLines(unittest.TestCase):

    def test_load_glyphs(self):
        glyphs = load_glyphs(Path("data/uni_dataset"))
        self.assertGreaterEqual(len(glyphs), 237)
        self.assertIn("ሀ", glyphs)
        self.assertGreater(len(glyphs["ሀ"]), 0)

    def test_generate_text_uses_available_chars(self):
        glyphs = load_glyphs(Path("data/uni_dataset"))
        rng = random.Random(42)
        text = generate_text(rng, glyphs, min_chars=5, max_chars=20, use_words=False)
        available = set(glyphs.keys())
        for ch in text.replace(" ", ""):
            self.assertIn(ch, available)

    def test_render_line_produces_image(self):
        glyphs = load_glyphs(Path("data/uni_dataset"))
        rng = random.Random(7)
        img = render_line(
            "ሰላም ኢትዮጵያ",
            glyphs,
            img_height=64,
            min_gap=4,
            max_gap=10,
            max_jitter=4,
            noise_level=0.02,
            rng=rng,
        )
        self.assertIsNotNone(img)
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.width, 0)
        self.assertEqual(img.height, 64)

    def test_render_line_skips_unknown_chars(self):
        glyphs = load_glyphs(Path("data/uni_dataset"))
        rng = random.Random(1)
        img = render_line(
            "ሰላም XYZ ኢትዮጵያ",
            glyphs,
            img_height=64,
            min_gap=4,
            max_gap=10,
            max_jitter=4,
            noise_level=0.0,
            rng=rng,
        )
        # XYZ has no glyphs; should still render the known chars.
        self.assertIsNotNone(img)

    def test_amharic_words_nonempty(self):
        self.assertGreater(len(AMHARIC_WORDS), 0)
        self.assertIn("ሰላም", AMHARIC_WORDS)

    def test_jsonl_roundtrip(self):
        # Verify the annotations file written by the script parses as JSONL.
        annotations = Path("data/lines.jsonl")
        if not annotations.exists():
            self.skipTest("data/lines.jsonl not present; run generator first")
        count = 0
        with annotations.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                self.assertIn("path", obj)
                self.assertIn("text", obj)
                count += 1
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()