"""Explicit Amharic (Ethiopic) character vocabulary.

The mapping between class indices and Ethiopic Unicode characters is explicit
and documented here. It is **never** derived from filesystem directory order,
which is unstable across operating systems and copy operations.

Two sources can populate the vocabulary:

1. A trained checkpoint's ``class_to_idx`` mapping (preferred). The supplied
   checkpoint ``models/amharic_ocr_v2.pth`` stores 237 classes in Ethiopian
   syllabary order: ``ሀ`` = 0, ``ሁ`` = 1, … ``ፖ`` = 236.
2. A hard-coded fallback list (below) in Ethiopian order, used only when no
   checkpoint is available so the app can still describe its label space.

Public helpers:
    load_vocab(checkpoint_path) -> Vocabulary
    Vocabulary.char_for_index(idx) -> str
    Vocabulary.index_for_char(char) -> int | None
    Vocabulary.size() -> int
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Fallback vocabulary (Ethiopian syllabary order, 237 core characters).
# Used only as a last resort when a checkpoint with class_to_idx is absent.
# ---------------------------------------------------------------------------

FALLBACK_VOCAB: List[str] = [
    "ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ",
    "ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ",
    "ሐ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ",
    "መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ",
    "ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ",
    "ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ", "ሯ",
    "ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ",
    "ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ", "ሿ",
    "ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ",
    "በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ",
    "ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ",
    "ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ", "ቷ",
    "ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ",
    "ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ",
    "ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ", "ኗ",
    "ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ", "ኟ",
    "አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ", "ኧ",
    "ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ", "኷",
    "ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ", "኿",
    "ዀ", "዁", "ዂ", "ዃ",
    "ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ",
    "ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ", "዗",
    "ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ", "ዟ",
    "ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ", "ዧ",
    "የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ", "ዯ",
    "ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ", "ዷ",
    "ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ", "ጇ",
    "ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ", "ጏ",
    "ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ", "ጧ",
    "ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ", "ጯ",
    "ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ", "ጷ",
    "ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ", "ጿ",
    "ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ",
    "ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ", "ፏ",
    "ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ", "ፗ",
]


@dataclass
class Vocabulary:
    """Ordered Ethiopic character vocabulary with bidirectional lookup."""

    # index -> character
    index_to_char: Dict[int, str] = field(default_factory=dict)
    # character -> index
    char_to_index: Dict[str, int] = field(default_factory=dict)
    source: str = "fallback"  # 'checkpoint' or 'fallback'

    def __post_init__(self) -> None:
        # Rebuild char_to_index from index_to_char to guarantee consistency.
        self.char_to_index = {ch: idx for idx, ch in self.index_to_char.items()}

    @classmethod
    def from_ordered_chars(cls, chars: List[str], source: str = "fallback") -> "Vocabulary":
        index_to_char = {i: ch for i, ch in enumerate(chars)}
        return cls(index_to_char=index_to_char, source=source)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> "Vocabulary":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        class_to_idx = checkpoint.get("class_to_idx")
        if not isinstance(class_to_idx, dict) or not class_to_idx:
            raise ValueError(f"Checkpoint {checkpoint_path} has no usable 'class_to_idx'")
        # class_to_idx maps char -> index; invert to index -> char.
        index_to_char = {int(idx): str(ch) for ch, idx in class_to_idx.items()}
        return cls(index_to_char=index_to_char, source="checkpoint")

    @classmethod
    def load(cls, checkpoint_path: Optional[Path] = None) -> "Vocabulary":
        """Load from checkpoint if available, otherwise fall back."""
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                vocab = cls.from_checkpoint(checkpoint_path)
                return vocab
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[vocabulary] Could not load checkpoint vocab ({exc}); using fallback.")
        return cls.from_ordered_chars(list(FALLBACK_VOCAB), source="fallback")

    def size(self) -> int:
        return len(self.index_to_char)

    def char_for_index(self, idx: int) -> str:
        return self.index_to_char.get(idx, "?")

    def index_for_char(self, char: str) -> Optional[int]:
        return self.char_to_index.get(char)

    def to_json(self, path: Path) -> None:
        payload = {
            "source": self.source,
            "size": self.size(),
            "index_to_char": {str(k): v for k, v in sorted(self.index_to_char.items())},
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        index_to_char = {int(k): v for k, v in payload["index_to_char"].items()}
        return cls(index_to_char=index_to_char, source=payload.get("source", "json"))

    def __contains__(self, char: str) -> bool:
        return char in self.char_to_index

    def __len__(self) -> int:
        return self.size()


__all__ = ["Vocabulary", "FALLBACK_VOCAB"]