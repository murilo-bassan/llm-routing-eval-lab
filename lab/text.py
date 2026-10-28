from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

STOP = {"o", "a", "os", "as", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "um", "uma", "pra", "para", "por",
        "que", "me", "the", "of", "for", "in", "a", "an", "is", "to", "this", "please", "com", "se", "eu", "ta"}


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", fold(text)) if t not in STOP and len(t) > 1]


def hashed_ngrams(texts: list[str], dim: int = 512) -> np.ndarray:
    """Offline sentence vectors: signed hashed char 3/4-grams + words, L2-normalized."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        words = tokens(text)
        padded = f" {' '.join(words)} "
        feats = [padded[i:i + n] for n in (3, 4) for i in range(len(padded) - n + 1)] + [f"w:{w}" for w in words]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            out[row, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)
