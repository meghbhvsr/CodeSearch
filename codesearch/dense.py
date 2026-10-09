"""Bi-encoder retrieval with sentence-transformers.

Works for the zero-shot base model (`microsoft/unixcoder-base`; sentence-transformers adds
mean pooling) and for any fine-tuned checkpoint saved with `model.save(...)` in Phase 2.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer


class DenseEncoder:
    def __init__(self, model_name_or_path: str, device: str | None = None, fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        if fp16 and self.device == "cuda":
            self.model.half()
        self.show_progress = True

    def encode(self, texts: list[str], max_len: int, batch_size: int = 128) -> torch.Tensor:
        """L2-normalized embeddings kept on the encoder's device, so cosine = dot product."""
        self.model.max_seq_length = max_len
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )
        return emb.float()  # score in fp32 so near-ties don't collapse
