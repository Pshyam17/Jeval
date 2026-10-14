from __future__ import annotations

import random
from typing import List, Tuple

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.ingest.base import Segment


class PairGenerator:
    """Generate positive/negative pairs for JEPA training."""

    def __init__(self, encoder: FrozenEncoder):
        self.encoder = encoder

    def generate_pairs(
        self, segments: List[Segment], num_pairs: int = 1000
    ) -> List[Tuple[str, str, int]]:
        """Generate (anchor, target, label) pairs where label=1 for positive, 0 for negative."""
        pairs = []
        embeddings = []
        for seg in segments:
            emb = self.encoder.encode(seg.text)
            embeddings.append((seg.text, emb))

        for _ in range(num_pairs // 2):
            # Positive pair: similar segments
            anchor_idx = random.randint(0, len(embeddings) - 1)
            anchor_text, anchor_emb = embeddings[anchor_idx]

            # Find most similar
            similarities = [
                (self.encoder.cosine_sim(anchor_emb, emb), i)
                for i, (_, emb) in enumerate(embeddings)
            ]
            similarities.sort(reverse=True)
            pos_idx = similarities[1][1]  # Skip self
            pos_text = embeddings[pos_idx][0]

            pairs.append((anchor_text, pos_text, 1))

            # Negative pair: dissimilar
            neg_idx = random.choice(
                [i for i in range(len(embeddings)) if i != anchor_idx]
            )
            neg_text = embeddings[neg_idx][0]
            pairs.append((anchor_text, neg_text, 0))

        return pairs[:num_pairs]