from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from jeval.artifacts.index import ArtifactIndex
from jeval.artifacts.detector import is_artifact
from jeval.encoders.base import Encoder
from jeval.encoders.predictor_head import PreLNTransformerPredictor
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.epe.core import EPEComputer
from jeval.epe.decomposer import EPEDecomposer
from jeval.strata.budget import BudgetAllocator, SegmentPlan
from jeval.strata.classifier import ContentClassifier
from jeval.compress.base import CompressorBackend
from jeval.compress.extractive import ExtractiveBackend
from jeval.compress.llm import LLMBackend
from jeval.ingest.base import Session, Segment


@dataclass(frozen=True)
class CompressionResult:
    compressed_text: str
    artifact_index: ArtifactIndex
    report: List[SegmentPlan]
    token_reduction: float


class AdaptiveCompressor:
    """Full pipeline that chooses budget and compresses per segment."""

    SEGMENT_SPLIT_PATTERN = re.compile(r"(?m)^(?:-\s+|\*\s+|\d+\.\s+|#+\s+)")

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        predictor: Optional[PreLNTransformerPredictor] = None,
        classifer: Optional[ContentClassifier] = None,
        backend: Optional[CompressorBackend] = None,
        artifact_index: Optional[ArtifactIndex] = None,
    ):
        self.encoder = encoder or FrozenEncoder()
        self.predictor = predictor or PreLNTransformerPredictor(self.encoder.dim())
        self.classifier = classifer or ContentClassifier()
        self.backend = backend or LLMBackend()
        self.artifact_index = artifact_index or ArtifactIndex()
        self.budget_allocator = BudgetAllocator()
        self.decomposer = EPEDecomposer()

    def _segment(self, text: str) -> List[str]:
        if not text.strip():
            return []

        splits = self.SEGMENT_SPLIT_PATTERN.split(text)
        segments = [seg.strip() for seg in splits if seg.strip()]
        if not segments:
            segments = [text.strip()]
        return segments

    def compress(self, session: Session) -> CompressionResult:
        if not isinstance(session, Session):
            raise TypeError("compress expects a Session")

        segment_texts: List[str] = []
        for segment in session:
            segment_texts.extend(self._segment(segment.text))

        if not segment_texts:
            return CompressionResult("", self.artifact_index, [], 1.0)

        embeddings = self.encoder.encode(segment_texts)

        if self.predictor is None:
            pred = embeddings
        else:
            try:
                import torch

                pred_t = self.predictor(torch.from_numpy(embeddings).float())
                pred = pred_t.detach().cpu().numpy()
            except (ModuleNotFoundError, ImportError):
                # Torch unavailable in lightweight environments; fallback to identity.
                pred = embeddings

        epe_values = EPEComputer.compute_batch(embeddings, pred)
        z_scores = EPEComputer.z_scores(list(epe_values))

        compressed_segments = []
        plans: List[SegmentPlan] = []

        for idx, seg in enumerate(segment_texts):
            content_type = self.classifier.top_label(seg)
            artifact_flag = is_artifact(seg)
            confidence = self.classifier.classify(seg).get(content_type, 0.0)
            plan = self.budget_allocator.allocate(
                segment_text=seg,
                epe=float(epe_values[idx]),
                z_score=z_scores[idx],
                content_type=content_type.split(",")[0].upper(),
                artifact_override=artifact_flag,
                confidence=confidence,
            )
            plans.append(plan)

            self.artifact_index.update_from_text(seg, turn=idx, epe=float(epe_values[idx]))
            segment_texts[idx] = seg

            budget = plan.budget
            compressed_seg = self.backend.compress(seg, budget)
            compressed_segments.append(compressed_seg)
            

        compressed_text = "\n".join(compressed_segments)
        token_reduction = 1.0 - (len(compressed_text.split()) / max(1, len(" ".join(segment_texts).split())))

        # Ensure at least min bound 0.0 to 1.0
        token_reduction = max(0.0, min(1.0, token_reduction))

        return CompressionResult(compressed_text, self.artifact_index, plans, token_reduction)
    
