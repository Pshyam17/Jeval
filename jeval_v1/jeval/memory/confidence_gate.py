from __future__ import annotations

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.entity_extraction import extract_entity_texts


class ConfidenceGate:
    """
    Query-conditioned confidence score for hot-cache entries.

    Confidence measures how many query-relevant entities survived compression:
        confidence = |E_q ∩ E_c| / max(|E_q ∩ E_o|, 1)

    where E_q, E_c, E_o are entity sets from the query, compressed entry,
    and original entry respectively.

    Entity extraction uses spaCy NER + regex patterns when spaCy is available;
    falls back to regex-only (ALL_CAPS, file paths, step refs, numbers, error
    classes) when spaCy is absent.  Generic nouns (deployment, migration) are
    not extracted by the regex path — queries with only generic terms yield
    empty entity sets and route to cold_storage, which is the correct safe
    default when no specific technical identifiers are present.

    Routing thresholds determine whether to serve the hot cache directly,
    enrich with cold storage, or fall back to cold storage entirely.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
    ) -> None:
        self._encoder = encoder
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold

    def score(
        self,
        query: str,
        compressed_entry: str,
        original_entry: str,
    ) -> float:
        """
        Confidence = |E_q ∩ E_c| / max(|E_q ∩ E_o|, 1).

        The denominator is the count of query-relevant entities that ever
        existed in the original (the maximum recoverable by compression);
        the numerator is how many of those survived into the compressed form.
        """
        e_q = extract_entity_texts(query)
        e_c = extract_entity_texts(compressed_entry)
        e_o = extract_entity_texts(original_entry)

        query_in_original = e_q & e_o
        query_in_compressed = e_q & e_c

        return len(query_in_compressed) / max(len(query_in_original), 1)

    def route(
        self,
        query: str,
        compressed_entry: str,
        original_entry: str,
    ) -> tuple[str, float]:
        """
        Returns (routing_decision, confidence_score).

        "hot_cache"   — confidence >= high_threshold; serve compressed directly
        "enriched"    — low_threshold <= confidence < high_threshold; append cold
        "cold_storage"— confidence < low_threshold; cold fallback, record miss
        """
        confidence = self.score(query, compressed_entry, original_entry)
        if confidence >= self._high_threshold:
            return "hot_cache", confidence
        if confidence >= self._low_threshold:
            return "enriched", confidence
        return "cold_storage", confidence

    def generate_training_label(
        self,
        query: str,
        compressed_entry: str,
        original_entry: str,
    ) -> float:
        """
        Same computation as score() — explicit method for AMA-Bench training
        pair generation so the training pipeline has a stable, named API.
        """
        return self.score(query, compressed_entry, original_entry)
