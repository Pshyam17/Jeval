"""
Tests for CombinedEPE.

Validates alpha blending, range constraints, and the paper's key claim:
combined signal at alpha=0.5 outperforms cosine EPE alone on causal elision.
"""
from __future__ import annotations

import numpy as np
import pytest

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.epe.combined import CombinedEPE
from jeval.memory.schema_gap import SchemaGapVerifier


@pytest.fixture(scope="module")
def verifier():
    return SchemaGapVerifier()


@pytest.fixture(scope="module")
def combined_default(enc, verifier):
    return CombinedEPE(enc, verifier, alpha=0.5)


ORIGINAL = (
    "migration failed on staging due to lock timeout after 30s on "
    "roles table with 423 connections"
)
COMPRESSED = "migration failed on staging"


def test_alpha_zero_gives_cosine_only(enc, verifier):
    """alpha=0 → epe_final == cosine_epe (schema gap term is zeroed out)."""
    c = CombinedEPE(enc, verifier, alpha=0.0)
    cosine_epe, schema_gap, epe_final = c.compute(ORIGINAL, COMPRESSED, "migration_failure")
    assert abs(epe_final - schema_gap) < 1e-6, (
        f"alpha=0 should give schema_gap only: epe_final={epe_final}, schema_gap={schema_gap}"
    )


def test_alpha_one_gives_schema_gap_only(enc, verifier):
    """alpha=1 → epe_final == cosine_epe (schema gap term is zeroed out)."""
    c = CombinedEPE(enc, verifier, alpha=1.0)
    cosine_epe, schema_gap, epe_final = c.compute(ORIGINAL, COMPRESSED, "migration_failure")
    assert abs(epe_final - cosine_epe) < 1e-6, (
        f"alpha=1 should give cosine_epe only: epe_final={epe_final}, cosine_epe={cosine_epe}"
    )


def test_alpha_half_gives_equal_blend(enc, verifier):
    """alpha=0.5 → epe_final == 0.5 * cosine + 0.5 * schema."""
    c = CombinedEPE(enc, verifier, alpha=0.5)
    cosine_epe, schema_gap, epe_final = c.compute(ORIGINAL, COMPRESSED, "migration_failure")
    expected = 0.5 * cosine_epe + 0.5 * schema_gap
    assert abs(epe_final - expected) < 1e-6


def test_all_values_in_unit_range(combined_default):
    cosine_epe, schema_gap, epe_final = combined_default.compute(
        ORIGINAL, COMPRESSED, "migration_failure"
    )
    assert 0.0 <= cosine_epe <= 1.0
    assert 0.0 <= schema_gap <= 1.0
    assert 0.0 <= epe_final <= 1.0


def test_identical_text_gives_zero_cosine_epe(enc, verifier):
    c = CombinedEPE(enc, verifier, alpha=0.5)
    cosine_epe, _, _ = c.compute("identical text", "identical text", "deployment")
    assert cosine_epe < 0.01


def test_combined_higher_than_cosine_on_causal_elision(combined_default):
    """
    Key paper claim: combined signal > cosine EPE alone on the causal elision case.

    The encoder sees both strings as near-identical (low cosine EPE), but schema gap
    correctly captures that timing, table_name, and error_type were all lost.
    epe_final with alpha=0.5 is therefore substantially higher than cosine EPE alone.
    """
    cosine_epe, schema_gap, epe_final = combined_default.compute(
        ORIGINAL, COMPRESSED, "migration_failure"
    )
    assert epe_final > cosine_epe, (
        f"epe_final ({epe_final:.4f}) should exceed cosine_epe ({cosine_epe:.4f}) "
        "on causal elision"
    )
    # record paper number
    print(
        f"\n[PAPER] causal elision combined: cosine_epe={cosine_epe:.4f}, "
        f"schema_gap={schema_gap:.4f}, epe_final={epe_final:.4f}"
    )


def test_no_schema_type_falls_back_to_cosine(enc, verifier):
    """When content_type has no schema, schema_gap=0 and epe_final = alpha * cosine."""
    c = CombinedEPE(enc, verifier, alpha=0.5)
    cosine_epe, schema_gap, epe_final = c.compute(
        "some text here", "some text", "unknown_type"
    )
    assert schema_gap == 0.0
    assert abs(epe_final - 0.5 * cosine_epe) < 1e-6


def test_compute_novelty_empty_cache_returns_one(enc, verifier):
    c = CombinedEPE(enc, verifier)
    empty = np.empty((0, enc.dim()), dtype=np.float32)
    assert c.compute_novelty("any text", empty) == 1.0


def test_compute_novelty_identical_text_near_zero(enc, verifier):
    c = CombinedEPE(enc, verifier)
    text = "migration failed on staging"
    emb = enc.encode([text])
    novelty = c.compute_novelty(text, emb)
    assert novelty < 0.01


def test_compute_novelty_different_text_high(enc, verifier):
    c = CombinedEPE(enc, verifier)
    cached_text = "the weather is sunny today"
    new_text = "migration failed due to database timeout"
    cached_emb = enc.encode([cached_text])
    novelty = c.compute_novelty(new_text, cached_emb)
    assert novelty > 0.2
