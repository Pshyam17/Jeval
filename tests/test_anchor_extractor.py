import pytest
from jeval.memory.anchor_extractor import AnchorExtractor, MIN_CORPUS_SEGMENTS_FOR_TIER2


@pytest.fixture
def extractor():
    return AnchorExtractor(max_anchors=10)


# ── Tier 1: rule-based unconditional anchors ──────────────────────────────────

def test_number_is_tier1_anchor(extractor):
    extractor.update_corpus("returned 847 failures after patch")
    anchors = extractor.extract("returned 847 failures after patch", [])
    assert "847" in anchors


def test_all_caps_identifier_is_tier1_anchor(extractor):
    extractor.update_corpus("JWT_SECRET missing from config")
    anchors = extractor.extract("JWT_SECRET missing from config", [])
    assert "JWT_SECRET" in anchors


def test_file_path_is_tier1_anchor(extractor):
    extractor.update_corpus("fixed auth in src/middleware/auth.ts")
    anchors = extractor.extract("fixed auth in src/middleware/auth.ts", [])
    assert "src/middleware/auth.ts" in anchors


def test_step_reference_entity_is_tier1_anchor(extractor):
    extractor.update_corpus("step 79 action: deployed")
    entities = [{"text": "step 79", "type": "step_reference", "value": None}]
    anchors = extractor.extract("step 79 action: deployed", entities)
    assert "step 79" in anchors


def test_cardinal_entity_is_tier1_anchor(extractor):
    extractor.update_corpus("step 42 returned 14 failures")
    entities = [{"text": "42", "type": "CARDINAL", "value": None}]
    anchors = extractor.extract("step 42 returned 14 failures", entities)
    assert "42" in anchors


def test_error_class_is_tier1_anchor(extractor):
    extractor.update_corpus("raised KeyError during migration")
    anchors = extractor.extract("raised KeyError during migration", [])
    assert "KeyError" in anchors


def test_xyzqlockout_not_selected_by_tier1(extractor):
    """xyzqlockout is not a number, file path, ALL_CAPS, error class, or step ref."""
    from jeval.memory.anchor_extractor import _is_tier1
    assert not _is_tier1("xyzqlockout")


# ── Tier 2: skipped below MIN_CORPUS_SEGMENTS_FOR_TIER2 ──────────────────────

def test_tier2_skipped_below_min_corpus_size():
    """Below MIN_CORPUS_SEGMENTS_FOR_TIER2, only Tier 1 anchors are returned.
    rare_query_tok is not a Tier 1 pattern, so it must NOT be selected."""
    ext = AnchorExtractor(max_anchors=10)
    # seed fewer than MIN_CORPUS_SEGMENTS_FOR_TIER2 segments
    assert MIN_CORPUS_SEGMENTS_FOR_TIER2 == 20
    for _ in range(MIN_CORPUS_SEGMENTS_FOR_TIER2 - 1):  # 19 segments
        ext.update_corpus("alpha beta gamma delta")
    anchors = ext.extract("rare_query_tok alpha beta", [])
    assert "rare_query_tok" not in anchors
    # but a Tier 1 anchor (number) in the same text IS returned
    anchors_with_num = ext.extract("rare_query_tok 847 alpha", [])
    assert "847" in anchors_with_num


# ── Tier 2: fires above MIN_CORPUS_SEGMENTS_FOR_TIER2 ────────────────────────

def test_tier2_fires_above_min_corpus_size():
    """With >= 20 segments, Tier 2 selects rare tokens above the corpus median IDF.

    Corpus design:
      - "alpha/beta/gamma/delta" appear in all 25 segments → IDF ≈ 0.67 (low cluster)
      - "rare_query_tok" appears in 1 segment → IDF ≈ 3.26 (high cluster)
    Median lands in the low cluster; rare_query_tok >> median → selected.
    """
    ext = AnchorExtractor(max_anchors=10)
    for _ in range(25):
        ext.update_corpus("alpha beta gamma delta alpha beta gamma delta")
    ext.update_corpus("rare_query_tok alpha beta")
    anchors = ext.extract("rare_query_tok alpha beta", [])
    assert "rare_query_tok" in anchors


def test_tier2_median_robust_to_bimodal_distribution():
    """Median threshold is stable even with a bimodal IDF corpus.

    Previous bug: mean + 1.5*std fails when many unique tokens share identical
    high IDF — the std is inflated and the threshold exceeds all unique-token IDFs.
    The median approach does not have this problem when the corpus is large enough
    (>= MIN_CORPUS_SEGMENTS_FOR_TIER2) and common tokens anchor the low cluster.
    """
    ext = AnchorExtractor(max_anchors=10)
    # Low-IDF cluster: "alpha/beta/gamma/delta" in all 25 segments
    for _ in range(25):
        ext.update_corpus("alpha beta gamma delta")
    # High-IDF cluster: one rare token
    ext.update_corpus("bimodalrare alpha")
    anchors = ext.extract("bimodalrare alpha", [])
    assert "bimodalrare" in anchors


# ── Stopwords and short tokens ────────────────────────────────────────────────

def test_stopwords_excluded(extractor):
    extractor.update_corpus("the is was to of")
    anchors = extractor.extract("the is was to of", [])
    for sw in ["the", "is", "was", "to", "of"]:
        assert sw not in anchors


# ── Cap and reset ─────────────────────────────────────────────────────────────

def test_max_anchors_cap():
    ext = AnchorExtractor(max_anchors=3)
    ext.update_corpus("token1 token2 token3 token4 token5 token6 token7 token8")
    entities = [
        {"text": f"token{i}", "type": "CARDINAL", "value": None}
        for i in range(1, 9)
    ]
    anchors = ext.extract(
        "token1 token2 token3 token4 token5 token6 token7 token8", entities
    )
    assert len(anchors) <= 3


def test_reset_clears_corpus(extractor):
    extractor.update_corpus("some text about deployments and migrations")
    extractor.reset()
    assert extractor._total_segments == 0
    assert len(extractor._seg_freq) == 0
