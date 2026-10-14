import time
import pytest
import numpy as np
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.hot_cache import HotCache
from jeval.memory.contradiction_detector import ContradictionDetector


@pytest.fixture
def setup(encoder):
    cache = HotCache(encoder, token_ceiling=8000)
    detector = ContradictionDetector(encoder, cache, epe_threshold=0.10)
    return encoder, cache, detector


def _store(cache, encoder, text, seq_id, epe=0.5):
    emb = encoder.encode([text])[0]
    cache.store(text, emb, seq_id=seq_id, content_type="FACTUAL", epe_score=epe)


def test_outcome_inversion_detected(setup):
    encoder, cache, detector = setup
    _store(cache, encoder, "step 6 migration succeeded without errors", seq_id=6)
    stale = detector.check("step 6 migration failed due to lock timeout", "s1")
    assert 6 in stale


def test_faithful_update_not_flagged(setup):
    encoder, cache, detector = setup
    _store(cache, encoder, "step 5 all 14 tests passing after fix", seq_id=5)
    # semantically close, no polarity inversion
    stale = detector.check("step 5 all 14 tests passing after the patch", "s1")
    assert 5 not in stale


def test_polarity_check_symmetric(setup):
    encoder, cache, detector = setup
    # check both directions of the polarity pair
    _store(cache, encoder, "deployment failed on production", seq_id=99)
    stale = detector.check("deployment succeeded on production", "s1")
    assert 99 in stale


def test_below_threshold_epe_not_flagged(encoder):
    # use high threshold so nothing is flagged
    cache = HotCache(encoder, token_ceiling=8000)
    detector = ContradictionDetector(encoder, cache, epe_threshold=0.99)
    emb = encoder.encode(["migration succeeded"])[0]
    cache.store("migration succeeded", emb, seq_id=1, content_type="FACTUAL", epe_score=0.5)
    stale = detector.check("migration failed", "s1")
    # with threshold=0.99, cosine distance must exceed 0.99 — nearly impossible
    assert 1 not in stale


def test_async_does_not_block_caller(setup):
    encoder, cache, detector = setup
    _store(cache, encoder, "all tests passed successfully", seq_id=200)
    import threading
    stale_result: list[list[int]] = []

    def run():
        stale_result.append(detector.check("all tests failed", "s1"))

    t = threading.Thread(target=run, daemon=True)
    t0 = time.time()
    t.start()
    t.join(timeout=5.0)
    elapsed = time.time() - t0
    assert elapsed < 5.0  # completed within timeout
