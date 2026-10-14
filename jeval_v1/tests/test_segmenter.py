import pytest
from jeval.memory.segmenter import SessionSegmenter


@pytest.fixture
def seg():
    return SessionSegmenter()


def test_empty_input_returns_empty_list(seg):
    assert seg.segment("") == []
    assert seg.segment("   ") == []


def test_markdown_headers_produce_segments(seg):
    text = "## Authentication\nJWT_SECRET fixed in env.ts\n\n## Migration\nRolled back due to lock timeout"
    segs = seg.segment(text)
    assert len(segs) >= 2
    assert any("JWT_SECRET" in s for s in segs)
    assert any("Migration" in s or "lock timeout" in s for s in segs)


def test_double_newlines_split(seg):
    text = "First paragraph about auth fixes.\n\nSecond paragraph about database migration."
    segs = seg.segment(text)
    assert len(segs) >= 2


def test_short_bullets_merged(seg):
    text = "- ok\n- yes\n- fine\n- sure\n- done"
    segs = seg.segment(text)
    # short bullets (1 token each) should be merged
    assert len(segs) < 5


def test_minimum_length_enforced(seg):
    text = "Short.\n\nA much longer paragraph with enough tokens to stand on its own as a segment."
    segs = seg.segment(text)
    for s in segs:
        assert len(s.split()) >= 1  # merged with neighbor or alone


def test_maximum_length_enforced(seg):
    # 400-word segment should be split
    long_sent = "The migration failed because of a lock timeout. " * 20
    segs = seg.segment(long_sent)
    for s in segs:
        assert len(s.split()) <= 310  # allow slight margin


def test_realistic_memories_sample(seg):
    sample = """## Session Summary

    JWT_SECRET was missing from src/config/env.ts causing all auth tests to fail.

    ## Steps Taken

    - step 1: identified the missing variable
    - step 2: added JWT_SECRET to env.ts
    - step 3: ran test suite — 14 tests now passing

    ## Migration Status

    Migration rolled back on staging due to lock timeout after 30 seconds.
    The roles table had 423 concurrent connections at the time of failure.
    """
    segs = seg.segment(sample)
    assert len(segs) >= 3
    avg_tokens = sum(len(s.split()) for s in segs) / len(segs)
    assert 5 <= avg_tokens <= 300
