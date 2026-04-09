import pytest
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.novelty_gate import NoveltyGate


@pytest.fixture(scope="module")
def encoder():
    return FrozenEncoder()


@pytest.fixture
def gate(encoder):
    return NoveltyGate(encoder, threshold=0.15, working_set_size=5)


def test_empty_working_set_always_novel(gate):
    is_novel, epe = gate.is_novel("anything at all")
    assert is_novel is True
    assert epe == 1.0


def test_identical_text_not_novel(gate):
    text = "migration failed on staging due to lock timeout after 30s"
    gate.update_working_set(text)
    is_novel, epe = gate.is_novel(text)
    assert is_novel is False
    assert epe < 0.05  # identical text → near-zero cosine distance


def test_semantically_different_text_is_novel(gate):
    gate.update_working_set("migration failed on staging due to lock timeout")
    is_novel, epe = gate.is_novel("the weather today is sunny and warm")
    assert is_novel is True
    assert epe > 0.15


def test_working_set_size_limit_fifo(encoder):
    gate = NoveltyGate(encoder, threshold=0.15, working_set_size=3)
    texts = [
        "step 1: create server",
        "step 2: add middleware",
        "step 3: run tests",
        "step 4: fix failures",  # pushes out step 1
    ]
    for t in texts:
        gate.update_working_set(t)
    assert len(gate._working_set) == 3


def test_threshold_parameter():
    enc = FrozenEncoder()

    # low threshold — gate is permissive, nearly everything is novel
    gate_low = NoveltyGate(encoder=enc, threshold=0.01)
    gate_low.update_working_set("migration failed on staging")
    is_novel, epe = gate_low.is_novel("migration failed on staging due to timeout")
    # epe ~0.10-0.15, well above 0.01 threshold — should be novel
    assert is_novel is True
    assert epe > 0.01

    # high threshold — gate is strict, related content is NOT novel
    gate_high = NoveltyGate(encoder=enc, threshold=0.99)
    gate_high.update_working_set("migration failed on staging")
    is_novel, epe = gate_high.is_novel("migration failed on staging due to timeout")
    # epe ~0.10-0.15, well below 0.99 threshold — should NOT be novel
    assert is_novel is False
    assert epe < 0.99

    # moderate threshold — truly unrelated content IS novel
    gate_mid = NoveltyGate(encoder=enc, threshold=0.50)
    gate_mid.update_working_set("migration failed on staging")
    is_novel, epe = gate_mid.is_novel("the quick brown fox jumps over the lazy dog")
    assert is_novel is True
    assert epe > 0.50
