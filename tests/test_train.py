import pytest
from unittest.mock import Mock, patch
from jeval.train.generate_pairs import PairGenerator
from jeval.train.train_predictor import PairDataset
from jeval.ingest.base import Segment, Session


def test_pair_generator():
    """Test pair generation for JEPA training."""
    # Mock encoder
    mock_encoder = Mock()
    mock_encoder.encode.side_effect = lambda text: [0.1, 0.2, 0.3] if "hello" in text.lower() else [0.4, 0.5, 0.6]
    mock_encoder.cosine_sim.side_effect = lambda a, b: 0.8 if a == b else 0.2

    generator = PairGenerator(mock_encoder)

    # Create test segments
    segments = [
        Segment(text="Hello world", role="user", turn=0, source="test"),
        Segment(text="Goodbye world", role="assistant", turn=1, source="test"),
        Segment(text="Hello again", role="user", turn=2, source="test"),
    ]

    pairs = generator.generate_pairs(segments, num_pairs=4)

    assert len(pairs) == 4  # num_pairs=4 means 4 pairs are returned
    for anchor, target, label in pairs:
        assert isinstance(anchor, str)
        assert isinstance(target, str)
        assert label in [0, 1]  # 0 for negative, 1 for positive


def test_pair_dataset():
    """Test pair dataset creation."""
    # Mock components
    mock_encoder = Mock()
    mock_encoder.encode.return_value = [0.1, 0.2, 0.3]

    mock_predictor = Mock()
    mock_predictor.return_value = Mock()  # Mock tensor output
    mock_predictor.return_value.squeeze.return_value = [0.4, 0.5, 0.6]

    pairs = [("anchor text", "target text", 1)]
    dataset = PairDataset(pairs, mock_encoder, mock_predictor)

    assert len(dataset) == 1

    # Test __getitem__
    item = dataset[0]
    assert "anchor_emb" in item
    assert "target_emb" in item
    assert "pred_emb" in item
    assert "label" in item


@pytest.mark.skip(reason="Training test requires actual ML setup")
def test_train_predictor():
    """Test predictor training (skipped due to ML dependencies)."""
    # This would require setting up actual torch models and training
    # For now, skip to avoid complex mocking
    pass