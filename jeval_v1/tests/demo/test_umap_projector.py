import numpy as np
import pytest

from demo.umap_projector import UMAPProjector


@pytest.fixture(scope="module")
def encoder():
    from jeval.encoders.sentence_encoder import FrozenEncoder
    return FrozenEncoder()


def test_transform_before_fit_returns_origin():
    proj = UMAPProjector()
    result = proj.transform(np.random.rand(768))
    assert result == [0.0, 0.0, 0.0]


def test_transform_after_fit_returns_three_floats(encoder):
    proj = UMAPProjector()
    texts = [
        "created auth middleware",
        "test suite failed with 14 errors",
        "deployment failed health check",
        "fixed JWT_SECRET mismatch",
        "all 23 tests passing",
        "modified routes to add rate limiting",
        "rolled back migration on users table",
        "error 503 staging server unresponsive",
        "npm install 847 packages",
        "TypeError at line 34",
    ]
    embeddings = encoder.encode(texts)
    proj.fit(embeddings)
    result = proj.transform(encoder.encode(["new segment arrives"])[0])
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_all_values_in_range(encoder):
    proj = UMAPProjector()
    texts = [
        "created auth middleware",
        "test suite failed with 14 errors",
        "deployment failed health check",
        "fixed JWT_SECRET mismatch",
        "all 23 tests passing",
        "modified routes to add rate limiting",
        "rolled back migration on users table",
        "error 503 staging server unresponsive",
        "npm install 847 packages",
        "TypeError at line 34",
    ]
    embeddings = encoder.encode(texts)
    proj.fit(embeddings)
    for text in texts[:5]:
        coords = proj.transform(encoder.encode([text])[0])
        for v in coords:
            assert -5.1 <= v <= 5.1, f"coordinate {v} out of [-5.1, 5.1]"


def test_seed_from_encoder_produces_fitted_projector(encoder):
    proj = UMAPProjector()
    assert not proj._fitted
    texts = [
        "created auth middleware",
        "test suite failed with 14 errors",
        "deployment failed health check",
        "fixed JWT_SECRET mismatch",
        "all 23 tests passing",
        "modified routes to add rate limiting",
        "rolled back migration on users table",
        "error 503 staging server unresponsive",
        "npm install 847 packages",
        "TypeError at line 34",
        "step 12 database connection pool exhausted",
    ]
    proj.seed_from_encoder(encoder, texts)
    assert proj._fitted
    result = proj.transform(encoder.encode(["verification step"])[0])
    assert len(result) == 3
    for v in result:
        assert -5.1 <= v <= 5.1
