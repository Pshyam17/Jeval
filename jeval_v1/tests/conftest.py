"""
Session-scoped encoder fixtures shared across all test modules.

Using scope="session" ensures a single FrozenEncoder instance (one model
load onto the MPS/GPU device) is shared for the entire pytest run.  The
previous per-module fixtures caused MPS OOM when running the full suite
because up to 9 model copies accumulated on the GPU simultaneously.
"""
from __future__ import annotations

import pytest

from jeval.encoders.sentence_encoder import FrozenEncoder


@pytest.fixture(scope="session")
def enc():
    return FrozenEncoder()


@pytest.fixture(scope="session")
def encoder():
    return FrozenEncoder()
