import numpy as np

from jeval.epe.core import EPEComputer


def test_epe_score_identity():
    orig = np.ones(768, dtype=np.float32) / np.sqrt(768)
    pred = orig.copy()
    assert EPEComputer.epe_score(orig, pred) == 0.0


def test_epe_score_orthogonal():
    orig = np.zeros(768, dtype=np.float32)
    orig[0] = 1.0
    pred = np.zeros(768, dtype=np.float32)
    pred[1] = 1.0
    assert np.isclose(EPEComputer.epe_score(orig, pred), 0.5)
