from jeval.artifacts.index import ArtifactIndex


def test_artifact_index_update_and_recall():
    idx = ArtifactIndex()
    idx.update_from_text("src/auth.ts fix JWT_SECRET", turn=0, epe=0.1)
    result = idx.to_list()
    print("Result:", result)
    assert any(e.path == "src/auth.ts" for e in result)


def test_artifact_eval_precision_recall():
    from jeval.eval.artifacts import ArtifactEval

    idx = ArtifactIndex()
    idx.update_from_text("/api/login 401", turn=0, epe=0.2)
    score = ArtifactEval.score(idx, "Please call /api/login and handle 401")
    assert score.recall == 1.0
    assert score.f1 > 0.0
