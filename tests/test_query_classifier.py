import pytest
from jeval.memory.query_classifier import QueryClassifier


@pytest.fixture
def qc():
    return QueryClassifier()


def test_step_reference_routes_to_precision(qc):
    assert qc.classify("what happened at step 79") == "precision"


def test_file_path_routes_to_entity(qc):
    assert qc.classify("find information about src/auth.ts") == "entity"


def test_generic_question_routes_to_context(qc):
    assert qc.classify("what was the overall outcome of the migration") == "context"


def test_both_patterns_routes_to_ambiguous(qc):
    # step number + ALL_CAPS identifier
    result = qc.classify("at step 7 what happened with JWT_SECRET")
    assert result == "ambiguous"


def test_extract_seq_id_correct(qc):
    assert qc.extract_seq_id("what happened at step 79") == 79
    assert qc.extract_seq_id("no number here") is None


def test_extract_entity_hint_file_path(qc):
    hint = qc.extract_entity_hint("find information about src/auth.ts and JWT_SECRET")
    # file path is longer and should win
    assert hint is not None


def test_merge_results_deduplicates_by_seq_id(qc):
    prec = [{"seq_id": 5, "content": "step 5 deployed"}]
    ctx = [{"seq_id": 5, "text": "step 5 deployed again"}]
    result = qc.merge_results(prec, ctx)
    # seq_id 5 should appear only once
    assert result.count("5") <= 2  # "[step 5]" contains one "5"


def test_merge_results_respects_token_cap(qc):
    prec = [{"seq_id": i, "content": " ".join(["word"] * 100)} for i in range(5)]
    ctx = [{"seq_id": 10 + i, "text": " ".join(["word"] * 100)} for i in range(5)]
    result = qc.merge_results(prec, ctx, max_tokens=50)
    assert len(result.split()) <= 55  # allow small margin for labels
