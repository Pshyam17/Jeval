from jeval_v2.benchmarks.prepare import locomo, longmemeval


def test_longmemeval_preserves_evidence_without_assigning_harm():
    example = {"question_id": "q1", "question": "Current job?", "answer": "Unemployed",
               "question_type": "knowledge-update", "answer_session_ids": ["s2"],
               "haystack_session_ids": ["s1", "s2"],
               "haystack_sessions": [[{"content": "hired"}], [{"content": "fired", "has_answer": True}]]}
    row = next(longmemeval([example]))
    assert row["evidence_turns"] == [{"session_id": "s2", "turn_index": 0}]
    assert "harm" not in row


def test_locomo_groups_all_questions_by_conversation():
    rows = list(locomo([{"sample_id": "c1", "qa": [
        {"question": "When?", "answer": "May", "evidence": ["D1:3"]},
        {"question": "Who?", "answer": "A", "evidence": []}]}]))
    assert [row["group_id"] for row in rows] == ["c1", "c1"]
    assert rows[0]["evidence_dialog_ids"] == ["D1:3"]
