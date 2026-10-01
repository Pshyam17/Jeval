from jeval_v2.locomo_prompt_pairs import build


def test_same_memory_and_question_blind_arms():
    rows = [{"id": "q1", "question": "Current job?", "compressed_memory": "Maya was hired.",
             "answer": "Unemployed", "cosine": .2, "epe": .4},
            {"id": "q2", "question": "When hired?", "compressed_memory": "Maya was hired.",
             "answer": "January", "cosine": .3, "epe": .1}]
    prompts, key, review = build(rows, seed=2)
    assert len(prompts) == len(key) == 4
    assert len(review) == 2
    assert {r["arm"] for r in key[:2]} == {"cosine", "epe"}
    assert "Maya was hired." in prompts[0]["prompt"] and "Maya was hired." in prompts[1]["prompt"]
    assert "Current job?" in prompts[0]["prompt"] and "Current job?" in prompts[1]["prompt"]
    assert "Unemployed" not in prompts[0]["prompt"]
