from jeval.strata.budget import BudgetAllocator


def test_budget_allocates_based_on_z_score_and_type():
    allocator = BudgetAllocator()
    plan = allocator.allocate(
        segment_text="example",
        epe=0.5,
        z_score=0.6,
        content_type="BACKGROUND",
        artifact_override=False,
        confidence=0.1,
    )
    assert plan.budget == 1.0

    plan2 = allocator.allocate(
        segment_text="example",
        epe=0.1,
        z_score=-0.6,
        content_type="BACKGROUND",
        artifact_override=False,
        confidence=0.1,
    )
    assert plan2.budget == 0.3
