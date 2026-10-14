from jeval.eval.report import CompressionReport
from jeval.strata.budget import SegmentPlan


def test_compression_report_from_plans():
    plans = [SegmentPlan(0.1, -0.2, "BACKGROUND", False, 0.3), SegmentPlan(1.0, 0.8, "FACTUAL", False, 1.0)]
    report = CompressionReport.from_plans(plans)
    assert report.segment_count == 2
    assert report.high_risk_count == 1
    assert report.low_risk_count == 1
