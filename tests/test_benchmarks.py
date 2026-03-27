from jeval.benchmarks.ama_bench import AMABenchLoader
from jeval.benchmarks.swe_bench import SWEBenchLoader


def test_ama_bench_loader():
    """Test AMA benchmark loader."""
    loader = AMABenchLoader()
    sessions = loader.load_sessions()

    assert len(sessions) == 10
    for session in sessions:
        assert session.session_id.startswith("ama_")
        assert len(session.segments) == 4  # question, response, follow-up, follow-up response
        assert all(isinstance(seg.text, str) for seg in session.segments)
        assert all(seg.turn >= 0 for seg in session.segments)


def test_swe_bench_loader():
    """Test SWE-Bench loader."""
    loader = SWEBenchLoader()
    sessions = loader.load_sessions()

    # Should load real dataset if available, or fallback to dummy
    assert len(sessions) > 0
    for session in sessions:
        assert isinstance(session.session_id, str)
        assert len(session.segments) >= 2  # At least problem and solution
        assert all(isinstance(seg.text, str) for seg in session.segments)