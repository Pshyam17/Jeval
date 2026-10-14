from jeval.benchmarks.ama_bench import AMABenchLoader
from jeval.benchmarks.swe_bench import SWEBenchLoader

__all__ = ["SWEBenchLoader", "AMABenchLoader", "DroidBenchLoader"]


def __getattr__(name: str):
    if name == "DroidBenchLoader":
        from jeval.benchmarks.droid_bench import DroidBenchLoader
        return DroidBenchLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")