import sys
from pathlib import Path

# make both jeval/ and demo/ importable from any pytest invocation
sys.path.insert(0, str(Path(__file__).parent))
