import sys
from pathlib import Path

# project root must be on sys.path for `demo` to be importable
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
