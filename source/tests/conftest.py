"""Make `activity_detection` importable when pytest is started from the project root."""
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
