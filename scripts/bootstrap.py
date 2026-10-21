"""Makes the packages under source/ importable when a script is started directly."""
import sys
from pathlib import Path

# This is the folder that holds the activity_detection package.
SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"

sys.path.insert(0, str(SOURCE_DIR))
