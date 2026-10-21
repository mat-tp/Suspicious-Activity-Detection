"""Puts source/ and scripts/ on the import path so the tests can import both."""
import sys
from pathlib import Path

# This is the folder with the activity_detection package.
SOURCE_DIR = Path(__file__).resolve().parent.parent

# This is the folder with the command-line scripts that some tests import.
SCRIPTS_DIR = SOURCE_DIR.parent / "scripts"

for folder in (SOURCE_DIR, SCRIPTS_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
