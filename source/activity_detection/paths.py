"""Fixed locations used by the project.

Layout:  <project root>/source/activity_detection/paths.py
"""
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SOURCE_DIR.parent

CONFIGS_DIR = SOURCE_DIR / "configs"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
