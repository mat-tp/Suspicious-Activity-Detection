"""Fixed locations used by the project.

Layout:  <project root>/source/activity_detection/paths.py
"""
import os
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SOURCE_DIR.parent

CONFIGS_DIR = SOURCE_DIR / "configs"
DATA_DIR = PROJECT_ROOT / "data"
# Results go to <project root>/experiments unless SDR_EXPERIMENTS_DIR names another folder
# (build.sh uses this to keep smoke-test runs apart from the real results).
EXPERIMENTS_DIR = PROJECT_ROOT / os.environ.get("SDR_EXPERIMENTS_DIR", "experiments")
