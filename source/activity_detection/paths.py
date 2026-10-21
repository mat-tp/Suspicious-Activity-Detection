"""Folders that the project reads from and writes to."""
import os
from pathlib import Path

# This is the source/ folder, the one that holds the activity_detection package.
SOURCE_DIR = Path(__file__).resolve().parent.parent
# This is the top folder of the project, one level above source/.
PROJECT_ROOT = SOURCE_DIR.parent

# This is the folder with the YAML experiment configs.
CONFIGS_DIR = SOURCE_DIR / "configs"
# This is the folder where the dataset is expected.
DATA_DIR = PROJECT_ROOT / "data"
# This is where the results of the runs go: experiments/ unless SDR_EXPERIMENTS_DIR names another
# folder (the build scripts use it to keep smoke runs apart from the real results).
EXPERIMENTS_DIR = PROJECT_ROOT / os.environ.get("SDR_EXPERIMENTS_DIR", "experiments")
