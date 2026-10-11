"""Entry point.  Run from the project root, for example:

    python source/run.py --config group_a_classical --target activity
    python source/run.py --list-configs
"""
import sys

from activity_detection.cli import main

if __name__ == "__main__":
    sys.exit(main())
