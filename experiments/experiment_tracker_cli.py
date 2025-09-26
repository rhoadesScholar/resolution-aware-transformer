#!/usr/bin/env python3
"""
Experiment Tracker CLI Tool

Provides command-line interface for experiment tracking and management.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.experiment_tracker import main

if __name__ == "__main__":
    main()