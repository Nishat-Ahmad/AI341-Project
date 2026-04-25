#!/usr/bin/env python
"""Training script runner."""
import sys
from pathlib import Path

# Add FleetThing to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model_a.main import main

if __name__ == "__main__":
    main()
