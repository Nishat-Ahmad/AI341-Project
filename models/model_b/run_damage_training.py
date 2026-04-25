#!/usr/bin/env python
"""Damage detector training script runner."""
import sys
import warnings
import multiprocessing as mp
from pathlib import Path

# Suppress noisy deprecation warning emitted by transformers/torch internals.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node.*deprecated.*",
    category=FutureWarning,
)

# Add FleetThing to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model_b.main import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
