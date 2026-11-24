"""
Utility functions for the EEG seizure detection pipeline.
Handles configuration loading, logging, and common helpers.
"""

import os
import yaml
import logging
import json
import numpy as np
from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level=logging.INFO):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("EEG-Pipeline")


def ensure_dirs(config):
    """Create all output directories specified in config."""
    for key, path in config["paths"].items():
        os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    """Save dictionary as JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)
