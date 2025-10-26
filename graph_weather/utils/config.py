"""Configuration loading and validation utilities"""

from typing import List, Optional

import yaml
from pydantic import BaseModel


class FeatureConfig(BaseModel):
    """Defines the schema for a single feature in the config."""

    name: str
    type: str
    source: str
    levels: Optional[List[int]] = None


class FeatureSetConfig(BaseModel):
    """Defines the schema for the top-level feature configuration."""

    features: List[FeatureConfig]


def load_feature_config(path: str) -> FeatureSetConfig:
    """Loads and validates the feature configuration YAML file."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return FeatureSetConfig(**config_dict)
