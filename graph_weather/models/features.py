"""Manages the loading, parsing, and assembly of weather features from a configuration file."""

from typing import Dict, List

import torch

from graph_weather.utils.config import load_feature_config


class FeatureManager:
    """Parses a feature config and provides an interface to manage feature dimensions and assembly."""

    def __init__(self, config_path: str):
        """Initializes the FeatureManager."""
        self.config = load_feature_config(config_path)
        self._process_config()

    def _process_config(self):
        """Internal method to parse loaded configuration."""
        self.feature_order: List[str] = []
        self.dynamic_variable_names: List[str] = []
        self.static_feature_names: List[str] = []
        self.level_map: Dict[str, List[int]] = {}

        num_dynamic_features = 0

        for feature in self.config.features:
            if feature.type == "dynamic":
                self.dynamic_variable_names.append(feature.name)
                if feature.levels:
                    self.dynamic_variable_names.append(feature.name)
                    self.level_map[feature.name] = feature.levels
                    for level in feature.levels:
                        self.feature_order.append(f"{feature.name}_L{level}")
                        num_dynamic_features += 1
                else:
                    self.feature_order.append(feature.name)
                    num_dynamic_features += 1
            elif feature.type == "static":
                self.static_feature_names.append(feature.name)
                self.feature_order.append(feature.name)

        self.num_features = len(self.feature_order)
        self.num_dynamic_features = num_dynamic_features
        self.num_static_features = len(self.static_feature_names)

    def assemble_features(self, data: Dict) -> torch.Tensor:
        """
        Assembles final input tensor from a dictionary of raw data tensors.

        Args:
            data (Dict): mapping feature names to tensors.
                         For multi-level variables, the tensors should
                         have shape (batch, nodes, levels). else (batch, nodes, 1).

        Returns:
            single concatenated tensor of shape (batch, nodes, num_features).

        """
        feature_tensors = []

        for feature_name_with_level in self.feature_order:
            parts = feature_name_with_level.split("_")

            # if feature name corresponds to a multi-level variable
            base_name = "_".join(parts[:-1])
            level_str = parts[-1]

            if base_name in self.level_map and level_str.isdigit():
                level = int(level_str)
                # data[base_name] has shape (nodes, num_levels); index for this pressure level
                try:
                    level_idx = self.level_map[base_name].index(level)
                    tensor_slice = data[base_name][:, level_idx].unsqueeze(-1)
                    feature_tensors.append(tensor_slice)
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f"Level {level} for variable {base_name} not found in config or data."
                    ) from e
            else:
                # single level/ derived feature
                base_name = feature_name_with_level
                if base_name in data:
                    tensor = data[base_name]
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(-1)
                    feature_tensors.append(tensor)
                else:
                    raise ValueError(
                        f"Feature {base_name} from config not found in data dictionary."
                    )

        return torch.cat(feature_tensors, dim=-1)
