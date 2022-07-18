from typing import Mapping, Any
import yaml

from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


class YAMLConfig:
    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as f:
            self._cfg: Mapping[str, Any] = yaml.safe_load(f)

    def __len__(self) -> int:
        return len(self._cfg)

    def __getitem__(self, key: str) -> Any:
        nested_keys = key.split(":")
        data = self._cfg
        try:
            for k in nested_keys[:-1]:
                data = data[k]
            return data[nested_keys[-1]]
        except KeyError as e:
            LOGGER.error("Invalid config key: %s", key)
            raise KeyError from e
