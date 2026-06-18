import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UTILS_PATH = PROJECT_ROOT / "graph_weather" / "utils"
sys.path.insert(0, str(UTILS_PATH))

from input_validation import validate_model_input


def test_rejects_invalid_shape():
    x = torch.randn(100, 64)
    with pytest.raises(ValueError, match="Expected input shape"):
        validate_model_input(x)


def test_rejects_non_tensor():
    with pytest.raises(TypeError):
        validate_model_input([1, 2, 3])


def test_accepts_valid_input():
    x = torch.randn(2, 10, 5)
    validate_model_input(x)
