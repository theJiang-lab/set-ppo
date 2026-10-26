from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from setppo.config import Config
from setppo.data import synthetic_dataset

torch.set_num_threads(1)


@pytest.fixture
def config():
    return Config.load(Path(__file__).resolve().parents[1] / 'configs/smoke.json')


@pytest.fixture
def dataset(config):
    return synthetic_dataset(config)

