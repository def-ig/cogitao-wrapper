import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

from .config import GeneratorConfig
from .generator import DatasetGenerator
from .dataset import CogitaoDataset


__all__ = ["GeneratorConfig", "DatasetGenerator", "CogitaoDataset"]
