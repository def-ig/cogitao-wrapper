from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeneratorConfig:
    """Configuration for the Task Generator

    This config defines how to generate synthetic tasks using the COGITAO generator.
    It controls grid size, shapes, transformations, and output formatting.
    """

    # Output settings
    output_file: str = "./data/dataset_train.h5"

    # Grid configuration
    grid_size: int = 32
    min_n_shapes: int = 2
    max_n_shapes: int = 3
    max_shape_size: int = 12

    # Generation
    num_tasks: int = 100000
    num_workers: int = 16

    # Upscaling settings
    image_size: int = 224
    upscale_method: str = "nearest"  # 'nearest' or 'bilinear'

    # Transformation settings
    allowed_transformations: Optional[List[str]] = field(
        default_factory=lambda: ["rotate90"]
    )
    allowed_combinations: Optional[List[List[str]]] = None

    # Visualization and inspection
    n_examples_to_visualize: int = 5

    # Advanced settings
    n_examples: int = 1
    min_transformation_depth: int = 0
    max_transformation_depth: int = 0
