"""
Task Generator for COGITAO training.

This module provides:
1. TaskGenerator: Simple wrapper around COGITAO generator for task generation
2. DatasetGenerator: Parallel cache generator using multiple worker processes
"""

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, is_dataclass
from logging import getLogger
from queue import Full as QueueFull
from typing import Any

import numpy as np
from arcworld.general_utils import generate_key
from arcworld.generator import Generator
from arcworld.utils.config_validation import ConfigValidator
from tqdm import tqdm

try:
    from omegaconf import DictConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    OmegaConf = Any
    DictConfig = None  # type: ignore

from .config import GeneratorConfig
from .dataset import HDF5CogitaoStore

try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Already set, ignore
    pass

_logger = getLogger(__name__)


def _config_to_dict(cfg) -> dict:
    """Convert config to dict, supporting both dataclass and OmegaConf.

    Args:
        cfg: Either a dataclass instance or OmegaConf DictConfig

    Returns:
        Dictionary representation of the config
    """
    if HAS_OMEGACONF and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    elif is_dataclass(cfg):
        return asdict(cfg)
    else:
        raise TypeError(
            f"Config must be a dataclass or OmegaConf DictConfig, got {type(cfg)}"
        )


def _sample_generation_worker(
    worker_id, sample_queue, root_gen_config, image_size, upscale_method
):
    """Worker process that generates samples and puts them in a shared queue.

    This worker does NOT write to any files - it only generates samples and
    puts them in the queue. The main process handles all H5 file writes.

    Args:
        worker_id: Worker ID for debugging
        sample_queue: Shared queue to put generated samples (all workers write here)
        root_gen_config: Pre-built config dict for root Generator (validated)
        image_size: Final image size for resizing
        upscale_method: Upscale method ('nearest' or 'bilinear')
    """
    # Create root generator directly with pre-built config
    from arcworld.constants import COLORMAP, NORM
    from arcworld.generator import Generator
    from PIL import Image

    gen = Generator(root_gen_config)

    failures = 0
    max_consecutive_failures = 50
    while True:
        try:
            # Generate task and convert to image
            task = gen.generate_single_task()
            input_grid = np.array(task["pairs"][-1]["input"])

            input_image = np.array(COLORMAP(NORM(input_grid)))[:, :, :3]

            # Resize
            resize_method = (
                Image.Resampling.NEAREST
                if upscale_method == "nearest"
                else Image.Resampling.BILINEAR
            )
            img_pil = Image.fromarray((input_image * 255).astype(np.uint8))
            img_resized = img_pil.resize((image_size, image_size), resize_method)
            img_array = np.array(img_resized).astype(np.float32) / 255.0

            # Convert to CHW format
            img_chw = np.transpose(img_array, (2, 0, 1))

            sample_queue.put(img_chw, timeout=1)
            failures = 0  # Reset on success
        except QueueFull:
            time.sleep(0.5)  # Wait before retrying
        except Exception as e:
            failures += 1
            if failures >= max_consecutive_failures:
                _logger.error(
                    f"Worker {worker_id} failed {max_consecutive_failures} times. Last error: {e}"
                )
                import traceback

                traceback.print_exc()
                break  # Stop this worker
            continue


class TaskGenerator:
    """Generator wrapper for creating training sample cache.

    This class wraps the COGITAO generator and provides functionality to:
    1. Generate synthetic task samples
    2. Save them to an HDF5 cache file for training

    The primary use case is to pre-generate a sample cache for training.
    """

    def __init__(self, cfg: GeneratorConfig, cache_path: str | None = None):
        """
        Initialize the task generator.

        Args:
            cfg: GeneratorConfig instance with all generation parameters
            cache_path: Optional path to HDF5 cache file. If None, uses 'data/sample_cache.h5'
        """
        self.cfg = cfg
        self.output_file = cfg.output_file
        self.image_size = cfg.image_size
        self.upscale_method = cfg.upscale_method

        # Build root generator config from our config
        self.root_gen_config = self._build_root_config(cfg)

        # Initialize the root generator
        self.gen = Generator(self.root_gen_config)

        # Setup cache
        # Convert config to dict (supports both dataclass and OmegaConf)
        cfg_container = _config_to_dict(cfg)
        if not isinstance(cfg_container, dict):
            raise ValueError("Failed to convert GeneratorConfig to dict")

        self.cache_path = cache_path or "data/sample_cache.h5"
        self.cache_config = {
            "grid_size": cfg.grid_size,
            "min_n_shapes": cfg.min_n_shapes,
            "max_n_shapes": cfg.max_n_shapes,
            "max_shape_size": cfg.max_shape_size,
            "allowed_transformations": cfg_container.get("allowed_transformations", []),
            "image_size": cfg.image_size,
        }

    @staticmethod
    def _build_root_config(cfg: GeneratorConfig):
        """Build root generator config from GeneratorConfig.

        Args:
            cfg: GeneratorConfig instance

        Returns:
            Validated root generator config dict
        """
        gen_config = {
            "min_n_shapes_per_grid": cfg.min_n_shapes,
            "max_n_shapes_per_grid": cfg.max_n_shapes,
            "n_examples": cfg.n_examples,
            "min_grid_size": cfg.grid_size,
            "max_grid_size": cfg.grid_size,
            "allowed_transformations": cfg.allowed_transformations,
            "min_transformation_depth": cfg.min_transformation_depth,
            "max_transformation_depth": cfg.max_transformation_depth,
            "shape_compulsory_conditionals": [
                f"is_shape_less_than_{cfg.max_shape_size}_rows",
                f"is_shape_less_than_{cfg.max_shape_size}_cols",
                "is_shape_fully_connected",
            ],
            "saving_path": None,
        }
        return ConfigValidator.model_validate(gen_config)

    def generate_task(self):
        """
        Generate a single task.

        Returns:
            dict: Task dictionary with 'input', 'output', and metadata
        """
        task = self.gen.generate_single_task()
        if len(task.keys()) == 0:
            raise ValueError("Generated task is empty")
        return self._format_task(task)

    def generate_task_hw(self) -> np.ndarray:
        """
        Generate a single task and return input as image at configured size.

        Returns:
            np.ndarray: Input image array of shape (3, image_size, image_size)
        """
        task = self.gen.generate_single_task()
        input_grid = np.array(task["pairs"][-1]["input"])  # (H, W)

        # Convert grid to RGB image using colormap
        from arcworld.constants import COLORMAP, NORM
        from PIL import Image

        input_image = np.array(COLORMAP(NORM(input_grid)))[:, :, :3]  # HWC format, RGB

        # Resize to configured image size
        resize_method = (
            Image.Resampling.NEAREST
            if self.upscale_method == "nearest"
            else Image.Resampling.BILINEAR
        )
        img_pil = Image.fromarray((input_image * 255).astype(np.uint8))
        img_resized = img_pil.resize((self.image_size, self.image_size), resize_method)
        img_array = np.array(img_resized).astype(np.float32) / 255.0  # HWC format

        # Convert to CHW format
        img_chw = np.transpose(img_array, (2, 0, 1))  # (3, image_size, image_size)

        return img_chw

    def generate_tasks(self, n_tasks: int | None = None):
        """
        Generate multiple tasks.

        Args:
            n_tasks(int | None): Number of tasks to generate. If None, uses cfg.num_tasks

        Returns:
            list[dict]: List of task dictionaries
        """
        if n_tasks is None:
            n_tasks = self.cfg.num_tasks
        tasks = []
        for i in range(n_tasks):
            try:
                task = self.generate_task()
                tasks.append(task)
                _logger.info(f"Generated {i + 1}/{n_tasks} tasks")
            except Exception as e:
                _logger.warning(f"Failed to generate task {i + 1}: {e}")
                continue
        return tasks

    def _format_task(self, task):
        """Format task to standard output format."""
        task_dict = {
            "task_key": generate_key(),
            "input": np.int_(task["pairs"][-1]["input"]).tolist(),
            "output": np.int_(task["pairs"][-1]["output"]).tolist(),
            "transformation_suite": task["transformation_suite"],
        }

        # Add demo examples if available
        if len(task["pairs"]) > 1:
            task_dict["demo_input"] = np.int_(task["pairs"][0]["input"]).tolist()
            task_dict["demo_output"] = np.int_(task["pairs"][0]["output"]).tolist()

        return task_dict

    def save_tasks(self, tasks, filename: str, output_file: str | None = None):
        """
        Save tasks to a JSON file.

        Args:
            tasks: List of task dictionaries
            output_file: Path to save the JSON file
        """
        if output_file is None:
            output_file = self.output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(tasks, f)

        _logger.info(f"Saved {len(tasks)} tasks to {output_file}")

    def to_images(self, task):
        """
        Convert task input and output grids to images.

        Args:
            task: Task dictionary
        Returns:
            tuple: (input_image, output_image) as numpy arrays in HWC format
        """
        from arcworld.constants import COLORMAP, NORM

        input_grid = np.array(task["input"])
        output_grid = np.array(task["output"])
        input_image = np.array(COLORMAP(NORM(input_grid)))[:, :, :3]  # HWC format, RGB
        output_image = np.array(COLORMAP(NORM(output_grid)))[:, :, :3]
        return input_image, output_image

    def save_image(self, image, filename: str, output_file: str | None = None):
        """
        Save an image array to a file.

        Args:
            image: Numpy array of the image in HWC format, values in [0, 1]
        """
        if output_file is None:
            output_file = self.output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        path = os.path.join(os.path.dirname(output_file), filename)
        from PIL import Image

        # Ensure it's in HWC format
        if image.ndim == 3 and image.shape[0] == 3:
            # Convert CHW to HWC
            image = np.transpose(image, (1, 2, 0))

        img = Image.fromarray((image * 255).astype(np.uint8))
        img.save(path)
        _logger.info(f"Saved image to {path}")


class DatasetGenerator:
    """Parallel cache generator using multiple worker processes.

    This class manages worker processes that generate samples in parallel
    and saves them to an HDF5 cache file efficiently.
    """

    def __init__(self, cfg: GeneratorConfig):
        """
        Initialize the parallel cache generator.

        Args:
            cfg: GeneratorConfig instance with all generation parameters
            cache_path: Optional path to HDF5 cache file. If None, uses 'data/sample_cache.h5'
            num_workers: Number of parallel worker processes for generation
        """
        self.cfg = cfg

        # Build root generator config
        self.root_gen_config = TaskGenerator._build_root_config(cfg)

        # Setup cache config
        # Convert config to dict (supports both dataclass and OmegaConf)
        cfg_container = _config_to_dict(cfg)

        if not isinstance(cfg_container, dict):
            raise ValueError("Failed to convert GeneratorConfig to dict")

        self.cache_config = {
            "grid_size": cfg.grid_size,
            "min_n_shapes": cfg.min_n_shapes,
            "max_n_shapes": cfg.max_n_shapes,
            "max_shape_size": cfg.max_shape_size,
            "allowed_transformations": cfg_container.get("allowed_transformations", []),
            "image_size": cfg.image_size,
        }

        self.image_size = cfg.image_size
        self.upscale_method = cfg.upscale_method

    def generate(
        self,
        num_samples: int | None = None,
        buffer_size: int = 1000,
        save_batch_size: int = 100,
    ):
        """
        Generate samples in parallel and save to HDF5 cache file.

        Workers generate samples and put them in a shared queue.
        Only the main process writes to the H5 file to avoid contention.

        Args:
            num_samples: Number of samples to generate. If None, uses cfg.num_tasks
            buffer_size: Size of the sample buffer queue
            save_batch_size: Batch size for saving to disk

        Returns:
            Path to cache file
        """
        if num_samples is None:
            num_samples = self.cfg.num_tasks

        _logger.info(
            f"Generating {num_samples} samples to store: {self.cfg.output_file}"
        )
        _logger.info(f"Using {self.cfg.num_workers} worker processes")

        # Initialize store (only main process will write to it)
        store = HDF5CogitaoStore(
            path=self.cfg.output_file,
            generator_config=self.cache_config,
        )

        # Create shared queue for all workers to put their samples
        sample_queue = mp.Queue(maxsize=buffer_size)

        # Start worker processes - they only generate and queue samples
        workers = []
        _logger.info("Starting worker processes...")
        for i in range(self.cfg.num_workers):
            p = mp.Process(
                target=_sample_generation_worker,
                args=(
                    i,
                    sample_queue,
                    self.root_gen_config,
                    self.image_size,
                    self.upscale_method,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        _logger.info(f"All {len(workers)} workers started and generating samples")

        # Main process: collect samples from queue and write to H5 in batches
        batch = []
        total_saved = 0

        try:
            with tqdm(total=num_samples, desc="Saving to store") as pbar:
                while total_saved < num_samples:
                    try:
                        # Get sample from shared queue (workers are adding to this)
                        sample = sample_queue.get(timeout=5.0)
                        batch.append(sample)

                        # Main process writes batch to H5 when full
                        if len(batch) >= save_batch_size:
                            store.save_batch(batch)
                            total_saved += len(batch)
                            pbar.update(len(batch))
                            batch = []

                        # Stop if we have enough samples
                        if total_saved >= num_samples:
                            break

                    except Exception as e:
                        # Check if workers are still alive
                        alive_workers = sum(1 for w in workers if w.is_alive())
                        if alive_workers == 0:
                            _logger.error("All workers have died!")
                            break
                        # Timeout is normal if queue is empty, continue
                        continue

                # Main process writes remaining samples to H5
                if batch and total_saved < num_samples:
                    to_save = batch[: num_samples - total_saved]
                    store.save_batch(to_save)
                    total_saved += len(to_save)
                    pbar.update(len(to_save))

        finally:
            # Cleanup workers
            _logger.info("Stopping worker processes...")
            for w in workers:
                w.terminate()
                w.join(timeout=1.0)

            # Drain any remaining items from queue
            try:
                while not sample_queue.empty() and total_saved < num_samples:
                    sample = sample_queue.get_nowait()
                    batch.append(sample)
                    if len(batch) >= save_batch_size:
                        store.save_batch(batch)
                        total_saved += len(batch)
                        batch = []

                if batch and total_saved < num_samples:
                    to_save = batch[: num_samples - total_saved]
                    store.save_batch(to_save)
                    total_saved += len(to_save)
            except Exception:
                pass

        _logger.info(
            f"Dataset generation complete! Saved {total_saved} samples to {self.cfg.output_file}"
        )

        return self.cfg.output_file
