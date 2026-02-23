"""
Task Generator for COGITAO training.

This module provides:
1. TaskGenerator: Simple wrapper around COGITAO generator for task generation
2. DatasetGenerator: Parallel cache generator using multiple worker processes
"""

import multiprocessing as mp
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
from .img_transform import to_image

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


def validate_config(cfg: GeneratorConfig):
    """Build root generator config from GeneratorConfig.

    Args:
        cfg: GeneratorConfig instance

    Returns:
        Validated root generator config dict
    """
    return ConfigValidator(
        min_n_shapes_per_grid=cfg.min_n_shapes,
        max_n_shapes_per_grid=cfg.max_n_shapes,
        n_examples=cfg.n_examples,
        min_grid_size=cfg.grid_size,
        max_grid_size=cfg.grid_size,
        allowed_transformations=cfg.allowed_transformations,
        min_transformation_depth=cfg.min_transformation_depth,
        max_transformation_depth=cfg.max_transformation_depth,
        shape_compulsory_conditionals=[
            f"is_shape_less_than_{cfg.max_shape_size}_rows",
            f"is_shape_less_than_{cfg.max_shape_size}_cols",
            "is_shape_more_than_2_cell",
            "is_shape_evenly_colored",
            "is_shape_fully_connected",
        ],
    )


def _sample_generation_worker(
    worker_id, sample_queue, root_gen_config, image_size, upscale_method, shutdown_event
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
        shutdown_event: Multiprocessing Event to signal worker shutdown
    """
    # Create root generator directly with pre-built config
    from arcworld.generator import Generator

    gen = Generator(root_gen_config)

    failures = 0
    max_consecutive_failures = 50

    while not shutdown_event.is_set():
        try:
            # Generate task and convert to image
            task = gen.generate_single_task()
            input_grid = np.array(task["pairs"][-1]["input"])

            # Convert grid to image using shared function
            img_chw = to_image(
                input_grid,
                image_size=image_size,
                upscale_method=upscale_method,
                output_format="CHW",
            )

            # Check shutdown before blocking put operation
            if shutdown_event.is_set():
                break

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

    # Ensure queue doesn't block process exit
    # This allows the process to exit even if queue has buffered items
    sample_queue.cancel_join_thread()
    _logger.debug(f"Worker {worker_id} exiting gracefully (failures: {failures})")


class TaskGenerator:
    """Generator wrapper for creating training sample cache.

    This class wraps the COGITAO generator and provides functionality to:
    1. Generate synthetic task samples
    2. Save them to an HDF5 cache file for training

    The primary use case is to pre-generate a sample cache for training.
    """

    def __init__(self, cfg: GeneratorConfig):
        """
        Initialize the task generator.

        Args:
            cfg: GeneratorConfig instance with all generation parameters
        """
        self.cfg = cfg
        self.output_file = cfg.output_file
        self.image_size = cfg.image_size
        self.upscale_method = cfg.upscale_method

        # Build root generator config from our config
        self.root_gen_config = validate_config(cfg)

        # Initialize the root generator
        self.gen = Generator(self.root_gen_config)

        # Setup cache
        # Convert config to dict (supports both dataclass and OmegaConf)
        cfg_container = _config_to_dict(cfg)
        if not isinstance(cfg_container, dict):
            raise ValueError("Failed to convert GeneratorConfig to dict")

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
        """
        self.cfg = cfg

        # Build root generator config
        self.root_gen_config = validate_config(cfg)

        # Setup cache config
        # Convert config to dict (supports both dataclass and OmegaConf)
        cfg_container = _config_to_dict(cfg)

        if not isinstance(cfg_container, dict):
            raise ValueError("Failed to convert GeneratorConfig to dict")

        self.image_size = cfg.image_size
        self.upscale_method = cfg.upscale_method

    def generate(
        self,
        num_samples: int | None = None,
        *,
        buffer_size: int | None = None,
        save_batch_size: int | None = None,
    ):
        """
        Generate samples in parallel and save to HDF5 cache file.

        Workers generate samples and put them in a shared queue.
        Only the main process writes to the H5 file to avoid contention.

        Args:
            num_samples(int | None = None): Number of samples to generate. If None, uses cfg.num_tasks

        Kwargs:
            buffer_size(int | None = None): Size of the sample buffer queue. If None, uses num_workers * 16
            save_batch_size(int | None = None): Batch size for saving to disk. If None, uses num_workers * 8

        Returns:
            Path to cache file
        """
        if num_samples is None:
            num_samples = self.cfg.num_tasks

        # Default batch size based on workers
        if save_batch_size is None:
            save_batch_size = self.cfg.num_workers * 8

        # Default buffer size for the queue
        if buffer_size is None:
            buffer_size = self.cfg.num_workers * 16

        _logger.info(
            f"Generating {num_samples} samples to store: {self.cfg.output_file}"
        )
        _logger.info(f"Using {self.cfg.num_workers} worker processes")

        # Initialize store (only main process will write to it)
        store = HDF5CogitaoStore(
            path=self.cfg.output_file,
            shape=(3, self.image_size, self.image_size),
            batch_size=self.cfg.batch_size,
        )

        # Create shared queue for all workers to put their samples
        sample_queue = mp.Queue(maxsize=buffer_size)

        # Create shutdown event for graceful worker termination
        shutdown_event = mp.Event()

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
                    shutdown_event,
                ),
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
                        sample = sample_queue.get(timeout=5.0)
                        batch.append(sample)

                        # Write batch
                        if len(batch) >= min(
                            save_batch_size, num_samples - total_saved
                        ):  # Fix to save if limit reached
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
                        time.sleep(0.1)  # Sleep briefly to avoid busy-wait
                        continue

                # Main process writes remaining samples to H5
                if batch and total_saved < num_samples:
                    to_save = batch[: num_samples - total_saved]
                    store.save_batch(to_save)
                    total_saved += len(to_save)
                    pbar.update(len(to_save))

        finally:
            # Cleanup workers gracefully
            _logger.info("Stopping worker processes...")
            shutdown_event.set()  # Signal workers to stop

            # Wait for workers to finish their current task
            for i, w in enumerate(workers):
                w.join(timeout=3.0)
                if w.is_alive():
                    _logger.warning(
                        f"Worker {i} did not exit within timeout, terminating..."
                    )
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
            finally:
                # Properly close the queue
                sample_queue.close()
                sample_queue.join_thread()

        _logger.info(
            f"Dataset generation complete! Saved {total_saved} samples to {self.cfg.output_file}"
        )

        return self.cfg.output_file
