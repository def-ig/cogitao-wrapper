import numpy as np

from .img_transform import to_state


def per_pixel_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compares pixels of two image arrays
    preds: (B, C, H, W)
    targets: (B, C, H, W)
    """
    return float(
        (preds == targets).mean()
    )  # Slight color variations will lead to massive inaccuracy...


def object_location_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compares image to target grid for non-white pixels overlap. Transforms images to cogitao grids first.
    preds: (B, C, H, W)
    targets: (B, grid_size, grid_size): Target grid
    """
    preds_grid = to_state(preds, original_size=targets.shape[-1])

    # Mask non-white pixels: essentially do >0
    preds_mask = preds_grid > 0
    targets_mask = targets > 0

    return float((preds_mask == targets_mask).mean())


def object_location_accuracy_target_image(
    preds: np.ndarray, targets: np.ndarray, grid_size: int | None = None
) -> float:
    """
    Compares two images for non-white pixels overlap, aka same object pattern. Transforms images to cogitao grids first.
    preds: (B, C, H, W) image batch
    targets: (B, C, H, W) image batch
    """
    preds_grid = to_state(preds, original_size=grid_size)
    targets_grid = to_state(targets, original_size=grid_size)

    # Mask non-white pixels: essentially do >0
    preds_mask = preds_grid > 0
    targets_mask = targets_grid > 0

    return float((preds_mask == targets_mask).mean())


def _extract_objects(grid: np.ndarray) -> list[np.ndarray]:
    """
    Extracts connected components of non-zero pixels as separate object grids.
    Each returned grid has the same shape as input, containing one object.
    """
    grid_working = grid.copy()
    objects = []

    while np.any(grid_working > 0):
        # find first colored pixel
        y, x = np.argwhere(grid_working > 0)[0]
        color = grid_working[y, x]
        # find all connected pixels of the same color
        object_mask = np.zeros_like(grid_working, dtype=bool)
        object_mask[y, x] = True
        # use BFS or DFS to find all connected pixels of the same color
        queue = {
            (y, x),
        }
        while queue:
            y, x = queue.pop()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < grid_working.shape[0]
                    and 0 <= nx < grid_working.shape[1]
                    and grid_working[ny, nx] == color
                    and not object_mask[ny, nx]
                ):
                    object_mask[ny, nx] = True
                    queue.add((ny, nx))

        # Append object to objects
        # Create a grid for the object to preserve its shape
        obj_grid = np.zeros_like(grid_working)
        obj_grid[object_mask] = grid_working[object_mask]
        objects.append(obj_grid)

        # remove object from grid
        grid_working[object_mask] = 0

    return objects


def number_of_perfectly_reconstructed_objects(
    objects: np.ndarray, target: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Checks how many objects were perfectly reconstructed.

    Args:
        objects: (N, C, H, W) images of N objects for one grid OR (C, H, W) single image
        target: (grid_size, grid_size) Resulting grid

    Returns:
        tuple[int, int, int, int]: (found, duplicates, missed, extra)
    """

    # 1. Extract objects from target grid
    target_objects = _extract_objects(target)

    # 2. Process objects input
    # Convert input to grids
    # If objects is (C, H, W), to_state returns (H, W) -> we need to extract objects from this grid
    # If objects is (N, C, H, W), to_state returns (N, H, W) -> these are already separated objects
    objects_state = to_state(objects, original_size=target.shape[-1])

    if objects_state.ndim == 2:
        # Single grid (H, W), extract objects
        found_object_grids = _extract_objects(objects_state)
    else:
        # List of grids (N, H, W), assume already separated
        found_object_grids = [g for g in objects_state]

    target_objects_found = [False] * len(target_objects)

    found = 0
    duplicates = 0
    missed = 0

    # Compare each object from the grid to all objects in the input
    for i, target_object in enumerate(target_objects):
        for j, found_object in enumerate(found_object_grids):
            if np.array_equal(target_object, found_object):
                if target_objects_found[i]:
                    duplicates += 1
                else:
                    target_objects_found[i] = True
                    found += 1
        if not target_objects_found[i]:
            missed += 1

    extra = len(found_object_grids) - found - duplicates

    return found, duplicates, missed, extra


def number_of_perfectly_reconstructed_objects_batch(
    objects: np.ndarray, target: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Checks how many objects were perfectly reconstructed.

    Args:
        objects: (B, N, C, H, W) images of N objects for one grid
        target: (B, grid_size, grid_size) Resulting grid

    Returns:
        tuple[int, int, int, int]: (found, duplicates, missed, extra)
    """

    found = 0
    duplicates = 0
    missed = 0
    extra = 0

    for i in range(objects.shape[0]):
        found_i, duplicates_i, missed_i, extra_i = (
            number_of_perfectly_reconstructed_objects(objects[i], target[i])
        )
        found += found_i
        duplicates += duplicates_i
        missed += missed_i
        extra += extra_i

    return found, duplicates, missed, extra


def number_of_perfectly_reconstructed_objects_target_image(
    preds, targets: np.ndarray, grid_size: int | None = None
) -> float:
    """
    Compares two images for non-white pixels overlap, aka same object pattern. Transforms images to cogitao grids first.
    preds: (B, C, H, W) image batch
    targets: (B, C, H, W) image batch
    """
    targets_grid = to_state(targets, original_size=grid_size)

    return number_of_perfectly_reconstructed_objects(preds, targets_grid)


def number_of_perfectly_reconstructed_objects_target_image_batch(
    preds, targets: np.ndarray, grid_size: int | None = None
) -> float:
    """
    Compares two images for non-white pixels overlap, aka same object pattern. Transforms images to cogitao grids first.
    preds: (B, C, H, W) image batch
    targets: (B, C, H, W) image batch
    """
    targets_grid = to_state(targets, original_size=grid_size)

    return number_of_perfectly_reconstructed_objects_batch(preds, targets_grid)


def compare_reconstruction_images(
    preds: np.ndarray, targets: np.ndarray, grid_size: int | None = None
) -> dict[str, float]:
    """
    Returns an aggregate of all metrics for a batch of images.

    preds: (B, C, H, W)
    targets: (B, C, H, W)
    """
    found, duplicates, missed, extra = (
        number_of_perfectly_reconstructed_objects_target_image_batch(
            preds, targets, grid_size
        )
    )
    return {
        "per_pixel_accuracy": per_pixel_accuracy(preds, targets),
        "object_location_accuracy": object_location_accuracy_target_image(
            preds, targets, grid_size
        ),
        "number_of_perfectly_reconstructed_objects": {
            "found": found,
            "duplicates": duplicates,
            "missed": missed,
            "extra": extra,
        },
    }
