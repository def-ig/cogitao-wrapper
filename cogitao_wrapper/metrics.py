import numpy as np

from .img_transform import to_state


def per_pixel_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
) -> float:
    """
    Accuracy of grid recreation from two image arrays.

    targets: (B, C, H, W) target images
    preds: (B, C, H, W) predicted images
    """
    targets_grid = to_state(targets, original_size=targets.shape[-1])
    preds_grid = to_state(preds, original_size=targets.shape[-1])

    return float((targets_grid == preds_grid).mean())


def object_location_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
) -> float:
    """
    Compares grid of predicted image to target grid using IOU for non-white pixels.

    targets: (B, C, H, W) target images
    preds: (B, grid_size, grid_size) predicted grids
    """
    preds_grid = to_state(preds, original_size=targets.shape[-1])

    # Mask non-white pixels: essentially do >0
    preds_mask = preds_grid > 0
    targets_mask = targets > 0

    # Result = IOU for non-white pixels
    return float((preds_mask & targets_mask).sum() / (preds_mask | targets_mask).sum())


def object_location_accuracy_target_image(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """
    Compares grid of predicted image to grid of target image using IOU for non-white pixels.

    targets: (B, C, H, W) target images
    preds: (B, C, H, W) predicted images
    """
    preds_grid = to_state(preds, original_size=grid_size)
    targets_grid = to_state(targets, original_size=grid_size)

    # Mask non-white pixels: essentially do >0
    preds_mask = preds_grid > 0
    targets_mask = targets_grid > 0

    # Result = IOU for non-white pixels
    return float((preds_mask & targets_mask).sum() / (preds_mask | targets_mask).sum())


def _extract_objects(grid: np.ndarray) -> list[np.ndarray]:
    """
    Extracts connected components of non-zero pixels as separate object grids.
    Each returned grid has the same shape as input, containing one object.

    grid: (H, W) grid
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
    target: np.ndarray,
    preds: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Checks how many objects were perfectly reconstructed.

    Args:
        target: (grid_size, grid_size) Resulting grid
        preds: (N, C, H, W) images of N objects for one grid OR (C, H, W) single image

    Returns:
        tuple[int, int, int, int]: (found, duplicates, missed, extra)
    """

    # 1. Extract objects from target grid
    target_objects = _extract_objects(target)

    # 2. Process objects input
    # Convert input to grids
    # If objects is (C, H, W), to_state returns (H, W) -> we need to extract objects from this grid
    # If objects is (N, C, H, W), to_state returns (N, H, W) -> these are already separated objects
    preds_grids = to_state(preds, original_size=target.shape[-1])

    if preds_grids.ndim == 2:
        # Single grid (H, W), extract objects
        found_object_grids = _extract_objects(preds_grids)
    else:
        # List of grids (N, H, W), assume already separated
        found_object_grids = [g for g in preds_grids]

    target_objects_found = [False] * len(target_objects)

    found = 0
    duplicates = 0
    missed = 0

    # Look if all target objects were found
    for i, target_object in enumerate(target_objects):
        # Look up for a match in found objects
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
    target: np.ndarray,
    preds: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Checks how many objects were perfectly reconstructed.

    Args:
        target: (B, grid_size, grid_size) Resulting grid
        preds: (B, N, C, H, W) images of N objects for one grid

    Returns:
        tuple[int, int, int, int]: (found, duplicates, missed, extra)
    """

    found = 0
    duplicates = 0
    missed = 0
    extra = 0

    for i in range(target.shape[0]):
        found_i, duplicates_i, missed_i, extra_i = (
            number_of_perfectly_reconstructed_objects(target[i], preds[i])
        )
        found += found_i
        duplicates += duplicates_i
        missed += missed_i
        extra += extra_i

    return found, duplicates, missed, extra


def number_of_perfectly_reconstructed_objects_target_image(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """
    Compares two images for non-white pixels overlap, aka same object pattern. Transforms images to cogitao grids first.
    targets: (C, H, W) image
    preds: (C, H, W) image
    """
    targets_grid = to_state(targets, original_size=grid_size)

    return number_of_perfectly_reconstructed_objects(targets_grid, preds)


def number_of_perfectly_reconstructed_objects_target_image_batch(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """
    Compares two images for non-white pixels overlap, aka same object pattern. Transforms images to cogitao grids first.
    targets: (B, C, H, W) image batch
    preds: (B, C, H, W) image batch
    """
    targets_grid = to_state(targets, original_size=grid_size)

    return number_of_perfectly_reconstructed_objects_batch(targets_grid, preds)


def compare_reconstruction_images(
    targets: np.ndarray,
    preds: np.ndarray,
    objects: np.ndarray | None = None,
    *,
    grid_size: int | None = None,
) -> dict[str, float]:
    """
    Returns an aggregate of all metrics for a batch of images.

    targets: (B, C, H, W) target images
    preds: (B, C, H, W) predicted images
    objects: (B, N, C, H, W) or (B, C, H, W) objects
    """
    if objects is not None:
        found, duplicates, missed, extra = (
            number_of_perfectly_reconstructed_objects_target_image_batch(
                targets, objects
            )
        )
    else:
        found, duplicates, missed, extra = (
            number_of_perfectly_reconstructed_objects_target_image_batch(targets, preds)
        )
    return {
        "per_pixel_accuracy": per_pixel_accuracy(
            targets,
            preds,
        ),
        "object_location_accuracy": object_location_accuracy_target_image(
            targets, preds, grid_size=grid_size
        ),
        "number_of_perfectly_reconstructed_objects": {
            "found": found,
            "duplicates": duplicates,
            "missed": missed,
            "extra": extra,
        },
    }
