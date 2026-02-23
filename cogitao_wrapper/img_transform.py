from typing import Literal

import numpy as np
from arcworld.constants import COLORMAP, NORM
from PIL import Image


def to_image(
    grid: np.ndarray,
    image_size: int | None = None,
    upscale_method: Literal["nearest", "bilinear"] = "nearest",
    output_format: Literal["HWC", "CHW"] = "CHW",
) -> np.ndarray:
    """Convert cogitao grid to image.

    Args:
        grid: Input grid as numpy array (H, W) with integer values
        image_size: Optional target size for resizing. If None, no resizing is performed
        upscale_method: Method for resizing - 'nearest' or 'bilinear'
        output_format: Output format - 'HWC' (height, width, channels) or 'CHW' (channels, height, width)

    Returns:
        Image array of shape (3, image_size, image_size) if output_format='CHW',
        or (image_size, image_size, 3) if output_format='HWC'.
        Values are float32 in range [0, 1].
    """

    # Convert grid to RGB image using colormap
    input_image = np.array(COLORMAP(NORM(grid)))[:, :, :3]  # HWC format, RGB

    # Resize if requested
    if image_size is not None:
        resize_method = (
            Image.Resampling.NEAREST
            if upscale_method == "nearest"
            else Image.Resampling.BILINEAR
        )
        img_pil = Image.fromarray((input_image * 255).astype(np.uint8))
        img_resized = img_pil.resize((image_size, image_size), resize_method)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
    else:
        img_array = input_image.astype(np.float32)

    # Convert to requested format
    if output_format == "CHW":
        return np.transpose(img_array, (2, 0, 1))
    else:
        return img_array


def to_state(
    image: np.ndarray,
    original_size: int | None = None,
    downscale_method: Literal["nearest", "bilinear"] = "nearest",
    input_format: Literal["HWC", "CHW"] = "CHW",
) -> np.ndarray:
    """Convert image back to cogitao grid.

    This function inverts the to_image transformation by mapping RGB colors
    back to their corresponding integer grid values (0-9).

    Args:
        image: Input image as numpy array. Expected to be float32 in range [0, 1].
               Shape should be (3, H, W) or (B, 3, H, W) if input_format='CHW',
               or (H, W, 3) or (B, H, W, 3) if input_format='HWC'.
        original_size: Optional original grid size for downscaling. If None, no downscaling is performed.
        downscale_method: Method for downscaling - 'nearest' or 'bilinear'
        input_format: Input format - 'HWC' (height, width, channels) or 'CHW' (channels, height, width)

    Returns:
        Grid array of shape (H, W) or (B, H, W) with integer values in range [0, 9].
    """
    # Build color lookup table (10 colors x 3 channels)
    # ARC grids have values 0-9

    batch_size: int | None = None # None <=> (1, C, H, W)
    color_palette = np.zeros((10, 3), dtype=np.float64)
    for i in range(10):
        test_grid = np.array([[i]])
        color_palette[i] = np.array(COLORMAP(NORM(test_grid)))[:, :, :3][0, 0]

    # Convert to HWC format if needed
    if image.ndim == 4:
        if input_format == "CHW":
            # (B, C, H, W) -> (B, H, W, C)
            img_hwc = np.transpose(image, (0, 2, 3, 1))
        else:
            img_hwc = image
        batch_size, height, width = img_hwc.shape[:3]
    else:
        if input_format == "CHW":
            img_hwc = np.transpose(image, (1, 2, 0))
        else:
            img_hwc = image
        height, width = img_hwc.shape[:2]

    # Downscale if requested
    if original_size is not None:
        downscale_method_pil = (
            Image.Resampling.NEAREST
            if downscale_method == "nearest"
            else Image.Resampling.BILINEAR
        )
        if batch_size is not None:
            # Process each image in batch
            # This is slow but PIL doesn't support batches
            imgs_downscaled = []
            for i in range(batch_size):
                img_pil = Image.fromarray((img_hwc[i] * 255).astype(np.uint8))
                img_downscaled = img_pil.resize(
                    (original_size, original_size), downscale_method_pil
                )
                imgs_downscaled.append(np.array(img_downscaled))
            img_hwc = np.stack(imgs_downscaled).astype(np.float32) / 255.0
            height, width = original_size, original_size
        else:
            img_pil = Image.fromarray((img_hwc * 255).astype(np.uint8))
            img_downscaled = img_pil.resize(
                (original_size, original_size), downscale_method_pil
            )
            img_hwc = np.array(img_downscaled).astype(np.float32) / 255.0
            height, width = original_size, original_size

    # Vectorized conversion: find closest color for all pixels using argmin
    # Reshape image to (N, 3) for efficient computation
    # where N = H*W or B*H*W
    pixels = img_hwc.reshape(-1, 3)

    # Compute squared Euclidean distance between each pixel and each color
    # Broadcasting: (N, 1, 3) - (1, 10, 3) = (N, 10, 3)
    distances = np.sum(
        (pixels[:, np.newaxis, :] - color_palette[np.newaxis, :, :]) ** 2, axis=2
    )

    # Find the index of the closest color for each pixel using argmin
    if batch_size is not None:
        grid = (
            np.argmin(distances, axis=1)
            .reshape(batch_size, height, width)
            .astype(np.int32)
        )
    else:
        grid = np.argmin(distances, axis=1).reshape(height, width).astype(np.int32)

    return grid
