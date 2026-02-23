from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from arcworld.constants import COLORMAP, NORM
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import CogitaoDataset


def color_analysis(
    dataset_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    output_path: str | Path = "color_distribution.png",
):
    """
    Analyze the color distribution of the dataset.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        output_path: Path to save the output plot
    """
    dataset = CogitaoDataset(dataset_path)

    # Build color palette (10 colors)
    color_palette = np.zeros((10, 3), dtype=np.float64)
    for i in range(10):
        test_grid = np.array([[i]])
        color_palette[i] = np.array(COLORMAP(NORM(test_grid)))[:, :, :3][0, 0]

    color_palette_t = torch.tensor(color_palette, dtype=torch.float32)

    # Optimization: move palette to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_palette_t = color_palette_t.to(device)

    # Create DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    color_counts = np.zeros(10, dtype=np.int64)

    print("Analyzing dataset colors...")
    for batch in tqdm(loader, desc="Batches"):
        # batch["imgs"] shape: (B, 3, H, W)
        imgs = batch["imgs"].to(device)
        B, C, H, W = imgs.shape

        # Reshape to (B*H*W, 3)
        pixels = imgs.permute(0, 2, 3, 1).reshape(-1, 3)

        # Distances to color_palette
        # (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
        distances = torch.sum(
            (pixels.unsqueeze(1) - color_palette_t.unsqueeze(0)) ** 2, dim=2
        )
        closest_colors = torch.argmin(distances, dim=1)

        counts = torch.bincount(closest_colors, minlength=10)
        color_counts += counts.cpu().numpy()

    # Plot
    total_pixels = color_counts.sum()
    percentages = (color_counts / total_pixels) * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(10), color_counts, color=color_palette, edgecolor="black", linewidth=1.5
    )
    plt.xlabel("Color Index")
    plt.ylabel("Pixel Count")
    plt.title("Color Distribution in Dataset")
    plt.xticks(range(10))

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{percentages[i]:.2f}%",
            ha="center",
            va="bottom",
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Color analysis saved to {output_path}")


def plot_dataset_examples(
    dataset_path: str | Path, output_path: str | Path, num_images: int = 64
):
    """
    Load a dataset and save a grid of images from it.
    """
    dataset = CogitaoDataset(dataset_path)
    print(f"Dataset '{dataset_path}' has {len(dataset)} items.")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = []
    num_to_print = min(num_images, len(dataset))

    for batch in dataloader:
        if len(images) >= num_to_print:
            break
        img = batch["imgs"]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Ensure image is [C, H, W]
        if len(img.shape) == 4:
            # If multiple images per task, take the first one or flatten
            img = img[0]

        images.append(img)

    print(f"Collected {len(images)} images.")

    if len(images) > 0:
        # Stack images into a batch [B, C, H, W]
        img_batch = torch.stack(images)

        import math

        B, C, H, W = img_batch.shape
        nrow = math.ceil(math.sqrt(B))
        ncol = math.ceil(B / nrow)

        if B < nrow * ncol:
            padding = torch.zeros(
                (nrow * ncol - B, C, H, W),
                dtype=img_batch.dtype,
                device=img_batch.device,
            )
            img_batch = torch.cat((img_batch, padding), dim=0)

        # Create grid layout
        grid = img_batch.view(nrow, ncol, C, H, W)
        grid = grid.permute(2, 0, 3, 1, 4).contiguous().view(C, nrow * H, ncol * W)

        # Convert to numpy and shape [H_total, W_total, C]
        image_np = grid.permute(1, 2, 0).numpy()

        # PIL needs uint8 array type
        if np.issubdtype(image_np.dtype, np.floating):
            # Scale if maximum is <= 1.0 (some margins for float accuracy)
            if image_np.max() <= 1.001:
                image_np = image_np * 255.0
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)  # For grayscale PIL saving

        # Save as a grid using PIL
        PIL.Image.fromarray(image_np).save(output_file)
        print(f"Saved image grid to {output_file}")
    else:
        print("No images found to process.")
