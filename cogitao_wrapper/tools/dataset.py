from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
