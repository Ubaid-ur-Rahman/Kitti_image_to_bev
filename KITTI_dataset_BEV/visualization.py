import matplotlib.pyplot as plt
import numpy as np
from .kitti import Kitti

if __name__ == "__main__":
    dataset = Kitti('/srv/hdd/datasets/KITTI/', 'val')
    data = dataset[10]
    bev_image = data['image']  # (496, 432, 3)
    pts = data['pts'][:, :3]  # (N, 3)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Normalize BEV image for visualization
    bev_display = bev_image.mean(axis=2)
    if bev_display.max() > 0:
        bev_display = bev_display / bev_display.max()  # Normalize to [0, 1] for display
    plt.imshow(bev_display, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Camera BEV")

    x, y = pts[:, 0], pts[:, 1]
    x_grid = ((x - 0) / 69.12 * 432).astype(int)
    y_grid = ((y + 39.68) / 79.36 * 496).astype(int)
    x_grid = np.clip(x_grid, 0, 431)
    y_grid = np.clip(y_grid, 0, 495)
    bev_lidar = np.zeros((496, 432))
    bev_lidar[y_grid, x_grid] = 1
    plt.subplot(1, 2, 2)
    plt.imshow(bev_lidar, cmap='viridis')
    plt.title("LiDAR Pillars")

    plt.tight_layout()
    plt.savefig('combined_bev_visualization_updated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # python -m pointpillars.dataset.visualization