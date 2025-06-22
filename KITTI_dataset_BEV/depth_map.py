import numpy as np

def create_sparse_depth_map(lidar_points, tr_velo_to_cam, r0_rect, P2, image_shape):
    """
    Project LiDAR points to the image to create a sparse depth map and associate image features.
    
    Args:
        lidar_points: (N, 4) array of LiDAR points (x, y, z, intensity)
        tr_velo_to_cam: (4, 4) transformation matrix (LiDAR to camera)
        r0_rect: (4, 4) rectification matrix
        P2: (4, 4) projection matrix
        image_shape: Tuple (h, w) of image dimensions (e.g., (375, 1245))
    
    Returns:
        depth_map: (h, w) array with depth values (0 where no LiDAR point projects)
        valid_pixels: (M, 2) array of valid pixel coordinates (u, v)
        valid_depths: (M,) array of corresponding depth values
        valid_indices: (M,) array of indices of valid LiDAR points
    """
    h, w = image_shape
    points = lidar_points[:, :3]
    extended_points = np.pad(points[:, None, :], ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    camera_points = extended_points @ rt_mat.T  # Shape: (N, 1, 4)
    depth_camera = camera_points[:, 0, 2]  # Shape: (N,)
    image_points = camera_points @ P2.T  # Shape: (N, 1, 4)
    z = image_points[:, 0, 2]  # Shape: (N,)
    
    # Create mask for valid points
    mask = (z > 0) & (depth_camera > 0)  # Shape: (N,)
    
    # Project to image coordinates
    image_points = image_points[:, 0, :2] / (z[:, None] + 1e-6)  # Shape: (N, 2)
    u, v = image_points[:, 0], image_points[:, 1]  # Shape: (N,), (N,)
    
    print(f"Camera points x-range before filtering: min={camera_points[:, 0, 0].min()}, max={camera_points[:, 0, 0].max()}")
    print(f"Camera points y-range before filtering: min={camera_points[:, 0, 1].min()}, max={camera_points[:, 0, 1].max()}")
    print(f"Raw u range: min={u.min()}, max={u.max()}")
    print(f"Raw v range: min={v.min()}, max={v.max()}")
    
    # Filter valid pixels within image bounds
    valid = mask & (~np.isnan(u)) & (~np.isnan(v)) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_indices = np.where(valid)[0]
    u, v, depth = u[valid], v[valid], depth_camera[valid]
    
    print(f"Number of valid pixels after projection to image: {len(valid_indices)}")
    print(f"Clipped u range: min={u.min()}, max={u.max()}")
    print(f"Clipped v range: min={v.min()}, max={v.max()}")
    
    depth_map = np.zeros((h, w), dtype=np.float32)
    u, v = u.astype(int), v.astype(int)
    depth_map[v, u] = depth
    
    valid_pixels = np.stack([u, v], axis=-1)
    valid_depths = depth
    
    return depth_map, valid_pixels, valid_depths, valid_indices

def pixels_to_camera_coords(u, v, depth, C):
    """
    Back-project 2D image pixels to 3D camera coordinates using depth and intrinsic matrix.
    
    Args:
        u, v: (M,) arrays of pixel coordinates
        depth: (M,) array of depth values
        C: (3, 3) intrinsic matrix from projection_matrix_to_CRT_kitti
    
    Returns:
        camera_coords: (M, 3) array of 3D camera coordinates (x, y, z)
    """
    C_inv = np.linalg.inv(C)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    camera_coords = (pixels @ C_inv.T) * depth[:, None]
    print(f"Back-projected camera coords x-range: min={camera_coords[:, 0].min()}, max={camera_coords[:, 0].max()}")
    return camera_coords


def points_to_bev_grid(points, features, valid_indices, bev_h, bev_w, x_range, y_range):
    """
    Project 3D points to BEV grid and associate image features.
    
    Args:
        points: (M, 3) array of 3D points in LiDAR coordinates
        features: (M, C) array of image features (e.g., RGB values)
        valid_indices: (K,) array of indices of points with valid image projections
        bev_h, bev_w: BEV grid dimensions (e.g., 496, 432)
        x_range: Tuple (x_min, x_max) of x-coordinate range
        y_range: Tuple (y_min, y_max) of y-coordinate range
    
    Returns:
        bev_map: (bev_h, bev_w, C) array of BEV feature map
    """
    x, y = points[:, 0], points[:, 1]
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Map x, y to grid indices with y-axis flip
    x_grid = ((x - x_min) / (x_max - x_min) * bev_w).astype(int)
    y_grid = bev_h - 1 - ((y - y_min) / (y_max - y_min) * bev_h).astype(int)
    
    # Clip to grid bounds
    x_grid = np.clip(x_grid, 0, bev_w - 1)
    y_grid = np.clip(y_grid, 0, bev_h - 1)
    
    print(f"LiDAR coords range: x_min={x.min()}, x_max={x.max()}, y_min={y.min()}, y_max={y.max()}")
    print(f"Grid indices range: x_grid_min={x_grid.min()}, x_grid_max={x_grid.max()}, y_grid_min={y_grid.min()}, y_grid_max={y_grid.max()}")
    
    # Initialize BEV map
    bev_map = np.zeros((bev_h, bev_w, features.shape[-1]), dtype=np.float32)
    counts = np.zeros((bev_h, bev_w), dtype=np.float32)
    
    # Aggregate features for valid points
    for i in valid_indices:
        if not np.any(np.isnan(features[i])):  # Skip if features contain NaN
            bev_map[y_grid[i], x_grid[i]] += features[i]
            counts[y_grid[i], x_grid[i]] += 1
    
    # Normalize by count with epsilon to avoid division by zero
    valid = counts > 0
    bev_map[valid] = bev_map[valid] / (counts[valid][..., None] + 1e-6)
    
    # Check for NaN in BEV map
    if np.any(np.isnan(bev_map)):
        print("Warning: NaN values detected in BEV map!")
        bev_map = np.nan_to_num(bev_map, nan=0.0)
    
    print(f"BEV map range: min={bev_map.min()}, max={bev_map.max()}, mean={bev_map[valid].mean() if np.any(valid) else 0.0}")
    print(f"Non-zero cells in BEV map: {np.sum(counts > 0)}")
    
    return bev_map