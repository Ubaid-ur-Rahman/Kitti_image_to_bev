import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset
from transformers import pipeline
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from .utils import read_pickle, read_points, bbox_camera2lidar, projection_matrix_to_CRT_kitti, points_camera2lidar
from .data_aug import data_augment
from .depth_map import create_sparse_depth_map, pixels_to_camera_coords, points_to_bev_grid


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Kitti(Dataset):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced', image_size=(1245, 375)):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.pts_prefix = pts_prefix
        self.data_infos = read_pickle(os.path.join(data_root, f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        db_infos = read_pickle(os.path.join(data_root, 'kitti_dbinfos_train.pkl'))
        db_infos = self.filter_db(db_infos)

        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config = dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10)
            ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            ), 
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]             
        )
        
    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        return db_infos
    
    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = data_info['image'], data_info['calib'], data_info['annos']

        # Point cloud input
        velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix)
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path)
        print(f"Total LiDAR points: {len(pts)}")
        print(f"Original LiDAR points range: x_min={pts[:, 0].min()}, x_max={pts[:, 0].max()}, y_min={pts[:, 1].min()}, y_max={pts[:, 1].max()}")

        # Image input
        image_path = os.path.join(self.data_root, image_info['image_path'])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)  # (375, 1245, 3)
        image = image.astype(np.float32) / 255.0
        print(f"Image shape: {image.shape}")

        # Calibration input
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)
        print("tr_velo_to_cam:\n", tr_velo_to_cam)
        print("r0_rect:\n", r0_rect)
        print("P2:\n", P2)

        # Transform image to BEV
        depth_map, valid_pixels, valid_depths, valid_indices = create_sparse_depth_map(
            pts, tr_velo_to_cam, r0_rect, P2, self.image_size
        )
        print(f"Number of valid pixels after projection to image: {len(valid_pixels)}")

        # Initialize features with default values
        mean_rgb = image.mean(axis=(0, 1)) if len(valid_pixels) > 0 else np.array([0.5, 0.5, 0.5], dtype=np.float32)
        image_features = np.tile(mean_rgb[None, :], (len(pts), 1))

        # Assign image features to valid LiDAR points
        if len(valid_pixels) > 0:
            u, v = valid_pixels[:, 0], valid_pixels[:, 1]
            image_features[valid_indices] = image[v, u]
            print(f"RGB features range: min={image_features.min()}, max={image_features.max()}, mean={image_features.mean()}")

        # Rasterize to BEV grid
        bev_h, bev_w = 496, 432
        x_range = (0, 69.12)
        y_range = (-39.68, 39.68)
        bev_image = points_to_bev_grid(
            pts[:, :3], image_features, valid_indices, bev_h, bev_w, x_range, y_range
        )
        print(f"Non-zero cells in BEV image: {np.sum(bev_image > 0)}")
        image = bev_image

        # Annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels),
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info,
            'image': image
        }
        if self.split in ['train', 'trainval']:
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)

        return data_dict
    
    
    def __len__(self):
        return len(self.data_infos)
 

if __name__ == '__main__':
    
    kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
                       split='train')
    kitti_data.__getitem__(9)
def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    '''
    points: shape=(N, 8, 3) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 8, 3)
    '''
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]