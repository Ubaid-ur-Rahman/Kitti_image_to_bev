import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import multiprocessing as mp


def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    batched_image_tensor_list = []
    
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_info, calib_info = data_dict['image_info'], data_dict['calib_info']
        image = data_dict['image']

        batched_pts_list.append(torch.from_numpy(pts).float())  # Ensure float32
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d).float())  # Ensure float32
        batched_labels_list.append(torch.from_numpy(gt_labels).long())  # Ensure int64
        batched_names_list.append(gt_names)
        batched_difficulty_list.append(torch.from_numpy(difficulty).long())  # Ensure int64
        batched_img_list.append(image_info)
        batched_calib_list.append(calib_info)
        batched_image_tensor_list.append(torch.from_numpy(image).permute(2, 0, 1).float())  # Ensure float32
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list,
        batched_images=torch.stack(batched_image_tensor_list)
    )
    return rt_data_dict

def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate_fn,
        pin_memory=True  # Optimize data transfer to GPU
    )
