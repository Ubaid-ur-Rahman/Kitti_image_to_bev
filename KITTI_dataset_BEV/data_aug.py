import copy
import numba
import numpy as np
import os
import pdb
from pointpillars.utils import bbox3d2bevcorners, box_collision_test, read_points, \
    remove_pts_in_bboxes, limit_period
from .depth_map import create_sparse_depth_map
from scipy.interpolate import griddata


import numpy as np
import os
import copy
from scipy.interpolate import griddata
import numba

def sample_objects_from_database(class_map, root_path, input_dict, sampler_dict, group_sampling_cfg):
    point_cloud, bbox3d_gt = input_dict['pts'], input_dict['gt_bboxes_3d']
    labels_gt, names_gt = input_dict['gt_labels'], input_dict['gt_names']
    level_gt = input_dict['difficulty']
    img_info, calib_data = input_dict['image_info'], input_dict['calib_info']
    image_rgb = input_dict['image']

    mat_velo2cam = calib_data['Tr_velo_to_cam'].astype(np.float32)
    mat_r0 = calib_data['R0_rect'].astype(np.float32)
    mat_proj = calib_data['P2'].astype(np.float32)

    all_sampled_pts, all_sampled_names, all_sampled_labels = [], [], []
    all_sampled_boxes, all_sampled_difficulties = [], []

    current_boxes = copy.deepcopy(bbox3d_gt)
    
    for category, count in group_sampling_cfg.items():
        required_num = count - np.sum(names_gt == category)
        if required_num <= 0:
            continue

        candidates = sampler_dict[category].sample(required_num)
        boxes_from_db = np.array([c['box3d_lidar'] for c in candidates], dtype=np.float32)

        current_box_corners = bbox3d2bevcorners(current_boxes)
        new_box_corners = bbox3d2bevcorners(boxes_from_db)

        all_corners = np.concatenate([current_box_corners, new_box_corners], axis=0)
        collision = box_collision_test(all_corners, all_corners)

        total_existing = len(current_box_corners)
        new_valid_boxes = []

        for i in range(total_existing, len(collision)):
            if np.any(collision[i]):
                collision[i] = False
                collision[:, i] = False
                continue

            current_item = candidates[i - total_existing]
            point_file = os.path.join(root_path, current_item['path'])
            object_pts = read_points(point_file)
            object_pts[:, :3] += current_item['box3d_lidar'][:3]

            depth_img, px_coords, px_depths, valid_idx = create_sparse_depth_map(
                object_pts, mat_velo2cam, mat_r0, mat_proj, image_rgb.shape[:2]
            )

            rgb_feats = np.zeros((len(object_pts), 3), dtype=np.float32)
            if len(px_coords) > 0:
                u_coords, v_coords = px_coords[:, 0], px_coords[:, 1]
                rgb_feats[valid_idx] = image_rgb[v_coords, u_coords]

                sampled_xy = object_pts[valid_idx, :2]
                interpolated = griddata(sampled_xy, rgb_feats[valid_idx], object_pts[:, :2], method='nearest', fill_value=image_rgb.mean(axis=(0, 1)))
                rgb_feats = interpolated
            else:
                rgb_feats[:] = image_rgb.mean(axis=(0, 1))

            print(f"[{category}] Raw RGB stats -> min: {rgb_feats.min()}, max: {rgb_feats.max()}, mean: {rgb_feats.mean()}")

            rgb_feats = (rgb_feats - rgb_feats.mean()) / (rgb_feats.std() + 1e-6)
            print(f"[{category}] Normalized RGB stats -> min: {rgb_feats.min()}, max: {rgb_feats.max()}, mean: {rgb_feats.mean()}")

            object_pts = np.concatenate([object_pts, rgb_feats], axis=1).astype(np.float32)
            all_sampled_pts.append(object_pts)
            all_sampled_names.append(current_item['name'])
            all_sampled_labels.append(class_map[current_item['name']])
            all_sampled_boxes.append(current_item['box3d_lidar'])
            new_valid_boxes.append(current_item['box3d_lidar'])
            all_sampled_difficulties.append(current_item['difficulty'])

        new_valid_boxes = np.array(new_valid_boxes, dtype=np.float32).reshape(-1, 7)
        current_boxes = np.concatenate([current_boxes, new_valid_boxes], axis=0)

    if len(all_sampled_boxes) > 0:
        filtered_pts = remove_pts_in_bboxes(point_cloud, np.stack(all_sampled_boxes, axis=0))
        combined_sampled_pts = np.concatenate(all_sampled_pts, axis=0) if all_sampled_pts else np.zeros((0, 7), dtype=np.float32)
        point_cloud = np.concatenate([combined_sampled_pts, filtered_pts], axis=0).astype(np.float32)

    bbox3d_gt = current_boxes.astype(np.float32)
    labels_gt = np.concatenate([labels_gt, np.array(all_sampled_labels, dtype=np.int64)], axis=0)
    names_gt = np.concatenate([names_gt, np.array(all_sampled_names)], axis=0)
    level_gt = np.concatenate([level_gt, np.array(all_sampled_difficulties)], axis=0)

    updated_dict = {
        'pts': point_cloud,
        'gt_bboxes_3d': bbox3d_gt,
        'gt_labels': labels_gt,
        'gt_names': names_gt,
        'difficulty': level_gt,
        'image_info': img_info,
        'calib_info': calib_data,
        'image': image_rgb
    }
    return updated_dict


@numba.jit(nopython=True)
def apply_noise_to_objects(point_array, box3d_array, box_bev_corners, translations, rotations_rad, rotation_matrices, box_masks):
    n_boxes, n_trials = translations.shape[:2]
    success_status = -np.ones(n_boxes, dtype=np.int_)

    for box_idx in range(n_boxes):
        for trial in range(n_trials):
            orig_corners = box_bev_corners[box_idx] - box3d_array[box_idx, :2]
            rot = rotation_matrices[box_idx, trial]
            trans = translations[box_idx, trial]
            rotated = orig_corners @ rot
            rotated += box3d_array[box_idx, :2] + trans[:2]
            if box_collision_test(np.expand_dims(rotated, 0), box_bev_corners)[0, box_idx]:
                continue
            box_bev_corners[box_idx] = rotated
            success_status[box_idx] = trial
            break

    visited = {}
    for box_idx in range(n_boxes):
        trial_idx = success_status[box_idx]
        if trial_idx == -1:
            continue
        delta = translations[box_idx, trial_idx]
        angle = rotations_rad[box_idx, trial_idx]
        rot_matrix = rotation_matrices[box_idx, trial_idx]
        for pt_idx in range(len(point_array)):
            if box_masks[pt_idx][box_idx] and pt_idx not in visited:
                pt = point_array[pt_idx]
                offset = pt[:3] - box3d_array[box_idx, :3]
                offset[:2] = offset[:2] @ rot_matrix
                offset += box3d_array[box_idx, :3] + delta
                pt[:3] = offset
                visited[pt_idx] = 1
        box3d_array[box_idx, :3] += delta
        box3d_array[box_idx, 6] += angle

    return box3d_array, point_array


def perturb_object_positions(data, num_trials=100, trans_std=(0.5, 0.5, 0.5), rotation_bounds=(-0.785, 0.785)):
    pc, bboxes = data['pts'], data['gt_bboxes_3d']
    n_bboxes = len(bboxes)

    translations = np.random.normal(scale=trans_std, size=(n_bboxes, num_trials, 3)).astype(np.float32)
    angles = np.random.uniform(rotation_bounds[0], rotation_bounds[1], size=(n_bboxes, num_trials)).astype(np.float32)

    cos_vals, sin_vals = np.cos(angles), np.sin(angles)
    rot_matrices = np.stack([np.stack([cos_vals, sin_vals], axis=-1),
                             np.stack([-sin_vals, cos_vals], axis=-1)], axis=-2)

    rot_matrices = np.transpose(rot_matrices, (1, 2, 3, 0))  # reshape to (n_bbox, num_try, 2, 2)

    box_corners = bbox3d2bevcorners(bboxes)
    point_masks = remove_pts_in_bboxes(pc, bboxes, rm=False)

    new_bboxes, new_points = apply_noise_to_objects(pc, bboxes, box_corners, translations, angles, rot_matrices, point_masks)
    data.update({'gt_bboxes_3d': new_bboxes, 'pts': new_points})
    return data


def maybe_flip_points(data, flip_probability):
    if np.random.rand() < flip_probability:
        pts = data['pts']
        pts[:, 1] = -pts[:, 1]
        data['pts'] = pts
        data['gt_bboxes_3d'][:, 1] = -data['gt_bboxes_3d'][:, 1]
        data['gt_bboxes_3d'][:, 6] = -data['gt_bboxes_3d'][:, 6]
    return data

def data_augment(CLASSES, data_root, data_dict, data_aug_config):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty, image_info, calib_info, image)
    data_aug_config: dict()
    return: data_dict
    '''
    db_sampler_config = data_aug_config['db_sampler']
    data_dict = sample_objects_from_database(CLASSES, data_root, data_dict, db_sampler_config['db_sampler'], db_sampler_config['sample_groups'])
    object_noise_config = data_aug_config['object_noise']
    data_dict = apply_noise_to_objects(data_dict, 
                             num_try=object_noise_config['num_try'],
                             translation_std=object_noise_config['translation_std'],
                             rot_range=object_noise_config['rot_range'])
    random_flip_ratio = data_aug_config['random_flip_ratio']
    data_dict = maybe_flip_points(data_dict, random_flip_ratio)
    global_rot_scale_trans_config = data_aug_config['global_rot_scale_trans']
    rot_range = global_rot_scale_trans_config['rot_range']
    scale_ratio_range = global_rot_scale_trans_config['scale_ratio_range']
    translation_std = global_rot_scale_trans_config['translation_std']
    data_dict = perturb_object_positions(data_dict, rot_range, scale_ratio_range, translation_std)
    data_dict['pts'] = limit_period(data_dict['pts'], period=np.pi)
    return data_dict