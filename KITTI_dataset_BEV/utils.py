import numpy as np
import os
import pickle


def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError
    
def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)

def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of kitti.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.
    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T

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


    

