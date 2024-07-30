from pathlib import Path
import torch
import numpy as np
from geometry import *
from utils.custom_typing import *
def compute_scale(dl, batch_size, stats_mode='avg', stats_pairs=False):
    """Compute the scalings of the translation and rotation from a dataset

    Args:
        dl (torch.utils.data.DataLoader): the dataloader representing the dataset on which to compute scale
        batch_size (int): the batch size for processing
        stats_mode (str: avg|max|custom, optional): How to compute the scale. one of:
            avg: compute the scale as the average translation/rotation distances 
                    between the samples of the batches (mean per batch, mean of batches) 
            max: The scales are the maximum distances seen in the batches
            custom: hard coded values 
            Defaults to 'avg'.
        stats_pairs (bool, optional): Compute the scales on pairs, not on full batches. Defaults to False.

    Returns:
        (float, float, float): (translation scale, rotation scale, clipping parameter (to be used after combining the distance matrices))
    """    
    print('Computing stats with mode {}'.format(stats_mode))
    sum_t = 0.0
    sum_r = 0.0
    t_max = 0.0
    r_max = 0.0
    
    dists_per_batch = (batch_size * 2) ** 2 - batch_size * 2
    if stats_pairs:
        dists_per_batch = batch_size
        print('Using stats on pairs!')
    for i, b in enumerate(iter(dl)):
        t_dist_matrix, r_dist_matrix = get_se3_dist_matrices(b[1].cpu().numpy())

        t_dist_matrix = torch.from_numpy(t_dist_matrix)
        r_dist_matrix = torch.from_numpy(r_dist_matrix)
        
        ts, rs = t_dist_matrix, r_dist_matrix

        if stats_pairs:
            t_dists_c_d = torch.tensor([t_dist_matrix[i, i + batch_size] for i in range(batch_size)], device=t_dist_matrix.device)
            r_dists_c_d = torch.tensor([r_dist_matrix[i, i + batch_size] for i in range(batch_size)], device=r_dist_matrix.device)
            ts, rs = t_dists_c_d, r_dists_c_d
            # print(torch.max(b['t_dist_matrix']).item(), torch.max(b['r_dist_matrix']).item())

        t_max = max(torch.max(ts).item(), t_max)
        r_max = max(torch.max(rs).item(), r_max)
        sum_t += torch.sum(ts).item()
        sum_r += torch.sum(rs).item()
    if stats_mode == 'avg':
        t_scale = sum_t / (len(dl) * dists_per_batch) # number of elements in distance matrix minus the diagonal
        r_scale = sum_r / (len(dl) * dists_per_batch)
        # t_scale = sum_t / (len(dl) * batch_size)
        # r_scale = sum_r / (len(dl) * batch_size)
        scale_clip = 2.0
        
    elif stats_mode == 'max':
        t_scale = t_max
        r_scale = r_max
        scale_clip = 2.0
    elif stats_mode == 'custom':
        t_scale = 0.01
        r_scale = np.radians(1.0)
        scale_clip = 50.0
    return t_scale, r_scale, scale_clip

def get_se3_dist_matrices(poses):
    """Compute the distance matrices (translation/rotation) between poses

    Args:
        poses (np.ndarray): an Nx6 matrix where each 6d vector is a pose

    Returns:
        (np.ndarray, np.ndarray): Translation and rotation distance matrices. These are NxN arrays, with D_i,i = 0 and D_i,j = D_j,i
    """    
    bs = len(poses)
    t_dist_matrix = np.empty((bs, bs), dtype=np.float32)
    r_dist_matrix = np.empty((bs, bs), dtype=np.float32)
    wTcs = batch_to_homogeneous_transform_with_axis_angle(poses[:, :3], poses[:, 3:])
    for i in range(bs):
        ciTw = batch_homogeneous_inverse(wTcs[i:i+1])
        ciTcs = np.matmul(ciTw, wTcs) 
        ts = ciTcs[:, :3, 3]
        rs = batch_rotation_matrix_to_axis_angle(ciTcs[:, :3, :3])
        t_dist_matrix[i] = np.linalg.norm(ts, axis=-1)
        r_dist_matrix[i] = np.linalg.norm(rs, axis=-1)
    np.fill_diagonal(r_dist_matrix, 0.0)
    return t_dist_matrix, r_dist_matrix


def compute_reprojection_error(points, p1, p2=None):
    """Compute the reprojection error between two sets of camera poses. If p2 is None, then compute error between every pose in p1

    Args:
        points (_type_): Ux3 array of points, in the scene
        p1 (_type_): Nx6 matrix representing a set of N poses. p1 = wTc, and are thus inverted before computation
        p2 (_type_): Mx6 matrix representing a set of M poses also inverted before computation, can be None

    Returns:
        if p2 is not None:
            An NxM matrix where each element i,j represents the average reprojection error of points p between cameras p1[i] and p2[j]
        If p2 is none the matrix is NxN
    """
    N, U = len(p1), len(points)

    h = np.ones((len(points), 1))
    hp = np.concatenate((points, h), axis=-1)
    hp = np.transpose(hp, (1, 0)) # 4 x K
    def get_2d(p):
        wTp = batch_to_homogeneous_transform_with_axis_angle(p[:, :3], p[:, 3:])
        T = batch_homogeneous_inverse(wTp)
        hpx = np.matmul(T, hp)
        px = hpx[:, :3] / hpx[:, 3:]
        K = np.eye(3).reshape(1, 3, 3).repeat(len(T), axis=0)
        uvw = np.matmul(K, px)
        uv = uvw[:, :2] / uvw[:, 2:]
        return uv
    uv1 = get_2d(p1)
    assert uv1.shape == (N, 2, U)
    if p2 is not None:
        M = len(p2)
        uv2 = get_2d(p2)
        assert uv2.shape == (M, 2, U)
        
        res = np.empty((N, M))

        for i in range(N):
            uv1i = uv1[i:i+1]
            err = uv2 - uv1i
            err = np.linalg.norm(err, ord=2, axis=1) # Euclidean distance between pixel reprojections
            err = np.mean(err, axis=-1)
            res[i] = err
        return res
    else:
        res = np.empty((N, N))
        for i in range(N):
            uv1i = uv1[i:i+1]
            err = uv1 - uv1i
            err = np.linalg.norm(err, ord=2, axis=1) # Euclidean distance between pixel reprojections
            err = np.mean(err, axis=-1)
            res[i] = err
        return res


def get_se3_dist_matrices_compare(p1, p2):
    """Compute the distance matrices (translation/rotation) between poses p1 and p2

    Args:
        p1 (np.ndarray): an Nx6 matrix where each 6d vector is a pose
        p2 (np.ndarray): an Mx6 matrix where each 6d vector is a pose
        

    Returns:
        (np.ndarray, np.ndarray): Translation and rotation distance matrices. These are NxM arrays, with D_i,i = 0 and D_i,j = D_j,i
    """    
    N = len(p1)
    M = len(p2)
    t_dist_matrix = np.empty((N, M), dtype=np.float32)
    r_dist_matrix = np.empty((N, M), dtype=np.float32)
    wTp1s = batch_to_homogeneous_transform_with_axis_angle(p1[:, :3], p1[:, 3:])
    wTp2s = batch_to_homogeneous_transform_with_axis_angle(p2[:, :3], p2[:, 3:])
    p1sTw = batch_homogeneous_inverse(wTp1s)
    for i in range(N):
        p1iTw = p1sTw[i:i+1]
        p1iTp2s = np.matmul(p1iTw, wTp2s)
        ts = p1iTp2s[:, :3, 3]
        rs = batch_rotation_matrix_to_axis_angle(p1iTp2s[:, :3, :3])
        t_dist_matrix[i] = np.linalg.norm(ts, axis=-1)
        r_dist_matrix[i] = np.linalg.norm(rs, axis=-1)
    return t_dist_matrix, r_dist_matrix

def pose_interaction_matrix(poses):
    '''
    Compute the interaction matrices associated to the given poses
    poses: an Nx6 numpy array: the poses with the first 3 components the translation (in m) 
    and the last 3 components the rotation in the axis/angle representation (in rad)
    '''
    def screw_symmetric_matrices(v):
        S = np.zeros((len(v), 3, 3))
        S[:, 0, 1] = -v[:, 2]
        S[:, 0, 2] = v[:, 1]
        S[:, 1, 0] = v[:, 2]
        S[:, 1, 2] = -v[:, 0]
        S[:, 2, 0] = -v[:, 1]
        S[:, 2, 1] = v[:, 0]
        return S
    L = np.zeros((len(poses), 6, 6))
    L[:, :3, :3] = -np.eye(3)
    L[:, 0, 4] = -poses[:, 2]
    L[:, :3, 3:] = screw_symmetric_matrices(poses[:, :3])

    thetas = np.linalg.norm(poses[:, 3:])
    skew_u = screw_symmetric_matrices(poses[:, 3:] / np.expand_dims(thetas, -1))
    z = (1 - (np.sinc(thetas) / np.sinc(thetas / 2.0) ** 2))
    Ltheta = np.expand_dims(np.eye(3), 0) - (thetas / 2.0) * skew_u + z * (skew_u @ skew_u)
    L[:, 3:, 3:] = Ltheta
    return L


def load_image_net_images_paths(root, take_start_index=0, max_take_count=10, num_classes=-1, class_list=[]):
    """Get the paths for a subset of imagenet images

    Args:
        root (pathlib.Path): the root of the imagenet folder. this folder should contain one subfolder per class
        take_start_index (int, optional): Start collecting image paths from this index.
                                        The scene images are alphabetically sorted Defaults to 0.
        max_take_count (int, optional): Maximum number of scenes to take, if less are available, then all will be taken. Defaults to 10.
        If max_take_count < 0, then all images after take_start_index will be taken
        num_classes (int, optional): Number of classes to take. Defaults to -1.
        class_list (list[str], optional): The classes to take. Defaults to [].

    Returns:
        list[str]: the absolute paths of the images to load to make a multiscene dataset.
    """    
    paths = []
    cls_index = 0
    for cls_folder in sorted(root.iterdir()):
        if len(class_list) > 0 and cls_folder.name not in class_list:
            continue
        images = sorted(cls_folder.iterdir())
        images = [image for image in images if '.jpg' in image.name]
        if cls_index == num_classes:
            break
        mx = max_take_count if max_take_count > 0 else len(images) - take_start_index

        for i in range(take_start_index, take_start_index + mx):
            if i < len(images):
                paths.append(str(images[i].absolute()))
            else:
                break
        cls_index += 1
    return paths

def load_image_woof_paths(root: Path, set: str) -> List[str]:
    classes_folder = root / set
    paths = []
    for class_folder in classes_folder.iterdir():
        if not class_folder.is_dir():
            continue
        for image_path in class_folder.iterdir():
            paths.append(str(image_path.absolute()))
    return paths



def add_optical_axis_rotations(Rs, rz_max):
    """Generate new rotation matrices, with a rotation added around the optical axis
    The added rotation is in [-rz_max, rz_max], with rz_max in radians

    Args:
        Rs (np.ndarray): The input rotation matrices, N x 3 x 3 array
        rz_max (float): max rotation around the optical axis, used for uniform sampling

    Returns:
        np.ndarray: The output rotation matrices, with the rotation around the optical axis added
    """
    z_axis = Rs[:, :, 2]
    random_z = np.random.uniform(-rz_max, rz_max, size=(len(Rs), 1))
    # theta_u_vectors = np.concatenate((np.zeros((self.num_samples, 2)), random_z), axis=-1)
    theta_u_vectors = z_axis * random_z
    Rz = batch_axis_angle_to_rotation_matrix(theta_u_vectors)
    return Rz

def generate_look_at_positions(num_samples, half_ranges):
    """Generate points on the scene poster.
    This makes the hypothesis that the scene is located on the plane with z = 0 with normal = [0, 0, 1]
    The center of the scene is [0,0,0]
    The points will be sampled in a rectangle around the center, with width half_range[0] *2 and height half_ranges[1] * 2
    Args:
        num_samples (unsigned): Number of points to generate
        half_ranges ((float, float)): the dimensions of the rectangle in which to sample the points

    Returns:
        np.ndarray: a num_samples X 3 array, the positions of the points
    """
    zs = np.zeros((num_samples, 1))
    xs = np.random.uniform(-half_ranges[0], half_ranges[0], size=zs.shape)
    ys = np.random.uniform(-half_ranges[1], half_ranges[1], size=zs.shape)
    return np.concatenate((xs, ys, zs), axis=-1)

def generate_look_from_positions(num_samples, half_ranges, z_center):
    """Generate points above the scene poster.
    Their distance from the scene is centered around z_center.
    They are sampled uniformly from the half_ranges

    Args:
        num_samples (float): number of positions to generate
        half_ranges ((float, float, float)): Ranges from which to generate the points
        z_center (float): Height of the central 3d rectangle.

    Returns:
        np.ndarray: num_samples X 3, the points
    """    
    ps = [np.random.uniform(-hr, hr, size=(num_samples, 1)) for hr in half_ranges]
    ps = np.concatenate(ps, axis=-1)
    ps[:, 2] += z_center
    return ps

def compute_rotation_matrix_look_at(look_at_points, look_from_points, random_vec=np.array([0.0, -1.0, 0.0])):
    """Compute the rotation matrices so that the cameras positioned at look_from_points are directly looking at the look_at_points.

    Args:
        look_at_points (np.ndarray): an Nx3 array, the points to look at. They are situated on the scene poster.
        look_from_points (np.ndarray): an Nx3 array, the positions of the camera
        random_vec (np.ndarray, optional): a 3 array, describing the world "up" vector, which is used to deduce the "up" direction of the camera image plane. Defaults to np.array([0.0, 1.0, 0.0]).

    Returns:
        np.ndarray: An Nx3x3 array, containing the rotation matrices
    """
    random_vec /= np.linalg.norm(random_vec, keepdims=True)

    fwd = look_at_points - look_from_points
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)

    right = np.cross(fwd, random_vec)
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(fwd, right)
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    Rs = np.concatenate([np.expand_dims(d, 1) for d in [right, up, fwd]], axis=1)
    return Rs

def generate_dir_vector_in_cone(base_vectors, cone_angle_deg):
    '''
    Generate unit vectors that lie in approximately the same directions as the base_vectors.
    The maximum accepted angle between generated and base vectors is set by cone_angle_deg
    '''
    # See: https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region/39003745
    # Note: If one of the base vectors is [0.0, 0.0, 1.0], there is a degenerate case => output  will be NaN TODO: fix this 
    N = len(base_vectors)
    angle = np.radians(cone_angle_deg)
    z = np.random.uniform(size=(N, 1)) * (1 - np.cos(angle)) + np.cos(angle)
    phi = np.random.uniform(size=(N, 1)) * 2 * np.pi
    x = np.sqrt(1 - z ** 2) * np.cos(phi)
    y = np.sqrt(1 - z ** 2) * np.sin(phi)
    xyz = np.concatenate((x, y, z), axis=-1) # points in a cone, but this cone is not centered around the base_vectors (it has direction [0, 0, 1])
    assert xyz.shape == (N, 3)
    unit_base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=-1, keepdims=True)
    north = np.array([0, 0, 1.0])
    u = np.cross(north, unit_base_vectors)
    assert u.shape == (N, 3)

    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    theta = np.arccos(np.dot(unit_base_vectors, north))
    assert theta.shape == (N,)
    thetau = u * theta[:, None]
    R = batch_axis_angle_to_rotation_matrix(thetau) # Rotation matrices, going from north orientation to the orientation of each vector
    xyz = np.matmul(R, xyz[:, :, None])[:, :, 0]
    xyz = np.nan_to_num(xyz, copy=False, nan=0.0) # Replace invalid values for 0 length vectors
    return xyz

def generate_dir_vector_in_cone_batch(base_vectors, count, cone_angle_deg):
    '''
    Generate unit vectors that lie in approximately the same directions as the base_vectors.
    The maximum accepted angle between generated and base vectors is set by cone_angle_deg
    base_vectors: N x 3 nd array
    count: number of random vectors generated around the base vector
    '''
    # See: https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region/39003745
    # Note: If one of the base vectors is [0.0, 0.0, 1.0], there is a degenerate case => output  will be NaN TODO: fix this 
    N = len(base_vectors)
    angle = np.radians(cone_angle_deg)
    z = np.random.uniform(low=np.cos(angle), high=1.0, size=(N, count, 1))
    phi = np.random.uniform(low=0.0, high=2 * np.pi, size=(N, count, 1))
    return generate_vector_with_base_ref(base_vectors, z, phi)

def generate_vector_with_base_ref(base_vectors, z, phi):
    N, count = z.shape[:2]
    x = np.sqrt(1 - z ** 2) * np.cos(phi)
    y = np.sqrt(1 - z ** 2) * np.sin(phi)
    xyz = np.concatenate((x, y, z), axis=-1) # points in a cone, but this cone is not centered around the base_vectors (it has direction [0, 0, 1])
    assert xyz.shape == (N, count, 3)
    unit_base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=-1, keepdims=True)
    north = np.array([0, 0, 1.0])
    u = np.cross(north, unit_base_vectors)
    assert u.shape == (N, 3)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    theta = np.arccos(np.dot(unit_base_vectors, north))
    assert theta.shape == (N,)
    thetau = u * theta[:, None]
    R = batch_axis_angle_to_rotation_matrix(thetau) # Rotation matrices, going from north orientation to the orientation of each vector
    xyz = np.matmul(R[:, None], xyz[:, :, :, None])[:,:,:, 0]
    xyz = np.nan_to_num(xyz, copy=False, nan=0.0) # Replace invalid values for 0 length vectors
    assert xyz.shape == (N, count, 3)
    return xyz

def generate_dir_vector_outside_cone_batch(base_vectors, count, cone_angle_deg):

    # See: https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region/39003745
    # Note: If one of the base vectors is [0.0, 0.0, 1.0], there is a degenerate case => output  will be NaN TODO: fix this 
    N = len(base_vectors)
    angle = np.radians(cone_angle_deg)
    z = np.random.uniform(low=-1, high=np.cos(angle), size=(N, count, 1))
    phi = np.random.uniform(low=0.0, high=2 * np.pi, size=(N,  count, 1))
    return generate_vector_with_base_ref(base_vectors, z, phi)



def gaussian_blur_kernel(ks, sigma):
    constant = 1 / (2 * np.pi * (sigma ** 2))
    kernel = np.empty((ks, ks))
    mid = ks // 2 if ks % 2 == 1 else ks // 2 + 0.5
    for x in range(ks):
        x_dist = x - mid
        for y in range(ks):
            y_dist = y - mid
            kernel[x, y] = constant * np.exp(-((x_dist ** 2 + y_dist ** 2) / (2 * (sigma ** 2))))

    return kernel / np.sum(kernel)