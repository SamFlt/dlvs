import numpy as np
import cv2
from geometry import *
import torch
def compute_diff(I1, I2):
    x = I1.astype(np.int) - I2.astype(np.int)
    x = x / 2
    x += 127.5
    return x.astype(np.uint8)
def compute_errors(cdsTw, current_poses):
    wTcs = batch_to_homogeneous_transform_with_axis_angle(current_poses[:, :3], current_poses[:, 3:])
    cdsTcs = np.matmul(cdsTw, wTcs)
    ts = cdsTcs[:, :3, 3]
    euclidean_dist = np.linalg.norm(ts, axis=-1)
    Rs = cdsTcs[:, :3, :3]
    tus = batch_rotation_matrix_to_axis_angle(Rs)
    thetas = np.linalg.norm(tus, axis=-1)
    return euclidean_dist, thetas
def to_gray(images):
    dims = images.size() if isinstance(images, torch.Tensor) else images.shape
    if len(dims) == 4:
        res = np.empty((len(images), images.shape[1], images.shape[2]))
        for i, image in enumerate(images):
            res[i] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return res
    else:
        return images

class VSArguments():
    def __init__(self, gain, num_iters, batch_size, save_path, h, w):
        self.gain = gain
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.save_path = save_path
        self.h = h
        self.w = w
    def as_dict(self):
        return self.__dict__