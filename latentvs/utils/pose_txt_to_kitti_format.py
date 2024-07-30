'''Script that converts visp poses (in .txt format) to kitti format'''

import numpy as np
import sys
import os
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from geometry import *
def save_trajectory_kitti_format(path, name, poses):
    wTcs = batch_to_homogeneous_transform_with_axis_angle(poses[:, :3], poses[:, 3:])
    wTcs = wTcs[:, :3]
    wTcs = np.reshape(wTcs, (wTcs.shape[0], -1))
    save_file_path = Path(path / '{}.kitti'.format(name))
    np.savetxt(str(save_file_path), wTcs)

if __name__ == '__main__':
    path = sys.argv[1]
    array = np.loadtxt(path, delimiter='\t', skiprows=1, usecols=(1, 4, 7, 10, 13, 16))
    p = Path(path)
    parent = p.parent
    n = p.name[:4]
    print(array)
    save_trajectory_kitti_format(parent, n, array)

