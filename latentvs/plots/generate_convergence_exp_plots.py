from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_translation_traj(file_path):
    """Read the translation part of a trajectory that is stored in a ViSP text

    Args:
        file_path (str): The file path pointing to the trajectory file

    Returns:
        [np.ndarray]: an Nx3 array containing the position along the trajectory 
    """
    array = np.loadtxt(file_path, delimiter='\t', skiprows=1, usecols=(1, 4, 7, 10, 13, 16))
    return array[:, :3]

def read_sample_dir(dir):
    """Read a VS result directory, returning the main results

    Args:
        dir (pathlib.Path): The directory path

    Returns:
        [(float, float, np.ndarray)]: (end translation error, end rotation error, 3D trajectory)
    """    
    with open(str(dir / 'end_translation_error.txt'), 'r') as f:
        et = float(f.readline())
    with open(str(dir / 'end_rotation_error.txt'), 'r') as f:
        er = float(f.readline())
    traj = read_translation_traj(str(dir / 'crw.txt'))
    return et, er, traj
if __name__ == '__main__':
    """ Generate the matplotlib plots related to the convergence experiment on the robot.
    Args:
        root
        name_start 
    The results directory should look like:
        - root/
                name_start + displacements[0]
                name_start + displacements[1]
                name_start + displacements[2]
                ...
        All the folders starting by name_start in the root directory will be used to get the results  
    """
    plt.rc('text', usetex=True) 
    root_folder = Path(sys.argv[1])
    name_start = sys.argv[2]
    displacements = [0.02, 0.05, 0.1, 0.2, 0.4]
    desired_pose_t = None
    convergence_values = []
    convergence_sets = []
    traj_sets = []
    directory_sets = []
    
    for displacement in displacements:
        ets, ers = [], []
        trajs_displacement = []
        directories = []
        results_folder = root_folder / (name_start + str(displacement))
        print(results_folder)
        for sample_dir in sorted(results_folder.iterdir()):
            if not sample_dir.is_dir():
                continue
            et, er, traj = read_sample_dir(sample_dir)
            ets.append(et)
            ers.append(er)
            trajs_displacement.append(traj)
            directories.append(sample_dir.name)
            if desired_pose_t is None:
                desired_pose = np.loadtxt(str(sample_dir / 'target_pose.txt'))
                desired_pose_t = desired_pose[:3]
        ets = np.array(ets)
        ers = np.array(ers)
        conved = np.logical_and(ets < 10.0, ers < 1.0)
        convergence_sets.append(conved)
        directory_sets.append(directories)
        conv_rate = np.sum(conved) / len(ets)
        convergence_values.append(conv_rate * 100)
        traj_sets.append(trajs_displacement)

    for displacement, conv_set, dir_set in zip(displacements, convergence_sets, directory_sets):
        print('Displacement: {}'.format(displacement))
        print('Did not converge:')
        for c, di in zip(conv_set, dir_set):
            if not c:
                print('\t: {}'.format(di))

    colors = ['tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:red']

    plt.figure()
    ax = plt.gca()
    plt.plot(range(1, len(convergence_values) + 1), convergence_values, color='k', zorder=1)
    for i, conv_rate, color in zip(range(1, len(displacements) + 1), convergence_values, colors):
        plt.scatter(i, conv_rate, c=color, zorder=2)
    plt.grid()
    plt.xticks(range(1, len(convergence_values) + 1))
    plt.ylim([0, 101])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.ylabel('Convergence rate, \\%', fontsize=14)
    plt.xlabel('Batch index', fontsize=14)
    
    plt.tight_layout()
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    ax.set_zlabel('Z')
    
    for traj_set, conv_set, color in zip(traj_sets, convergence_sets, colors):
        for traj, conved in zip(traj_set, conv_set):
            traj -= desired_pose_t
            if conved:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color, linewidth=1)
            else:
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c=color)
    plt.show()