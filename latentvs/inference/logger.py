import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from geometry import *
from inference.utils import *


class Logger():
    def __init__(self, save_paths, batch_size, num_iters, error_shape, cdsTw, save_extra, save_plots, save_kitti):
        self.num_iters = num_iters
        self.save_paths = save_paths
        self.cdsTw = cdsTw
        self.errors = np.empty((batch_size, num_iters + 1, error_shape))
        self.errors_t = np.empty((batch_size, num_iters + 1))
        self.errors_r = np.empty((batch_size, num_iters + 1))
        self.poses = np.empty((batch_size, num_iters + 1, 6))
        self.cdrc = np.empty((batch_size, num_iters + 1, 6))
        self.velocities = np.empty((batch_size, num_iters, 6))
        self.save_extra = save_extra
        self.save_plots = save_plots
        self.save_kitti = save_kitti
        self.start_t_error, self.start_r_error = None, None
    def on_update_poses(self, i, poses, error_vector):
        euclidean_dist, rot_dist = compute_errors(self.cdsTw, poses)
        wTcs = batch_to_homogeneous_transform_with_axis_angle(poses[:, :3], poses[:, 3:])
        cdsTcs = np.matmul(self.cdsTw, wTcs)
        cdrc = batch_to_pose_vector(cdsTcs)
        self.errors[:, i] = error_vector
        self.errors_t[:, i] = euclidean_dist
        self.errors_r[:, i] = rot_dist
        self.poses[:, i] = poses
        self.cdrc[:, i] = cdrc

        e_t_mm = np.round(np.mean(euclidean_dist, axis=0) * 1000, 3)
        e_r_deg = np.round(np.degrees(np.mean(rot_dist, axis=0)), 3)
        e_z = np.round(np.mean(np.linalg.norm(error_vector, ord=2, axis=-1)), 3)
        
        converged_count = 0
        if self.start_r_error is None or self.start_t_error is None:
            self.start_t_error = euclidean_dist
            self.start_r_error = rot_dist
        else:
            has_converged = Logger._has_converged(self.start_t_error, self.start_r_error, euclidean_dist, rot_dist)
            converged_count = np.sum(has_converged)


        display_str = '{}/{}: {}mm, {}°, ||z-z*||: {}, converged: {}/{}'.format(i, self.num_iters, e_t_mm, e_r_deg, e_z, converged_count, len(self.poses))
        print(display_str, end='\r')
    def new_velocity(self, i, vc):
        self.velocities[:, i] = vc
    def on_end_vs(self, desired_poses, method_name):
        b = len(self.velocities)
        self.start_t_error, self.start_r_error = None, None
        for n, d in zip(['error.txt', 'error_translation.txt', 'error_rotation.txt'], [self.errors, self.errors_t, self.errors_r]):
            self.foreach_zip(self.save_array_compressed, [n] * b, d)
        self.foreach_zip(self.save_array_compressed, ['vc.txt'] * b, self.velocities)
        self.foreach_zip(self.save_array_compressed, ['cdrc.txt'] * b, self.cdrc)
        if self.save_kitti:
            self.foreach_zip(self.save_trajectory_kitti_format, [method_name] * b, self.poses)
        if self.save_plots:
            self.foreach_zip(self.plot_errors, self.errors, self.errors_t, self.errors_r)
            self.foreach_zip(self.plot_trajectory, self.poses, desired_poses)
            self.foreach_zip(self.plot_velocity, self.velocities)
            self.foreach_zip(self.plot_pose_diff, self.cdrc)

    def map_data_and_fn_to_action(self, d):
        methods_dict = {method_name: getattr(self, method_name) for method_name in dir(self)
                  if callable(getattr(self, method_name))}
        for fn_name, data, is_extra in d:
            if is_extra and not self.save_extra:
                continue
            if fn_name not in methods_dict:
                print('Action', fn_name, 'was ignored because it was not found')
            else:
                if isinstance(data, tuple) or isinstance(data, list):
                    self.foreach_zip(methods_dict[fn_name], *data)
                else:
                    self.foreach_zip(methods_dict[fn_name], data)
    def save_array(self, path, name, data):
        np.savetxt(str(path / name), data)
    def save_array_compressed(self, path, name, data):
        np.savez_compressed(str(path / name), data)
    
    def save_image(self, path, name, img):
        path = path / 'images'
        path.mkdir(exist_ok=True)
        cv2.imwrite(str(path / name), img)
    def save_desired_image(self, path, img):
        path = path / 'images'
        path.mkdir(exist_ok=True)
        cv2.imwrite(str(path / 'orig_Id.png'), img)
    def save_starting_image(self, path, img):
        path = path / 'images'
        path.mkdir(exist_ok=True)
        cv2.imwrite(str(path / 'orig_I0.png'), img)
    def save_rebuilt_desired_image(self, path, img):
        self.save_image(path, 'rec_Id.jpg', img)

    def save_desired_reconstruction_error_image(self, path, Id, Idr):
        self.save_image(path, 'error_rec_Id.jpg',  compute_diff(Id, Idr))
        
    def save_current_reconstruction_error_image(self, path, idx, I, Ir):
        self.save_image(path, 'error_rec_I_{:05d}.jpg'.format(idx), compute_diff(I, Ir))
        
    def save_rebuilt_error_desired_current_image(self, path, idx, Ir, Idr):
        self.save_image(path, 'rec_Idiff_{:05d}.jpg'.format(idx), compute_diff(Ir, Idr))
    def save_error_desired_current_image(self, path, idx, I, Id):
        self.save_image(path, 'orig_Idiff_{:05d}.jpg'.format(idx), compute_diff(I, Id))
    def save_current_image(self, path, i, img):
        self.save_image(path, 'orig_{:05d}.png'.format(i), img)
    def save_rebuilt_current_image(self, path, i, img):
        self.save_image(path, 'rec_{:05d}.jpg'.format(i), img)
    def save_rebuilt_idiff(self, path, i, img):
        self.save_image(path, 'rec_{:05d}.jpg'.format(i), img)
        
    def foreach_zip(self, fn, *args):
        for d in zip(self.save_paths, *args):
            fn(*d)
    def save_starting_pose(self, path, pose):
        np.savetxt(str(path / 'starting_pose.txt'), pose)
    def save_desired_pose(self, path, pose):
        np.savetxt(str(path / 'desired_pose.txt'), pose)
    
    def save_initial_errors(self, save_dir_sample, error_t, error_r):
        np.savetxt(str(save_dir_sample / 'initial_error_t.txt'), [error_t])
        np.savetxt(str(save_dir_sample / 'initial_error_r.txt'), [np.degrees(error_r)])
    def plot_errors(self, save_dir_sample, error_w, error_t, error_r):
        error_r = np.degrees(error_r)
        for error, name in zip([error_w, error_t, error_r, np.linalg.norm(error_w, axis=-1)], ['error_w.png', 'error_t.png', 'error_r.png', 'error_w_norm.png']):
            plt.plot(error)
            plt.xlabel('Iterations')
            plt.ylabel('Error')
            plt.grid()
            # plt.show()
            plt.savefig(str(save_dir_sample / name))
            plt.close()
            plt.clf()
        
    def plot_velocity(self, save_dir_sample, velocities):
        plt.plot(velocities)
        plt.grid()
        plt.legend([
            r'$\mathbf{\upsilon}_x$',
            r'$\mathbf{\upsilon}_y$',
            r'$\mathbf{\upsilon}_z$',
            r'$\mathbf{\omega}_x$',
            r'$\mathbf{\omega}_y$',
            r'$\mathbf{\omega}_z$',
        ])
        plt.savefig(str(save_dir_sample / 'velocity.png'))
        plt.close()
        plt.clf()
    def plot_pose_diff(self, save_dir_sample, cdrcs):
        plt.plot(cdrcs)
        plt.grid()
        plt.legend([r'$\Delta \mathbf{t}_x$', r'$\Delta \mathbf{t}_y$', r'$\Delta \mathbf{t}_z$', r'$\Delta\mathbf{\theta}_x$', r'$\Delta\mathbf{\theta}_y$', r'$\Delta\mathbf{\theta}_z$'])
        plt.savefig(str(save_dir_sample / 'cdrc.png'))
        plt.close()
        plt.clf()
    def plot_trajectory(self, path, poses, desired_pose):
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        Rs = batch_axis_angle_to_rotation_matrix(poses[:, 3:])
        Rd = batch_axis_angle_to_rotation_matrix(np.array([desired_pose[3:]]))[0]
        rotation_axes_scale = 0.05
        autoscale = True
        plot_frames = False
        mins = np.min(poses[:, :3], axis=0)
        maxs = np.max(poses[:, :3], axis=0)
        if autoscale:
            ax.set_xlim([mins[0], maxs[0]])
            ax.set_ylim([mins[1], maxs[1]])
            ax.set_zlim([mins[2], maxs[2]])
        scales_rots = [(r[1] - r[0]) * rotation_axes_scale for r in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]]
        if plot_frames:
            def plot_frame(position, R):
                colors = [(0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
                for i, c, s in zip(range(3), colors, scales_rots):
                    ax.quiver(position[0], position[1], position[2], R[0, i], R[1, i], R[2, i], arrow_length_ratio=0.0, colors=c, length=s)
            
            for i in range(len(poses) // 100):
                p = poses[i * 100, :3]
                #plot_frame(p, Rs[i * 100])
            # plot_frame(poses[-1, :3], Rs[-1])
            # plot_frame(desired_pose[:3], Rd)
        
        ax.plot(poses[:, 0], poses[:, 1], poses[:, 2])

        ax.scatter(desired_pose[0], desired_pose[1], desired_pose[2], c='r', s=10)
        ax.scatter(poses[0, 0], poses[0, 1], poses[0, 2], s=10)
        ax.scatter(poses[-1, 0], poses[-1, 1], poses[-1, 2], c='g', s=10)
        # ax.set_xlim3d(-generator_parameters['half_length_m'], generator_parameters['half_length_m'])
        # ax.set_ylim3d(-generator_parameters['half_width_m'], generator_parameters['half_width_m'])
        plt.savefig(str(path / 'trajectory.png'))
        plt.close()
    def save_trajectory_kitti_format(self, path, name, poses):
        wTcs = batch_to_homogeneous_transform_with_axis_angle(poses[:, :3], poses[:, 3:])
        wTcs = wTcs[:, :3]
        wTcs = np.reshape(wTcs, (wTcs.shape[0], -1))
        save_file_path = Path(path / '{}.kitti'.format(name))
        np.savetxt(str(save_file_path), wTcs)
    def make_video(self, path, expr, video_name, text, h, w):
        import ffmpeg
        (
        ffmpeg
        .input(str(path / expr), framerate=25)
        .drawtext(text=text, fontsize=24, y=h - (h // 10), x=w//2, fontcolor='white')
        .output(str(path / video_name))
        .overwrite_output()
        .global_args('-loglevel', 'quiet')
        .run()
        )
    @staticmethod
    def save_dict(path, d):
        with open(str(path), 'w') as f:
            yaml.dump(d, f)
    @staticmethod
    def _has_converged(set, ser, eet, eer):
        ct1 = eet / np.maximum(set, 0.0001) < 0.1
        ct2 = eet < 0.01
        cr1 = eer / np.maximum(ser, 0.0001) < 0.1
        cr2 = eer < 1.0
        ct = np.logical_or(ct1, ct2)
        cr = np.logical_or(cr1, cr2)
        c = np.logical_and(ct, cr)
        return c

