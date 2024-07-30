from operator import itemgetter
import matplotlib

from inference.augmentors import Augmentor
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from pathlib import Path
import torch
import sys
import numpy as np
import cv2
import yaml

from inference.pose_generators import PoseGeneratorResults

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from inference.methods import *
from inference.interaction_matrix_mixer import *
from generator import MultiGeneratorHandler
from inference.optimizers import *
from geometry import *
from inference.logger import Logger
import yaml

from aevs.inference.methods import NNIMVS
from traditional.inference.methods import *
from metric_learning.inference.methods import *

from utils.custom_typing import *

IndexArray = NDArray[(Any,), UInt]

class InferenceRun():
    def __init__(self, allow_overwrite: bool, vs_method: VSMethod, vs_arguments: VSArguments,
                generator: GeneratorParameters, samples_data: PoseGeneratorResults,
                Id_augmentor: Augmentor, save_extra: bool, save_plots: bool, save_kitti: bool, early_stopping: bool):
        self.vs_method = vs_method
        self.vs_arguments = vs_arguments
        self.vs_arguments.save_path.mkdir(exist_ok=True)

        self.save_extra = save_extra
        self.save_plots = save_plots
        self.generator = generator

        self.starting_poses = samples_data.starting_poses
        self.desired_poses = samples_data.desired_poses
        self.num_samples = len(self.starting_poses)
        self.I = samples_data.starting_images
        self.Id = samples_data.desired_images
        self.initial_overlaps = samples_data.overlap
        self.points_look_at = samples_data.looked_at_points
        self.points_look_at_desired = samples_data.looked_at_desired
        self.scene_indices = np.array(samples_data.scene_indices) if samples_data.scene_indices is not None else np.zeros(len(self.starting_poses))
        self.scene_indices = self.scene_indices.astype(np.int)
        self.Id_augmentor = Id_augmentor
        self.early_stopping = early_stopping
        self.save_kitti = save_kitti

        self.save_dirs = [vs_arguments.save_path / str(i) for i in range(self.num_samples)]
        for path in self.save_dirs:
            path.mkdir(exist_ok=allow_overwrite)
        assert self.num_samples % self.vs_arguments.batch_size == 0


    def run_iter(self, current_poses: PoseArray, _desired_poses: PoseArray,
                desired_image_processed: GrayImageTorchArray, current_images_processed: GrayImageTorchArray,
                scene_indices: IndexArray, iteration: UInt,
                logger: Logger, run_indices: Optional[IndexArray] = None) -> Tuple[PoseArray, RawRGBImageArray, VelocityArray, VSErrorArray]:
        vcs, error, to_save = self.vs_method.compute_vc(current_images_processed, desired_image_processed, iteration, run_indices, current_poses)
        vcs = vcs.cpu().numpy()
        h, w = self.vs_arguments.h, self.vs_arguments.w
        logger.map_data_and_fn_to_action(to_save)
        new_data = self.generator.move_from_poses(len(vcs), [self.vs_arguments.gain for _ in range(len(vcs))], h, w, current_poses, vcs, scene_indices)
        new_poses = np.array([s.pose_vector() for s in new_data])
        new_images = np.array([s.image() for s in new_data])
        return new_poses, new_images, vcs, error

    def run_iter_early_stopping(self, current_poses: PoseArray, desired_poses: PoseArray,
                desired_image_processed: GrayImageTorchArray, current_images_processed: GrayImageTorchArray,
                scene_indices: IndexArray, iteration: UInt,
                logger: Logger) -> Tuple[PoseArray, RawRGBImageArray, VelocityArray, VSErrorArray]:
        vcs = logger.velocities[:, :iteration]
        should_stop = self._should_stop_vs_array(vcs)
        to_run_is = np.argwhere(np.logical_not(should_stop))

        to_run_is = np.squeeze(to_run_is, axis=-1)
        new_poses = current_poses.copy()
        images_size = current_images_processed.size()
        image_size = images_size[2:] if len(images_size) > 3 else images_size[1:]
        new_images = np.zeros((len(current_poses), *image_size, 3), dtype=np.uint8)
        new_vcs = np.zeros((len(current_poses), 6))
        new_error = logger.errors[:, iteration].copy()
        if len(to_run_is) > 0:
            new_poses_r, new_images_r, vcs_r, error_r = self.run_iter(current_poses[to_run_is], desired_poses[to_run_is], desired_image_processed[to_run_is],
                                                                     current_images_processed[to_run_is], scene_indices, iteration, logger, to_run_is)
            new_poses[to_run_is] = new_poses_r
            new_images[to_run_is] = new_images_r
            new_vcs[to_run_is] = vcs_r
            new_error[to_run_is] = error_r



        return new_poses, new_images, new_vcs, new_error

    def _should_stop_vs_array(self, velocities: VelocityTrajectoryArray) -> NDArray[(Any), bool]:
        # B x T x 6
        window_size = 100
        min_iters = 300
        if velocities.shape[1] < min_iters: # too early to even check whether to stop
            return np.zeros(velocities.shape[0], dtype=bool)
        last_velocities = velocities[:, -window_size:]
        last_vts = last_velocities[:, :, :3]
        last_vrs = last_velocities[:, :, 3:]
        vt_mm = np.linalg.norm(last_vts, axis=-1) * 1000.0
        vr_deg = np.degrees(np.linalg.norm(last_vrs, axis=-1))
        vt_conved = np.all(vt_mm < 0.001, axis=-1)
        vr_conved = np.all(vr_deg < 0.001, axis=-1)
        conved = np.logical_and(vt_conved, vr_conved)
        return conved



    def run_batch(self, batch_idx: int) -> None:
        iter_fn = self.run_iter
        if self.early_stopping:
            iter_fn = self.run_iter_early_stopping
        print('\n {}: batch {}'.format(self.vs_method.name(), (batch_idx + 1)))
        num_iters = self.vs_arguments.num_iters
        batch_size = self.vs_arguments.batch_size
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        sample_dirs = self.save_dirs[start_idx:end_idx]
        starting_poses = self.starting_poses[start_idx:end_idx]
        I = self.I[start_idx:end_idx]
        Id = self.Id[start_idx:end_idx]
        Id = self.Id_augmentor(Id)
        initial_overlaps = self.initial_overlaps[start_idx:end_idx]
        scene_indices = self.scene_indices[start_idx:end_idx].tolist()

        desired_poses = self.desired_poses[start_idx:end_idx]
        wTcds = batch_to_homogeneous_transform_with_axis_angle(desired_poses[:, :3], desired_poses[:, 3:])
        cdsTw = batch_homogeneous_inverse(wTcds)
        logger = Logger(sample_dirs, batch_size, num_iters, self.vs_method.error_shape(), cdsTw, self.save_extra, self.save_plots, self.save_kitti)

        self._initial_logging(logger, batch_size, Id, I, desired_poses,
                            starting_poses, initial_overlaps, scene_indices,
                            self.points_look_at[start_idx: end_idx] if  self.points_look_at is not None else None,
                            self.points_look_at_desired[start_idx: end_idx] if self.points_look_at_desired is not None else None)


        Id_processed = self.vs_method.process_image(Id)
        poses = starting_poses
        to_save = self.vs_method.on_new_batch(batch_idx, Id_processed, desired_poses)
        logger.map_data_and_fn_to_action(to_save)
        for i in range(num_iters):
            self.vs_method.on_iter_begin(i)
            I_processed = self.vs_method.process_image(I)
            index_for_zip = [i for _ in range(batch_size)]
            # new_poses, new_images, vc, error_vector = self.run_iter(poses, desired_poses, Id_processed, I_processed, i, logger)
            new_poses, new_images, vc, error_vector = iter_fn(poses, desired_poses, Id_processed, I_processed, scene_indices, i, logger)
            logger.on_update_poses(i, poses, error_vector)
            logger.new_velocity(i, vc * self.vs_arguments.gain)
            poses = new_poses
            I = new_images
            if self.save_extra:
                logger.foreach_zip(logger.save_current_image, index_for_zip, I)
                logger.foreach_zip(logger.save_error_desired_current_image, index_for_zip, I, Id)
            I_processed = self.vs_method.process_image(I)
            self.vs_method.on_end_iter(i)

        _,_,_, error_vector = self.run_iter(poses, desired_poses, Id_processed, I_processed, scene_indices, num_iters, logger)
        logger.on_update_poses(num_iters, poses, error_vector)
        logger.on_end_vs(desired_poses, self.vs_method.name())
        if self.save_extra:
            hs = [self.vs_arguments.h] * batch_size
            ws = [self.vs_arguments.w] * batch_size
            logger.foreach_zip(logger.make_video, ['images/orig_%05d.png'] * batch_size, ['I.mkv']  * batch_size, ['I'] * batch_size, hs, ws)
            logger.foreach_zip(logger.make_video, ['images/orig_Idiff_%5d.jpg'] * batch_size, ['Idiff.mkv']  * batch_size, ['I - I*'] * batch_size, hs, ws)


    def run(self) -> None:
        '''
        Run servoing for all the samples
        '''
        Logger.save_dict(self.vs_arguments.save_path / 'vs_params.yaml', self.vs_arguments.as_dict())

        for i in range(self.num_samples // self.vs_arguments.batch_size):
            self.run_batch(i)

    def _initial_logging(self, logger: Logger, batch_size: int, Id: RawRGBImageArray, I: RawRGBImageArray,
                        desired_poses: PoseArray, starting_poses: PoseArray, overlaps: NDArray[(Any, 1), Float],
                        scene_indices: IndexArray, p_lat: Optional[Point3DArray], p_lat_d: Optional[Point3DArray]) -> None:
        for fn, data in [(logger.save_desired_image, Id), (logger.save_desired_pose, desired_poses),
                        (logger.save_starting_image, I), (logger.save_starting_pose, starting_poses)]:
            logger.foreach_zip(fn, data)
        logger.foreach_zip(logger.save_array, ['initial_overlap.txt'] * batch_size, overlaps)
        logger.foreach_zip(logger.save_array, ['scene_index.txt'] * batch_size, np.expand_dims(scene_indices, 1))

        if p_lat is not None and p_lat_d is not None:
            logger.foreach_zip(logger.save_array, ['point_look_at.txt'] * batch_size, p_lat)
            logger.foreach_zip(logger.save_array, ['point_look_at_desired.txt'] * batch_size, p_lat_d)

class GenerateLossLandscape():
    def __init__(self, save_dir: Path, method: VSMethod, gp_params: GeneratorParameters,
                generator: MultiGeneratorHandler, batch_size: UnsignedInt,
                samples_per_dim: UnsignedInt, center_pose: Pose, ranges: List[float],
                groups: List[Tuple[str, str]], plot_error: bool, generate_each_component: bool):
        assert len(center_pose) == 6
        assert len(ranges) == 6
        self.n_to_i = {
            'tx': 0,
            'ty': 1,
            'tz': 2,
            'rx': 3,
            'ry': 4,
            'rz': 5,
        }
        self.generator = generator
        self.batch_size = batch_size
        self.center_pose = center_pose
        self.ranges = ranges
        self.method = method
        self.samples_per_dim = samples_per_dim
        self.plot_error = plot_error
        self.groups = groups
        self.h, self.w = itemgetter('h', 'w')(gp_params)
        self.generate_each_component = generate_each_component
        self.save_dir = save_dir / 'plots'
        self.save_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            data = self.generator.image_at_poses(1, self.h, self.w, 0.0, np.array([self.center_pose]), [0], False)
            self.center_pose_images_unprocessed = np.array([s.image() for s in data])
            b = self.method.border
            self.center_pose_image = np.ascontiguousarray(self.center_pose_images_unprocessed[:, b:-b, b:-b])
            self.center_pose_image = to_gray(self.center_pose_image)[0]
            self.center_pose_features, self.Lzd = self.method.compute_features_and_interaction_matrix(self.method.process_image(self.center_pose_images_unprocessed))
            # self.center_pose_features = self.center_pose_features / torch.norm(self.center_pose_features, p=2, dim=-1, keepdim=True)
            self.center_pose_features = self.center_pose_features.cpu().numpy()[0]
            self.Lzd = permute_im_to_vec_rep_if_required_minimal_checks(self.Lzd)[0].cpu().numpy()
            


    def run(self):
        print('Generating plots for method', self.method.name())
        for g in self.groups:
            self.latent_vs_plots(g)
    def _generate_group_with_poses(self, g):
        group_save_dir = self.save_dir / '{}_{}'.format(*g)
        group_save_dir.mkdir(exist_ok=True)
        assert len(g) == 2
        indices = [self.n_to_i[gv] for gv in g]
        poses = np.empty((self.samples_per_dim ** 2, 6))
        range_1 = self.ranges[indices[0]]
        range_2 = self.ranges[indices[1]]
        d1 = np.linspace(-range_1 / 2.0, range_1 / 2.0, self.samples_per_dim)
        d2 = np.linspace(-range_2 / 2.0, range_2 / 2.0, self.samples_per_dim)
        for i in range(self.samples_per_dim):
            for j in range(self.samples_per_dim):
                poses[i * self.samples_per_dim + j] = self.center_pose
                poses[i * self.samples_per_dim + j, indices[0]] += d1[i]
                poses[i * self.samples_per_dim + j, indices[1]] += d2[j]

        if g[0][0] == 'r':
            d1 = np.degrees(d1)
        if g[1][0] == 'r':
            d2 = np.degrees(d2)
        X, Y = np.meshgrid(d1, d2, indexing='ij')
        return poses, X, Y, d1, d2, indices, group_save_dir

    def latent_vs_plots(self, g):
        print('Generating group', g)
        poses, X, Y, d1, d2, indices, group_save_dir, = self._generate_group_with_poses(g)
        g_3d_dir = group_save_dir / '3d'
        g_2d_dir = group_save_dir / '2d'
        g_3d_dir.mkdir(exist_ok=True)
        g_2d_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for i in range(self.samples_per_dim ** 2 // self.batch_size):
                data = self.generator.image_at_poses(self.batch_size, self.h, self.w, 0.0, poses[i * self.batch_size:(i + 1) * self.batch_size], [0 for _ in range(self.batch_size)], False)
                images = np.array([s.image() for s in data])
                f, Lzb = self.method.compute_features_and_interaction_matrix(self.method.process_image(images))
                # f = f / torch.norm(f, p=2, dim=-1, keepdim=True)
                f = f.cpu().numpy()
                Lzb = Lzb.cpu().numpy()
                if i == 0:
                    features = np.empty((len(poses), *f.shape[1:]))
                    Lzs = np.empty((len(poses), *Lzb.shape[1:]))
                features[i * self.batch_size:(i + 1) * self.batch_size] = f
                Lzs[i * self.batch_size:(i + 1) * self.batch_size] = Lzb

        def _rank(Ls: np.ndarray) -> None:
            ranks = np.empty((len(Ls),))
            for i in range(len(Ls)):
                ranks[i]  = np.linalg.matrix_rank(Ls[i])
            print('Has below rank 6 matrix: {}'.format(np.any(ranks < 6)))
        _rank(Lzs)
        Iapprox = np.linalg.pinv(Lzs) @ Lzs
        identity_diff = Iapprox - np.expand_dims(np.eye(6), 0)
        identity_diff_norm = np.linalg.norm(identity_diff, ord='fro', axis=(1, 2))
        print(f'||Lz^* @ Lz  - I||_f: mean = {np.mean(identity_diff_norm)}, max = {np.max(identity_diff_norm)}')

        error = features - self.center_pose_features
        if self.generate_each_component:
            Lzs = permute_im_to_vec_rep_if_required_minimal_checks(torch.tensor(Lzs)).cpu().numpy()
            Lzs_invs = np.linalg.pinv(Lzs)

        error = np.reshape(error, (len(error), -1))
        assert len(error.shape) == 2
        features = np.reshape(features, (len(features), -1))
        error_norm = np.linalg.norm(error, axis=-1)

        # print('Normalization and dot product!!!')
        # error_norm = np.dot(features, self.center_pose_features)
        

        error_component_name = 'I' if 'dvs' in self.method.name() else 'z'
        z_label = r"$\lVert\mathbf{{{0}}} - \mathbf{{{0}}}^*\rVert^2$".format(error_component_name)
        Ze = np.reshape(error_norm, (self.samples_per_dim, self.samples_per_dim))
        self.save_plot_3d(g, X, Y, Ze, z_label, '{}_error.pdf'.format(self.method.name()), group_save_dir)

        def plot_2d_velocity_field(err, vc, save_name, save_dir, colorbar=False):
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel(g[0])
            ax.set_ylabel(g[1])
            plt.imshow(err, cmap=cm.plasma, extent=[d1[0], d1[-1], d2[0], d2[-1]], aspect='auto', origin='lower')
            if isinstance(vc, torch.Tensor):
                vc = vc.cpu().numpy()
            vc_indices = [0, 1] if vc.shape[1] == 2 else indices # if vc.shape == 2, we already have the axes that we want

            vc_1 = np.reshape(vc[:, vc_indices[0]], (self.samples_per_dim, self.samples_per_dim))
            vc_2 = np.reshape(vc[:, vc_indices[1]], (self.samples_per_dim, self.samples_per_dim))
            if indices[0] >= 3: # Using axis index, not the vc_index (Axis Index >= 3, working in degrees)
                vc_1 = np.degrees(vc_1)
            if indices[1] >= 3:
                vc_2 = np.degrees(vc_2)


            if colorbar:
                plt.colorbar()
            plt.quiver(X, Y, vc_1, vc_2, scale_units='xy', angles='xy')
            plt.savefig(str(save_dir / save_name))
            plt.close()

        if hasattr(self.method, 'optimizer'):
            vc_Lid = self.method.optimizer(torch.tensor(self.Lzd, device=self.method.device).unsqueeze(0).float(),
                                        torch.tensor(error, device=self.method.device).float())
            vc_Li = self.method.optimizer(torch.tensor(Lzs, device=self.method.device).float(),
                                        torch.tensor(error, device=self.method.device).float())
            vc_ESM = self.method.optimizer(torch.tensor(0.5 * (Lzs + self.Lzd[None, :, :]), device=self.method.device).float(),
                                        torch.tensor(error, device=self.method.device).float())

            for vc, im_type in [(vc_Lid, 'Lid'), (vc_Li, 'Li'), (vc_ESM, 'ESM')]:
                save_name = '{}_{}.jpg'.format(self.method.name(), im_type)
                plot_2d_velocity_field(Ze, vc, save_name, group_save_dir)




            if Lzs is not None:
                Lz_diff = Lzs - self.Lzd
                diff_norm = np.linalg.norm(Lz_diff, ord='fro', axis=(-2, -1))
                diff_norm = np.reshape(diff_norm, (self.samples_per_dim, self.samples_per_dim))
                save_name = '{}_distance_IM_{}_{}.jpg'.format(self.method.name(), g[0], g[1])
                self.save_plot_3d(g, X, Y, diff_norm, r'$\lVert \mathbf{L_z} - \mathbf{L}_{\mathbf{z}^*}\rVert_F$', save_name, group_save_dir)

                condition_numbers = np.linalg.cond(Lzs, p=2)
                condition_numbers = np.reshape(condition_numbers, (self.samples_per_dim, self.samples_per_dim))
                save_name  = '{}_IM_condition_number.jpg'.format(self.method.name())
                self.save_plot_3d(g, X, Y,condition_numbers, r'$cond(\mathbf{L_z})$', save_name, group_save_dir)
                print('Average Lz condition number: {}'.format(np.average(condition_numbers)))
                condition_numbers = np.linalg.cond(np.transpose(Lzs, axes=(0,2,1)) @ Lzs, p=2)
                condition_numbers = np.reshape(condition_numbers, (self.samples_per_dim, self.samples_per_dim))
                save_name  = '{}_LzTLz_condition_number.jpg'.format(self.method.name())
                self.save_plot_3d(g, X, Y, condition_numbers, r'$cond(\mathbf{L_z}^T\mathbf{L_z})$', save_name, group_save_dir)
                print('Average LzT^Lz condition number: {}'.format(np.average(condition_numbers)))



        if self.generate_each_component and error.shape[-1] < 512:
            for i in range(error.shape[-1]):
                e = error[:, i]
                f = features[:, i]
                Lg1i = Lzs_invs[:, indices[0], i]
                Lg2i = Lzs_invs[:, indices[1], i]

                Lp = np.concatenate((np.expand_dims(Lg1i, -1), np.expand_dims(Lg2i, -1)), axis=-1)
                ve = -Lp * np.expand_dims(e, -1)

                for j in range(2):
                    if indices[j] >= 3:
                        ve[:, j] = np.degrees(ve[:, j])

                save_name = '{}_error_{}_{}_s_{}_2d.jpg'.format(self.method.name(), g[0], g[1], i)
                plot_2d_velocity_field(np.reshape(e ** 2, (self.samples_per_dim, self.samples_per_dim)), ve, save_name, g_2d_dir, True)
                save_name = '{}_{}_{}_s_{}_2d.jpg'.format(self.method.name(), g[0], g[1], i)

                plot_2d_velocity_field(np.reshape(f, (self.samples_per_dim, self.samples_per_dim)), Lzs_invs[:, :, i], save_name, g_2d_dir, True)

                self.save_plot_3d(g, X, Y, np.reshape(e ** 2, (self.samples_per_dim, self.samples_per_dim)),
                        r'$(s_{{{0}}} - s^*_{{{0}}})^2$'.format(i),
                        '{}_error_s_{}.jpg'.format(self.method.name(), i),
                        g_3d_dir
                        )





    def save_plot_3d(self, g, X, Y, Z, z_label, save_name, save_dir):
        _ = self.make_plot_3d(g, X, Y, Z, z_label)
        plt.savefig(str(save_dir / save_name))
        plt.close()

    def make_plot_3d(self, g, X, Y, Z, z_label):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(g[0])
        ax.set_ylabel(g[1])
        ax.set_zlabel(z_label)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                linewidth=0, antialiased=True)
        ax.contourf(X, Y, Z, cmap=cm.plasma, zdir='z', offset=np.min(Z))
        return fig, ax

