import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
import torch
import sys
import numpy as np
import cv2
import yaml
import faulthandler
import argparse


from generator import ClusteringTripletOutput, Generator, MultiGeneratorHandler, DVS, SceneSet, DCTBatchServo
from geometry import *
from inference.post_run_operations import *
from inference.pose_generators import *
from inference.logger import Logger
from inference.optimizers import *
from inference.interaction_matrix_mixer import *
from inference.methods import *
from inference.utils import *
from inference.experiments import *
from utils.datasets import load_image_woof_paths
matplotlib.use('agg')

if __name__ == "__main__":
    # data_root = Path('~/code/visp_inference/data').expanduser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_type', help='Which type of test is to be ran', type=str)
    parser.add_argument('--dump_images', help='Save the images at each iteration', action='store_true')
    parser.add_argument('--save_plots', help='Generate the plots for each servoing example', action='store_true')
    parser.add_argument('--num_iters', help='VS number of iterations', type=int, default=1500)
    parser.add_argument('--num_samples_per_scene', help='VS number of samples per scene (no effect if pose_type == chosen)', type=int, default=10)
    parser.add_argument('--batch_size', help='Number of VS examples ran in parallel', type=int, default=100)
    parser.add_argument('--no_inference', help='Do not run inference', action='store_true')
    parser.add_argument('--early_stopping', help='Stop running VS for samples that already converged (wrt velocities)', action='store_true')
    parser.add_argument('--save_kitti', help='Save trajectories as Kitti format', action='store_true')
    parser.add_argument('--device', help='Device to run VS on: cuda/cpu', type=str, default='cuda')
    parser.add_argument('--run_file', help='YAML file containing the methods to run', type=str)
    parser.add_argument('--image_set', help='One of train|val|test: which image set to run examples on', type=str, default='test')
    

    args = parser.parse_args()

    r = Path('/local/sfelton')
    data_root = r / 'generator_scenes/imagewoof2'
    
    seed = 8191
    np.random.seed(41)
    torch.manual_seed(121)

    base_stds = np.asarray([0.1, 0.1, 10.0, 16.0])
    generator_parameters = {
        # 'base_seed': [seed * (i + 1) for i in range(num_workers)],
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.3,
        'half_width_m': 0.4,
        'image_paths': [],
        'desired_pose_max_rotation_xy': 5.0,
        # 'image_paths': [str((data_root / 'scenes_cvpr09_hollywood' / 'scene_real_lower_res.jpg').expanduser())],
        # 'image_paths': [str((data_root / 'scene_natural.jpg').expanduser())],
        #'gaussian_sets_sigmas': [base_stds],
        #'gaussian_sets_probabilities': [1],
        'gaussian_sets_sigmas': [base_stds],
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.1,
        'base_camera_height': 0.6,
        'h': 234,
        'w': 234,
        'px': 570,
        'py': 570,
        'u0': 124,
        'v0': 114,
        'augmentation_on_two_images': True,
        'max_translation_height': 0.1,
        'gaussian_lights_gain_std': 0.2,
        'gaussian_lights_max_count': 4,
        'gaussian_lights_base_std_scene_rel': 0.5,
        'gaussian_lights_std_spread_scene_rel': 0.2,
        'global_lighting_augmentation_std': 0.2,
        'global_lighting_augmentation_bias_std': 0.1,
        'visibility_threshold': 0.25,
        'use_scene_cutout': False,
        'scene_cutout_size_min_rel': 0.01,
        'scene_cutout_size_max_rel': 0.2,
        'scene_cutout_max_dist_from_center': 0.3,
        'scene_cutout_use_pixel_level': False,
        'scene_cutout_max_count': 3
    }
    Z = 0.7
    pose_type = args.pose_type
    dump_images = args.dump_images
    save_plots = args.save_plots
    num_iters = args.num_iters
    augmentation_factor = 0.0
    h, w = 234, 234
    batch_size = args.batch_size
    device = args.device
    run_inference = not args.no_inference
    early_stopping = args.early_stopping
    save_kitti = args.save_kitti

    
    # save_dir = Path('{}/vs_results'.format(model_path.parent if not servo_on_original_images else '{}/true_dvs'.format(model_path.parent.parent))).expanduser()
    # generator = MultiGeneratorHandler(batch_size, scenes, generator_parameters)
    points_look_at = None
    points_look_at_desired = None
    root = r / 'nnimvs/vs_results'

    root.mkdir(exist_ok=True)
    dir_name = 'nnimvs'
    model_path = Path(r / 'models/{}'.format(dir_name)).expanduser()
    run_file_path = args.run_file
    
    def load_dicts():
        def load_yaml(path: Path) -> yaml.Node:
            with open(str(path), 'r') as f:
                res = yaml.safe_load(f)
            return res
        inference = Path('inference')
        return load_yaml(inference / 'globals.yaml'), load_yaml(inference / 'default_parameters.yaml'), load_yaml(run_file_path)
    globals, defaults, runs = load_dicts()

    defaults['Z'] = Z
    defaults['batch_size'] = batch_size
    from inference.io import *
    builder_data = vs_builders_list_from_yaml(runs, defaults, globals, pose_type, device)
    

    # Random vs trained net experiment
    # for i in range(5):
    #     print('Adding random and trained nets #{} for random vs trained exp'.format(i))
    #     nnimvs_fn = make_nnimvs_fixed_params_fn(5e-1, generator_parameters, 16, False,
    #                                             mixer=AverageCurrentAndDesiredInteractionMatrices(),
    #                                             final_name_suffix='ESM', optimizer=make_default_opt())
    #     builder_data = builder_data + [nnimvs_fn('random_net_{}'.format(i)), nnimvs_fn('trained_net_{}'.format(i))]
        
    from functools import partial
    from utils.datasets import load_image_net_images_paths
    vs_argument_fn = lambda g, s, bs: VSArguments(g, num_iters, bs, s, h, w)
    if run_inference:
        for bd  in builder_data:
            gain, save_name, fn = bd
            scene_paths = load_image_woof_paths(data_root, set=args.image_set)
            scenes = SceneSet(scene_paths)
            g = generator_parameters.copy()
            g['image_paths'] = scene_paths

            samples_per_scene = args.num_samples_per_scene
            scene_count = len(scene_paths)
            num_samples = samples_per_scene * scene_count
            bs = batch_size
            while num_samples % bs != 0:
                bs -= 1
            print(f'Batch size is {bs}')
            generator = MultiGeneratorHandler(bs, scenes, g)
            vs_args = vs_argument_fn(gain, root / save_name, bs)
            (root / save_name).mkdir(exist_ok=True)
            if pose_type == 'look_at_with_noise':
                starting_poses, starting_images, desired_poses,\
                desired_images, overlaps, points_look_at,\
                points_look_at_desired, scene_indices = multiscene_poses_look_at_with_noise(
                                                                generator, samples_per_scene, bs, h, w, 0.0,
                                                                [0.0, 0.0, -Z, 0.0, 0.0, 0.0],
                                                                [0.0, 0.0, 0.0], [0.4, 0.4, 0.15], 0.1, scene_count)
            elif pose_type == 'screw_motion':
                starting_poses, starting_images, desired_poses, desired_images, overlaps, scene_indices = multiscene_poses_screw_motion(
                                                                generator, samples_per_scene, bs, h, w, 0.0,
                                                                [0.0, 0.0, -Z, 0.0, 0.0, 0.0], 0.3, 70,
                                                                scene_count)
            
            method = fn(vs_args)

            runner = InferenceRun(True, method, vs_args, generator,
                                starting_poses, starting_images, desired_poses, desired_images,
                                overlaps, points_look_at, points_look_at_desired, scene_indices,
                                dump_images, save_plots, save_kitti, early_stopping=early_stopping)
            runner.run()
   
    error_generator = GeneratePoseErrorStats({
        root / bd[1]: bd[1] for bd in builder_data
    })
    error_generator.visit_multiple_model_results(root)
