from pathlib import Path
import torch
import sys
import numpy as np
import yaml
import argparse
from typing import Callable, Tuple
from datetime import datetime


from generator import MultiGeneratorHandler, SceneSet
from geometry import *
from inference.augmentors import str_to_augmentor
from inference.post_run_operations import *
from inference.pose_generators import *
from inference.optimizers import *
from inference.interaction_matrix_mixer import *
from inference.methods import *
from inference.utils import *
from inference.experiments import *
from utils.custom_typing import *

MethodBuildTuple = Tuple[float, str, Callable[[VSArguments], VSMethod]]

def fix_backwards_model_compat():
    import aevs.model.resnet_im_computable
    import aevs.model.im_computable
    sys.modules['models.resnet_im_computable'] = aevs.model.resnet_im_computable
    sys.modules['models.im_computable'] = aevs.model.im_computable

def get_specific_poses(Z):
    radius = 0.2
    angles = [np.radians(45.0 * i) for i in range(8)]
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    s = np.array([[x[i], y[i], -Z, 0.0, 0.0, 0.0] for i in range(len(angles))])
    d = np.array([[0.0, 0.0, -Z, 0.0, 0.0, 0.0] for i in range(len(s))])
    specific_poses = (s, d)
    return specific_poses


def get_pose_generator(base_args, pose_type):
    if pose_type == 'chosen':
        s, d = get_specific_poses(0.6)
        return SpecificPoseGenerator(base_args, s, d)
    if pose_type == 'random':
        return RandomPoseGenerator(base_args)
    elif pose_type == 'look_at':
        return PoseLookingAtSamePointGenerator.get_default(base_args)
    elif pose_type == 'look_at_with_noise':
        return PoseLookingAtSamePointWithNoiseGenerator.get_default(base_args, 'circle')
    elif pose_type == 'look_at_with_noise_rz':
        return PoseLookingAtSamePointWithNoiseAndRotationZGenerator.get_default(base_args)
    elif pose_type == 'screw_motion':
        return PoseScrewMotionGenerator.get_default(base_args)
    elif pose_type == 'translation_xy':
        txy_motions(generator, num_samples, batch_size, h, w, 0.0, 10, 0.6, 0.1, desired_poses=np.zeros((1, 6)))
    elif pose_type == 'translation':
        txyz_motions(generator, num_samples, batch_size, h, w,
            0.0, 1, Z, 0.3, 0.3, desired_poses=[[0, 0, -Z, 0, 0, 0]])





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', help='Scene image to create the scene. Should be in the data folder, specified in the config.yaml', type=str, default='hollywood.jpg')
    parser.add_argument('--run_file', help='YAML file containing the methods to run', type=str)
    parser.add_argument('--pose_type', help='Which type of test is to be ran', type=str)
    parser.add_argument('--id_augmentation', help='How to augment desired images for VS', default=None, type=str)
    parser.add_argument('--dump_images', help='Save the images at each iteration', action='store_true')
    parser.add_argument('--save_plots', help='Generate the plots for each servoing example', action='store_true')
    parser.add_argument('--num_iters', help='VS number of iterations', type=int, default=1500)
    parser.add_argument('--num_samples', help='VS number of samples (no effect if pose_type == chosen)', type=int, default=500)
    parser.add_argument('--batch_size', help='Number of VS examples ran in parallel', type=int, default=250)
    parser.add_argument('--generate_landscape', help='Generate the VS loss landscape plots', action='store_true')
    parser.add_argument('--generate_landscape_per_component', help='Generate the VS loss landscape plots for each of the feature component', action='store_true')
    parser.add_argument('--no_inference', help='Do not run inference', action='store_true')
    parser.add_argument('--early_stopping', help='Stop running VS for samples that already converged (wrt velocities)', action='store_true')
    parser.add_argument('--save_kitti', help='Save trajectories as Kitti format', action='store_true')
    
    
    parser.add_argument('--device', help='Device to run VS on: cuda/cpu', type=str, default='cuda')

    args = parser.parse_args()

    fix_backwards_model_compat()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    with open('config.yaml', 'r') as f:
        project_conf = yaml.load(f)
        data_root = Path(project_conf['data_dir'])
        save_root = Path(project_conf['inference_save_dir']) / date_time
        save_root.mkdir(exist_ok=True)

    scene_list = [str(data_root / args.image_name)]
    scenes = SceneSet(scene_list)
    seed = 8191
    np.random.seed(41)
    torch.manual_seed(121)

    base_stds = np.asarray([0.1, 0.1, 10.0, 16.0])
    generator_parameters: GeneratorParameters = {
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.3,
        'half_width_m': 0.4,
        'image_paths': scene_list,
        'desired_pose_max_rotation_xy': 5.0,
        'gaussian_sets_sigmas': [base_stds],
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.1,
        'base_camera_height': 0.6,
        'h': 234,
        'w': 234,
        'px': 300,
        'py': 300,
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
    Z = 0.6
    pose_type = args.pose_type
    specific_poses = None
    if pose_type == 'chosen': # 2D example
        radius = 0.2
        angles = [np.radians(45.0 * i) for i in range(8)]
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        s = np.array([[x[i], y[i], -Z, 0.0, 0.0, 0.0] for i in range(len(angles))])
        d = np.array([[0.0, 0.0, -Z, 0.0, 0.0, 0.0] for i in range(len(s))])
        specific_poses = (s, d)

    run_file_path = Path(args.run_file)
    dump_images = args.dump_images
    save_plots = args.save_plots
    num_samples = args.num_samples if specific_poses is None else len(specific_poses[0])
    num_iters = args.num_iters
    augmentation_factor = 0.0
    h, w = 234, 234
    batch_size = min(args.batch_size, num_samples)
    device = args.device
    generate_landscape = args.generate_landscape
    generate_each_landscape_component = args.generate_landscape_per_component
    run_inference = not args.no_inference
    early_stopping = args.early_stopping
    save_kitti = args.save_kitti

    assert run_file_path.exists(), 'run file {} could not be found'.format(run_file_path)

    generator = MultiGeneratorHandler(batch_size, scenes, generator_parameters)
    points_look_at = None
    points_look_at_desired = None

    pose_sampler_base_args = (generator, num_samples, batch_size, h, w, augmentation_factor)

    pose_generator = get_pose_generator(pose_sampler_base_args, pose_type)
    sample_data = pose_generator()
    
        # elif pose_type == 'translation_xy':
        #     starting_poses, starting_images, desired_poses, desired_images, overlaps = txy_motions(generator, num_samples, batch_size, h, w, 0.0, 10, 0.6, 0.1, desired_poses=np.zeros((1, 6)))
        # elif pose_type == 'translation':
        #     starting_poses, starting_images, desired_poses, desired_images, overlaps = txyz_motions(generator, num_samples, batch_size, h, w,
        #      0.0, 1, Z, 0.3, 0.3, desired_poses=[[0, 0, -Z, 0, 0, 0]])

    

    save_root.mkdir(exist_ok=True)
    def load_dicts():
        '''Load default and global parameters for inference'''
        def load_yaml(path: Path) -> yaml.Node:
            with open(str(path), 'r') as f:
                res = yaml.safe_load(f)
            return res
        inference = Path('inference')
        return load_yaml(inference / 'globals.yaml'), load_yaml(inference / 'default_parameters.yaml'), load_yaml(run_file_path)
    global_params, defaults, runs = load_dicts()

    defaults['Z'] = Z
    defaults['batch_size'] = batch_size
    from inference.io import *
    builder_data = vs_builders_list_from_yaml(runs, defaults, global_params, pose_type, device)

    id_augmentation_str = args.id_augmentation
    # Id_augmentor = str_to_augmentor(args.id_augmentation)

    vs_argument_fn = lambda g, s: VSArguments(g, num_iters, batch_size, s, h, w)
    if run_inference or generate_landscape:
        for gain, save_name, fn in builder_data:

            generator = MultiGeneratorHandler(batch_size, scenes, generator_parameters)
            method_save_dir = save_root / save_name
            args = vs_argument_fn(gain, method_save_dir)
            method_save_dir.mkdir(exist_ok=True)
            method = fn(args) # Build method
            
            if generate_landscape:
                samples_per_dim = 25
                g2 = MultiGeneratorHandler(samples_per_dim, scenes, generator_parameters)
                center_pose = [0.0, 0.0, -Z, 0.0, 0.0, 0.0]
                ranges = [0.2, 0.2, 0.2, np.pi/4, np.pi/4, np.pi/2]
                GenerateLossLandscape(save_root / save_name, method, generator_parameters, g2, samples_per_dim,
                                    samples_per_dim, center_pose, ranges,
                                    [['tx', 'ty'], ['tx', 'tz'], ['tx', 'ry'], ['tz', 'rz']],
                                    True, generate_each_component=generate_each_landscape_component).run()

            if run_inference:
                torch.manual_seed(489) # Reset seed for data augmentation to be consistent across methods
                Id_augmentor = str_to_augmentor(id_augmentation_str)
                runner = InferenceRun(True, method, args, generator, sample_data,
                                    Id_augmentor, dump_images, save_plots, save_kitti, early_stopping=early_stopping)
                runner.run()
            del method
            torch.cuda.empty_cache()

    
    
    global_results_path = save_root / 'global_results'
    global_results_path.mkdir(exist_ok=True)

    error_generator = GeneratePoseErrorStats({
        save_root / bd[1]: bd[1] for bd in builder_data
    }, global_results_path)
    print(error_generator.path_to_name_dict)
    error_generator.visit_multiple_model_results(save_root)
    
    

    ground_truth_name = None
    for gain, save_name, fn in builder_data[::-1]: # Find which method is the ground truth
        args = vs_argument_fn(gain, save_root / save_name)
        method = fn(args)

        if isinstance(method, TruePBVS): # In our case, oracle is PBVS
            print(f'Ground truth name {save_name}')
            ground_truth_name = save_name
            break
        torch.cuda.empty_cache()
    if ground_truth_name is not None:
        error_generator = ComputeTrajectoryStats({
            save_root / bd[1]: bd[1] for bd in builder_data
        }, save_root / ground_truth_name, global_results_path, save_plots)
        error_generator.visit_multiple_model_results(save_root)
    else:
        print('Ground truth PBVS not found, not computing trajectory metrics!')
