import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'text.latex.preamble' : [r'\usepackage{amsmath}']})
import argparse
from enum import Enum

import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from functools import partial


from metric_learning.inference.methods import GridPoseSampler, PoseSampler, UniformPoseSampler

from metric_learning.model.models import ImageEncoder, PoseEmbedder, PoseEmbedderSIREN

from utils.torchvision_transforms import *


def find_k(config, model_path: Path):
    '''
    Plot the correlation between the true latent representation of a pose, and its KNN approximation, for different values of K and weightings
    '''

    # Configure reproducibility parameters: Seeds and pytorch algorithms
    torch.manual_seed(config['torch_seed'])
    np.random.seed(config['np_seed'])
    if not 'cuda_reproducibility' in config:
        config['cuda_reproducibility'] = True
    torch.backends.cudnn.benchmark = not config['cuda_reproducibility']
    torch.backends.cudnn.deterministic = config['cuda_reproducibility']


    print('Creating training dataset...')
    pose_sampler = GridPoseSampler(6, 9, [0.0, 0.0, -0.6], ranges=[0.2, 0.2, 0.2, 60, 60, 90])
    val_pose_sampler = UniformPoseSampler(10 * 1000, [0.0, 0.0, -0.6], ranges=[0.2, 0.2, 0.2, 60, 60, 90])

    pose_encoder = torch.load(model_path, map_location='cpu')

    def compute_pose_reps_and_im(pose_sampler):
        from metric_learning.inference.methods import compute_im
        with torch.no_grad():
            poses = pose_sampler()
            L = compute_im(poses)
            zp, Lp = pose_encoder.forward_with_interaction_matrix(poses, L)
            return zp.cpu().numpy(), Lp.cpu().numpy()
    
    # poses, ims = compute_pose_reps_and_im(pose_sampler)
    print('Computing latent representations of poses...')
    poses, ims = compute_pose_reps_and_im(pose_sampler)
    val_poses, val_ims = compute_pose_reps_and_im(val_pose_sampler)


    uniform_vals = []
    distance_vals = []
    distance_vals_et = []
    distance_vals_er = []
    with torch.no_grad():
        target = torch.from_numpy(np.array([0.0, 0.0, -0.6, 0.0, 0.0, 0.0])).unsqueeze(0).float()
        target = pose_encoder(target).numpy()
    
    from sklearn.neighbors.regression import KNeighborsRegressor
    # Tested K values for KNN
    K = list(range(1, 20, 2)) + list(range(20, 101, 10))
    # K = [1, 10 , 20, 50]
    # G = list(range(3, 11))
    # results = np.zeros((len(G), len(K)))
    knn = KNeighborsRegressor(1, weights='distance', n_jobs=4).fit(poses, ims.reshape(len(ims), -1))
    knn_uniform = KNeighborsRegressor(1, weights='uniform', n_jobs=4).fit(poses, ims.reshape(len(ims), -1))
    def compute_vc(ims, reps):
        '''
        Compute velocity from latent space control law
        '''
        im_inv = np.linalg.pinv(ims)
        v = -np.matmul(im_inv, (reps - target)[..., None])[..., 0]
        return v
    def compute_scores(pred_vc: np.ndarray) -> Tuple[float, float, float, float]:
        '''
        Compute agreement between the "true" (i.e. using the true latent representation of a pose) velocity and the KNN approximated one
        Done for both translational and rotational velocities. Only their direction is compared (cosine similarity).
        Returns the average and standard deviation for the whole pred_vc set, for translational and rotational velocities.
        '''
        normed_true_t = true_vc[:, :3] / np.linalg.norm(true_vc[:, :3], axis=-1, keepdims=True)
        normed_true_r = true_vc[:, 3:] / np.linalg.norm(true_vc[:, 3:], axis=-1, keepdims=True)
        normed_pred_t = pred_vc[:, :3] / np.linalg.norm(pred_vc[:, :3], axis=-1, keepdims=True)
        normed_pred_r = pred_vc[:, 3:] / np.linalg.norm(pred_vc[:, 3:], axis=-1, keepdims=True)

        dot_prod_t = np.einsum('Bt,Bt -> B', normed_true_t, normed_pred_t)
        dot_prod_r = np.einsum('Bt,Bt -> B', normed_true_r, normed_pred_r)

        return np.mean(dot_prod_t), np.std(dot_prod_t), np.mean(dot_prod_r), np.std(dot_prod_r)

    true_vc = compute_vc(val_ims, val_poses)
    for ki, k in enumerate(K):
        knn = knn.set_params(n_neighbors=k)
        knn_uniform = knn_uniform.set_params(n_neighbors=k)
        print(f'Testing for K={k}')

        from sklearn.metrics import r2_score
        pred_ims = knn.predict(val_poses)
        scores = r2_score(val_ims.reshape(len(val_ims), -1), pred_ims, multioutput='raw_values')

        distance_vals.append((np.mean(scores), np.std(scores)))
        pred_vc = compute_vc(pred_ims.reshape(len(val_ims), -1, 6), val_poses)
        # err_t = np.mean(np.linalg.norm(pred_vc[:, :3] - true_vc[:, :3], axis=-1, ord=2))
        # err_r = np.mean(np.linalg.norm(pred_vc[:, 3:] - true_vc[:, 3:], axis=-1, ord=2))
        err_t, std_t, err_r, std_r = compute_scores(pred_vc)

        distance_vals_et.append((err_t, std_t))
        distance_vals_er.append((err_r, std_r))

        pred_ims = knn_uniform.predict(val_poses)
        scores = r2_score(val_ims.reshape(len(val_ims), -1), pred_ims, multioutput='raw_values')
        print(np.mean(scores))
        uniform_vals.append((np.mean(scores), np.std(scores)))



    # Plot R2 score for uniform and distance weighted KNNs
    fig = plt.figure()
    fs = 14
    def plot_error(xs, l, label):
        means = list([v[0] for v in l])
        #stds = list([v[1] for v in l])
        plt.plot(xs, means, label=label)

    plot_error(K, distance_vals, label='Distance weighting')
    plot_error(K, uniform_vals, label='Uniform weighting')
    plt.xlabel('K', fontsize=fs)
    plt.ylabel(r'$R^2$', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid()
    plt.savefig('Fig_k_uniform_vs_dist.pdf')
    plt.close()
    # Plot cosine similarity for 
    fig = plt.figure()
    plot_error(K, distance_vals_et, label='Translation')
    plot_error(K, distance_vals_er, label='Rotation')
    plt.xlabel(r'$K$', fontsize=fs)
    plt.ylabel(r'Agreement between KNN $\mathbf{v}$ and true $\mathbf{v}$', fontsize=fs)
    plt.legend(fontsize=fs)
    fig.gca().tick_params(axis='both', which='major', labelsize=fs)
    plt.grid()
    plt.savefig('Fig_k_agreement.pdf')



if __name__ == '__main__':
    import yaml
    import argparse
    

    with open('config.yaml', 'r') as f:
        project_conf = yaml.load(f)
        data_root = Path(project_conf['data_dir'])
        save_dir = Path(project_conf['model_save_root_dir']) / 'mlvs'

    scene_list = [str(data_root / 'scene_real_lower_res.jpg')]
    scene_list_val = [str(data_root / 'scene_real_lower_res.jpg')]
    # Dataset parameters
    base_stds = np.asarray([0.1, 0.1, 10.0, 16.0])
    seed = 420
    Z = 0.6
    generator_parameters = {
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.3,
        'half_width_m': 0.4,
        'image_paths': scene_list,
        'gaussian_sets_sigmas': [base_stds],
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.1,
        'base_camera_height': Z,
        'desired_pose_max_rotation_xy': 5.0,
        'h': 234,
        'w': 234,
        'px': 300,
        'py': 300,
        'u0': 124,
        'v0': 114,
        'augmentation_on_two_images': True,
        'max_translation_height': 0.05,
        'gaussian_lights_gain_std': 0.2,
        'gaussian_lights_max_count': 4,
        'gaussian_lights_base_std_scene_rel': 0.5,
        'gaussian_lights_std_spread_scene_rel': 0.2,
        'global_lighting_augmentation_std': 0.2,
        'global_lighting_augmentation_bias_std': 0.1,
        'visibility_threshold': 0.2,
        'use_scene_cutout': False,
        'scene_cutout_size_min_rel': 0.01,
        'scene_cutout_size_max_rel': 0.2,
        'scene_cutout_max_dist_from_center': 0.3,
        'scene_cutout_use_pixel_level': False,
        'scene_cutout_max_count': 3
    }
    # Create validation dataset, change only the seed and the scene images
    val_params = generator_parameters.copy()
    val_params['base_seed'] = 8192
    val_params['image_paths'] = scene_list_val
    print('Created params')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    latent_dim = 32
    config = {
        'device': device,
        'border': 5,
        'Z': Z,
        # seeds
        'np_seed': 17,
        'torch_seed': 42,
        # Hyperparameters
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_plateau_params': {
            'factor': 0.5,
            'patience': 10,
            'threshold': 0.01,
            'threshold_mode': 'rel',
            'min_lr': 1e-6,
            'verbose': 1,
            'mode': 'min'
        },
        'num_epochs': 200,
        'num_samples_per_epoch': 200 * 1000,
        'batch_size': 256,
        'latent_dim': latent_dim,
        'num_workers': 0,
        # Scene loading and save parameters
        'data_root': data_root,
        'models_root': save_dir,
        'generator_params': generator_parameters,
        'val_generator_params': val_params,

        # Pose generation
        'dataset_params': {
            'center_Z': -Z,
            'look_at_half_ranges': [0.2, 0.2],
            'look_from_half_ranges': [0.2, 0.2, 0.2],
            'rz_max_deg': 180
        },
        'use_invariance_loss': True,
        'epochs_train_decoder_only': 0,
        'cuda_reproducibility': False
    }

    def make_name(_dt, config):
        return 'test_model'
    find_k(config, make_name)
