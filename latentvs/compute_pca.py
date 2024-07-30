import faulthandler

# matplotlib.use('agg')
from datetime import datetime
from operator import itemgetter

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA



from datasets.AEDataset import StoredLookAtAEDataset
def compute_PCA(config, make_name):
    faulthandler.enable()
    print('Starting training...')
    print('Config:')
    print(config)
    np.random.seed(config['np_seed'])
    root = config['models_root']
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = make_name(date_time, config)
    save_root = (root / '{}'.format(model_name)).expanduser()
    save_root.mkdir(exist_ok=True)

    batch_size = itemgetter('batch_size')(config)

    batches_per_epoch = config['num_samples_per_epoch'] // batch_size
    n_components = config['n_components']
    num_workers, border = itemgetter('num_workers', 'border')(config)
    h, w = config['generator_params']['h'], config['generator_params']['w']
    dataset = StoredLookAtAEDataset(batches_per_epoch, batch_size, num_workers, config['generator_params'], look_at_parameters=config['dataset_params'], augmentation_factor=0.0)

    train_data = dataset.data
    train_data = train_data[:, 0, border:-border, border:-border]
    train_data = train_data.reshape((len(train_data), -1))
    print(train_data)
    print('Train data shape = ', train_data.shape)
    pca = PCA(n_components, random_state=config['pca_seed'])
    pca = pca.fit(train_data)

    print(pca.explained_variance_ratio_)
    np.save(str(save_root / 'components.npy'), pca.components_)
    np.save(str(save_root / 'mean.npy'), pca.mean_)


if __name__ == '__main__':
    faulthandler.enable()
    print('Start!')
    # data_root = Path('~/code/visp_inference/data').expanduser()
    data_root = Path('/local/sfelton/generator_scenes')
    scene_list = [str(data_root / 'scene_real_lower_res.jpg')]
    scene_list_val = [str(data_root / 'scene_real_lower_res.jpg')]

    base_stds = np.asarray([0.1, 0.1, 10.0, 16.0])
    seed = 420
    Z = 0.6
    generator_parameters = {
        # 'base_seed': [seed * (i + 1) for i in range(num_workers)],
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.3,
        'half_width_m': 0.4,
        'image_paths': scene_list,
        # 'image_paths': [str((data_root / 'scenes_cvpr09_hollywood' / 'scene_real_lower_res.jpg').expanduser())],
        # 'image_paths': [str((data_root / 'scene_natural.jpg').expanduser())],
        #'gaussian_sets_sigmas': [base_stds],
        #'gaussian_sets_probabilities': [1],
        'gaussian_sets_sigmas': [base_stds],
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.1,
        'base_camera_height': Z,
        'desired_pose_max_rotation_xy': 5.0,
        'h': 234,
        'w': 234,
        'px': 570,
        'py': 570,
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
    val_params = generator_parameters.copy()
    val_params['base_seed'] = 8192
    val_params['image_paths'] = scene_list_val
    print('Created params')
    n_components = 500
    dims = (generator_parameters['h'], generator_parameters['w'])
    config = {
        'border': 5,
        'np_seed': 17,
        'pca_seed': 43,
        'num_samples_per_epoch': 10 * 1000,
        'batch_size': 50,
        'n_components': n_components,
        'num_workers': 0,
        'data_root': data_root,
        'models_root': Path('/local/sfelton/models/pca'),
        'generator_params': generator_parameters,
        'val_generator_params': val_params,
        'dataset_params': {
            'center_Z': -Z,
            'look_at_half_ranges': [0.2, 0.2],
            'look_from_half_ranges': [0.4, 0.4, 0.2]
        },
        'Z': Z,
    }
    def make_name(date_time, config):
        return 'PCA_{}_{}'.format(date_time, config['n_components'])

    compute_PCA(config, make_name)
