import argparse
from enum import Enum
from datetime import datetime
from operator import itemgetter
import pprint
import yaml
import argparse
import torch
import torchvision as tv
import numpy as np
from pathlib import Path
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from metric_learning.model.models import ImageEncoder, PoseEmbedder
from metric_learning.losses import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets.AEDataset import FourDOFDataset, StoredLookAtAEDatasetWithNearestNeighbors, TranslationDataset
from utils.callbacks import DispatchCallback, SaveModelCallback
from utils.metrics import MetricsList, Metric
from utils.datasets import compute_reprojection_error, get_se3_dist_matrices
from torchvision import transforms
from utils.torchvision_transforms import *


class AddGaussianNoise(object):
    '''
    Gaussian noise augmentation
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def train(config, make_model_name):
    print('Starting training...')
    print('Config:')
    pprint.pprint(config)
    device = config['device']

    
    # Configure reproducibility parameters: Seeds and pytorch algorithms
    torch.manual_seed(config['torch_seed'])
    np.random.seed(config['np_seed'])
    if not 'cuda_reproducibility' in config:
        config['cuda_reproducibility'] = True
    torch.backends.cudnn.benchmark = not config['cuda_reproducibility']
    torch.backends.cudnn.deterministic = config['cuda_reproducibility']

    # Where to save the model
    root = config['models_root']
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = make_model_name(date_time, config) # Generate experiment name with given function
    save_root = (root / model_name).expanduser()
    save_root.mkdir(exist_ok=True)

    #Create Tensorboard writer for logging
    tb_dir = save_root / 'logs'
    tb_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(str(tb_dir.expanduser()))

    # Dump the training parameters
    with open(str(save_root / 'config.yaml'), 'w') as f:
        import yaml
        yaml.dump(config, f)

    # Traning parameters
    num_epochs, batch_size = itemgetter('num_epochs', 'batch_size')(config)
    batches_per_epoch = config['num_samples_per_epoch'] // batch_size
    latent_dim = config['latent_dim']
    num_workers, border = itemgetter('num_workers', 'border')(config)
    h, w = config['generator_params']['h'], config['generator_params']['w']
    b = config['border']

    # Create the training metrics, one for each loss
    make_metric = partial(lambda n: Metric(n, tb_writer))
    metrics = MetricsList([
        make_metric('Image-pose distance'),
        make_metric('Metric learning poses'),
        make_metric('Image invariance'),

        make_metric('Loss'),
    ])

    distance_metric = 'se3'
    neighboring_poses = config['neighbor_poses']
    count_perturbations = config['image_perturbations']

    # Model creation
    image_encoder = ImageEncoder(latent_dim, encoder_version='34', width_factor=1.0, use_coord_conv=False).to(device)
    pose_encoder = PoseEmbedder(latent_dim, hidden_counts=[32, 64, 128, 64]).to(device)

    print(f'Image Encoder has {sum(p.numel() for p in image_encoder.parameters())} parameters')
    print(f'Pose Encoder has {sum(p.numel() for p in pose_encoder.parameters())} parameters')
    print('Training on', device)



    print('Creating training dataset...')
    dataset = StoredLookAtAEDatasetWithNearestNeighbors(batches_per_epoch, batch_size, select_k_nn=neighboring_poses, distance_metric=distance_metric,
                                                         num_workers=num_workers, generator_parameters=config['generator_params'],
                                                         look_at_parameters=config['dataset_params'],
                                                         rz_max_deg=config['dataset_params']['rz_max_deg'])


    dl = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=dataset.get_worker_init_fn())
    print('Creating validation dataset...')

    val_dataset = StoredLookAtAEDatasetWithNearestNeighbors(batches_per_epoch // 2, batch_size,  neighboring_poses, distance_metric, num_workers,
                                                            config['val_generator_params'], config['dataset_params'],
                                                            rz_max_deg=config['dataset_params']['rz_max_deg'])

    val_dl = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=val_dataset.get_worker_init_fn())


    # Optimizer
    optimizer = Adam([{'params': image_encoder.parameters(), 'lr': config['lr']}, {'params': pose_encoder.parameters()}],
                     lr=config['lr'], eps=1e-5, weight_decay=config['weight_decay'])
    lr_routine = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_plateau_params'])

    def model_forward(Is, poses):
        zI = image_encoder(Is)
        zP = pose_encoder(poses)
        return zI, zP

    


    # Save Model callback
    callbacks = DispatchCallback([SaveModelCallback('val_Loss', 'min', image_encoder, None,
                                    save_root / (model_name + '_image.pth'), None),
                                    SaveModelCallback('val_Loss', 'min', pose_encoder, None,
                                    save_root / (model_name + '_pose.pth'), None)])




    augmentation = transforms.Compose([
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), inplace=False),
        ColorJitter(brightness=0.6, contrast=0.0),
        AddGaussianNoise(0.0, 30.0),
    ])

    def generate_perturbation(Is, count):
        B, C, H, W = Is.size()
        res = torch.empty((B, count, C, H, W), device=Is.device, requires_grad=False)
        for b in range(B):
            for i in range(count):
                res[b, i] = augmentation(Is[b])
        return res

    def save_images(I, count_perturb):
        from PIL import Image
        I = I.cpu().numpy()
        for i in range(len(I)):
            for p in range(count_perturb + 1):
                im = Image.fromarray(I[i, p, 0]).convert('L')
                im.save(results_dir / f'{i}_{p}.png', 'PNG')


    def do_iter(epoch, batch_idx, batch, is_train):
        '''Perform one training/validation iteration (single batch)'''
        should_do_log = not is_train and batch_idx == 0 and (epoch % (num_epochs // 10) == 0 or epoch == 1)

        if is_train:
            callbacks.on_batch_begin(batch_idx)

        Is, rs = batch[0].to(device), batch[1].to(device).float()
        Is = Is[:, :, b:-b, b:-b].contiguous() # Images acquired at 234 x 234 to match with AEVS, cropped to 224 x 224
        B, C, H, W = Is.size()
        Is_perturb = generate_perturbation(Is, count_perturbations)
        Is = Is.unsqueeze(1)
        Is = torch.cat((Is, Is_perturb), dim=1)

        Is = Is.view(-1, C, H, W)
        Is = image_encoder.preprocess(Is)
        if is_train:
            optimizer.zero_grad()

        zI_full, zP = model_forward(Is, rs)
        zI_full = zI_full.view(B, count_perturbations + 1, -1)

        # Compute losses
        image_pose_loss = loss_distance_image_pose(zI_full, zP)

        if config['use_invariance_loss']:
            image_perturb_loss = image_invariance_loss(zI_full)
        else:
            image_perturb_loss = torch.tensor(0.0, device=device)

        target_dist = None
        if distance_metric == 'se3':
            t_dist, r_dist = get_se3_dist_matrices(rs.cpu().numpy())
            t_mean, r_mean = dataset.t_dist_mean, dataset.r_dist_mean
            target_dist = t_dist / t_mean + r_dist / r_mean
            if batch_idx == 0:
                print(t_mean, r_mean)
        elif distance_metric == 'reprojection_error':
            target_dist = compute_reprojection_error(dataset.points3d, rs.cpu().numpy())

        if isinstance(dataset, TranslationDataset):
            metric_loss = metric_learning_loss(zP, torch.from_numpy(t_dist).to(device))
        elif isinstance(dataset, FourDOFDataset):
            metric_loss = metric_learning_loss(zP, torch.from_numpy(target_dist).to(device))
        else:
            metric_loss = metric_learning_loss(zP, torch.from_numpy(target_dist).float().to(device))

        loss = image_pose_loss + metric_loss + image_perturb_loss
        metric_values = [image_pose_loss.item(), metric_loss.item(), image_perturb_loss.item(), loss.item()]

        if is_train:
            metrics.new_values_train(metric_values)
            loss.backward()
            optimizer.step()
            callbacks.on_batch_end(batch_idx)
            f = (batch_idx + 1, len(dl), *metrics.get_train_values_as_display())
            print('{}/{}: Image-pose distance: {}, ML loss: {}, Invariance loss: {}, Loss: {}'.format(*f), end='\r', sep='')
        else:
            metrics.new_values_val(metric_values)

        if should_do_log:
            # Add embedding to tensorboard
            Ics = Is
            Itb = image_encoder.unprocess(Ics)
            tb_writer.add_embedding(zI_full.view(-1, latent_dim), label_img=Itb / 255.0, global_step=epoch, tag=date_time)

            # Add input images to tensorboard
            Itb = torch.clamp(Itb, 0, 255)
            tb_writer.add_image('originals', tv.utils.make_grid(Itb.type(torch.uint8)), epoch)


    results_dir = save_root / 'results'
    results_dir.mkdir(exist_ok=False)
    print('Starting training...')

    for epoch in range(1, num_epochs + 1):
        image_encoder.train()
        pose_encoder.train()
        
        print('Epoch: ', epoch, '/', num_epochs, sep='')
        for batch_idx, batch in enumerate(iter(dl)):
            do_iter(epoch, batch_idx, batch, True)

        print('Running eval')
        dataset.on_epoch_end(epoch)
        with torch.no_grad():
            image_encoder.eval()
            pose_encoder.eval()
            for batch_idx, batch in enumerate(iter(val_dl)):
                do_iter(epoch, batch_idx, batch, False)
        val_dataset.on_epoch_end(epoch)

        env_dict = metrics.values_as_dict()
        lr_routine.step(env_dict['val_Loss'])
        print(env_dict)
        metrics.write_to_tensorboard(epoch)
        callbacks.on_epoch_end(epoch, env_dict)
        metrics.reset()
        image_encoder.train()

    del image_encoder, dl, val_dl, optimizer


class ImagesDataset(Enum):
    '''
    Training data source
    '''
    HollywoodTriangle = 'hollywood'
    Electronics = 'electronics'
    Pipe = 'pipe'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a model ')
    parser.add_argument('--images', type=str, default=ImagesDataset.HollywoodTriangle.value,
                    help='Dataset to use for training: one of hollywood/cars/dogs')
    parser.add_argument('--train_take_count', type=int, default=-1,
                    help='''When using a multi-image dataset,
                     how many different images to take for training''')

    with open('config.yaml', 'r') as f:
        project_conf = yaml.load(f)
        data_root = Path(project_conf['data_dir'])
        save_dir = Path(project_conf['model_save_root_dir']) / 'mlvs'

    args = parser.parse_args()
    images_dataset = ImagesDataset(args.images)

    scene_list, scene_list_val = None, None
    if images_dataset == ImagesDataset.HollywoodTriangle:
        scene_list = [str(data_root / 'hollywood.jpg')]
        scene_list_val = [str(data_root / 'hollywood.jpg')]
    elif images_dataset == ImagesDataset.Electronics:
        scene_list = [str(data_root / 'electronics.png')]
        scene_list_val = [str(data_root / 'electronics.png')]
    elif images_dataset == ImagesDataset.Pipe:
        scene_list = [str(data_root / 'pipe.png')]
        scene_list_val = [str(data_root / 'pipe.png')]



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
        'num_samples_per_epoch': 1 * 1000,
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
        'image_perturbations': 2,
        'neighbor_poses': 3,
        'cuda_reproducibility': False
    }

    def make_name(_dt, _config):
        return 'test_model'
    train(config, make_name)
