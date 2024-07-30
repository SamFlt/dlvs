import faulthandler
from gettext import translation
from matplotlib.pyplot import isinteractive
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
from enum import Enum
# matplotlib.use('agg')
from datetime import datetime
from operator import itemgetter
import pprint

import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from functools import partial


from torch.utils.tensorboard.writer import SummaryWriter
import tensorflow as tf
import tensorboard as tb

from metric_learning.model.models import ImageEncoder, PoseEmbedder, PoseEmbedderSIREN
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader



from datasets.AEDataset import FourDOFDataset, StoredLookAtAEDataset, StoredLookAtAEDatasetWithNearestNeighbors, StoredLookAtAEDatasetWithNearestNeighborsV2, TranslationDataset
from utils.callbacks import DispatchCallback, SaveModelCallback
from utils.metrics import MetricsList, Metric
from utils.losses import *
from utils.datasets import compute_reprojection_error, get_se3_dist_matrices, load_image_net_images_paths, load_image_woof_paths, pose_interaction_matrix
from torchvision import transforms
from utils.torchvision_transforms import *
from efficientnet_pytorch.model import EfficientNet

       
def train(config, make_model_name):
    faulthandler.enable()
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
    save_root = (root / '{}'.format(model_name)).expanduser()
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
    train_image_size = (h - border * 2, w - border * 2)

    # Create the training metrics, one for each loss
    make_metric = partial(lambda n: Metric(n, tb_writer))
    metrics = MetricsList([
        make_metric('Translation'),
        make_metric('Rotation'),
        make_metric('Loss'),
    ])

    task_weights = torch.tensor(np.array([0.0, 0.0]), dtype=torch.float, device=device, requires_grad=True)

    # Model creation
    image_encoder = ImageEncoder(latent_dim, encoder_version='34', width_factor=1.0, use_coord_conv=False).to(device)

    print(f'Image Encoder has {sum(p.numel() for p in image_encoder.parameters())} parameters')
    print('Training on', device)

    print('Creating training dataset...')
    distance_metric = 'se3'
    dataset = StoredLookAtAEDatasetWithNearestNeighbors(batches_per_epoch, batch_size, select_k_nn=3, distance_metric=distance_metric,
                                                         num_workers=num_workers, generator_parameters=config['generator_params'], 
                                                         look_at_parameters=config['dataset_params'], 
                                                         rz_max_deg=config['dataset_params']['rz_max_deg'])
    
    dl = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=dataset.get_worker_init_fn())
    print('Creating validation dataset...')
    val_dataset = StoredLookAtAEDatasetWithNearestNeighbors(batches_per_epoch // 2, batch_size,  3, distance_metric, num_workers,
                                                            config['val_generator_params'], config['dataset_params'], 
                                                            rz_max_deg=config['dataset_params']['rz_max_deg'])
    
    val_dl = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=val_dataset.get_worker_init_fn())

    optimizer = Adam([{'params': image_encoder.parameters()}, {'params': task_weights}],
                        lr=config['lr'], eps=1e-5, weight_decay=config['weight_decay'])
    lr_routine = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_plateau_params'])
    
    def mtl_loss_fn(task_weights, task_losses):
        sigma2s = torch.exp(-task_weights)
        weights = 1.0 / (2.0 * sigma2s)
        objectives_loss = (weights * task_losses).sum()
        reg_loss = torch.sum(task_weights)
        return objectives_loss + reg_loss


    def model_forward(Is, poses):
        zI = image_encoder(Is)
        return zI

    callbacks = DispatchCallback([SaveModelCallback('val_Loss', 'min', image_encoder, None,
                                    save_root / (model_name + '_image.pth'), None)])
    
    
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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

    loss_fn = nn.MSELoss()

    def do_iter(epoch, batch_idx, batch, is_train):
        b = 5
        should_do_log = not is_train and batch_idx == 0 and (epoch % (num_epochs // 10) == 0 or epoch == 1)

        if is_train:
            callbacks.on_batch_begin(batch_idx)

        Is, rs = batch[0].to(device), batch[1].to(device).float()
        Is = Is[:, :, b:-b, b:-b].contiguous()
        B, C, H, W = Is.size()
        count_perturbations = 2
        Is_perturb = generate_perturbation(Is, count_perturbations)
        Is = Is.unsqueeze(1)
        Is = torch.cat((Is, Is_perturb), dim=1)
        Is = Is.view(-1, C, H, W)
        Is = image_encoder.preprocess(Is)
        IsIn = Is
        if is_train:
            optimizer.zero_grad()
        
        zI_full = model_forward(IsIn, rs)
        zI_full = zI_full.view(B, count_perturbations + 1, 6)
        # zI_perturb = zI_full[:, 1:]
        zI = zI_full[:, 0]
        rs_repeat = rs.view(-1, 1, 6).repeat(1, count_perturbations + 1, 1)
        translation_loss = loss_fn(zI_full[..., :3], rs_repeat[..., :3])
        rotation_loss = loss_fn(zI_full[..., 3:], rs_repeat[..., 3:])
        

        

        # image_perturb_loss = image_invariance_loss(zI, zI_perturb)
        loss = mtl_loss_fn(task_weights, torch.cat([l.unsqueeze(0) for l in [translation_loss, rotation_loss]])) #+ image_perturb_loss
        metric_values = [translation_loss.item(), rotation_loss.item(), loss.item()]

        if is_train:
            metrics.new_values_train(metric_values)
            loss.backward()
            optimizer.step()
            callbacks.on_batch_end(batch_idx)
            f = (batch_idx + 1, len(dl), *metrics.get_train_values_as_display(), np.around(task_weights.detach().cpu().numpy(), decimals=3))
            print('{}/{}: translation: {}, rotation: {}, Loss: {}, var: {}'.format(*f), end='\r', sep='')
        else:
            metrics.new_values_val(metric_values)

        if should_do_log:
            # Add embedding to tensorboard
            Ics = IsIn[:, :, border:-border, border:-border].contiguous()
            Itb = image_encoder.unprocess(Ics)
            # tb_writer.add_embedding(zI_full, label_img=Itb / 255.0, global_step=epoch, tag=date_time)

            # Add first conv filters: Only for ResNet and EfficientNet for now
            c = None

            # Add input images to tensorboard
            IcsIn = IsIn[:, :, border:-border, border:-border].contiguous()
            Itb = image_encoder.unprocess(IcsIn)
            Itb = torch.clamp(Itb, 0, 255)
            tb_writer.add_image('originals', tv.utils.make_grid(Itb.type(torch.uint8)), epoch)
            

    results_dir = save_root / 'results'
    results_dir.mkdir(exist_ok=False)
    print('Starting training...')

    if config['epochs_train_decoder_only'] > 0:
        for parameter in image_encoder.encoder.parameters():
            parameter.requires_grad = False
    for epoch in range(1, num_epochs + 1):
        image_encoder.train()
        if epoch - 1 == config['epochs_train_decoder_only']:
            print('Activating encoder')
            for parameter in image_encoder.parameters():
                parameter.requires_grad = True
        print('Epoch: ', epoch, '/', num_epochs, sep='')
        for batch_idx, batch in enumerate(iter(dl)):
            do_iter(epoch, batch_idx, batch, True)

        print('Running eval')
        dataset.on_epoch_end(epoch)
        with torch.no_grad():
            image_encoder.eval()
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
    HollywoodTriangle = 'hollywood'
    Electronics = 'electronics'
    ImageNetCars = 'cars'
    ImageWoof = 'dogs'
    Pipe = 'pipe'

if __name__ == '__main__':
    import yaml
    print('Start!')

    import argparse
    parser = argparse.ArgumentParser(description='Train an autoencoder for visual servoing')
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
        scene_list = [str(data_root / 'scene_real_lower_res.jpg')]
        scene_list_val = [str(data_root / 'scene_real_lower_res.jpg')]

    if images_dataset == ImagesDataset.Electronics:
        scene_list = [str(data_root / 'electronics.png')]
        scene_list_val = [str(data_root / 'electronics.png')]
        
    elif images_dataset == ImagesDataset.ImageNetCars:
        assert args.train_take_count > 0, 'You should specify how many images to take'
        scene_list = load_image_net_images_paths(data_root, take_start_index=0,
                                                    max_take_count=args.train_take_count, 
                                                    num_classes=-1, 
                                                    class_list=['car'])
        scene_list = load_image_net_images_paths(data_root, take_start_index=args.train_take_count,
                                                    max_take_count=-1, # Take all the remaining images
                                                    num_classes=-1, 
                                                    class_list=['car'])
    elif images_dataset == ImagesDataset.ImageWoof:
        scene_list = load_image_woof_paths(data_root / 'imagewoof2', load_train=True)
        scene_list_val = load_image_woof_paths(data_root / 'imagewoof2', load_train=False)[:200]
    elif images_dataset == ImagesDataset.Pipe:
        scene_list = [str(data_root / 'pipe.png')]
        scene_list_val = [str(data_root / 'pipe.png')]


    print(f'{len(scene_list)} to generate training data, {len(scene_list_val)} for validation')
        
        

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
    
    latent_dim = 6
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
        'num_epochs': 50,
        'num_samples_per_epoch': 100 * 1000,
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

        
        'epochs_train_decoder_only': 0,
        'cuda_reproducibility': False
    }

    def make_name(_dt, config):
        return 'pose_regressor_6dof_resnet_34_100k_samples_180'
    train(config, make_name)
