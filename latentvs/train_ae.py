import faulthandler
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
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from models.common import DVSInteractionMatrix, ImageGradients
from aevs.model.efficient_net_im_computable import EfficientNetAE
from aevs.model.im_computable_models import ResNetAEIMComputable, ImComputableVGG
from aevs.model.im_computable import Flatten, L2Normalize, FlattenWrapper, SequentialWrapper
from aevs.model.im_computable import Sin

from aevs.losses import *


from datasets.AEDataset import StoredLookAtAEDataset
from utils.callbacks import DispatchCallback, SaveModelCallback
from utils.metrics import MetricsList, Metric
from utils.losses import static_weighted_loss
from utils.datasets import get_se3_dist_matrices, load_image_net_images_paths, load_image_woof_paths, pose_interaction_matrix


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
        make_metric('AE'),
        make_metric('ML DCT loss'),
        make_metric('Filter_loss'),
        make_metric('Pose loss'),
        make_metric('Pose IM loss'),
        make_metric('IM smoothing unsup loss'),
        make_metric('IM lower triangle'),
        make_metric('Loss'),
    ])

    # The losses to use during training: each item is a tuple: containing a boolean (whether to use the loss or not) and a float (the weight/contribution of the loss to the total loss)
    losses = [
        *itemgetter('ae', 'metric_learning_dct', 'filter_smoothing',
                    'pose_loss', 'pose_interaction_matrix_loss',
                    'interaction_matrix_smoothing_unsup',
                    'interaction_matrix_triangular_superior_loss')(config['losses'])
    ]
    use_losses = [l[0] for l in losses]
    use_ae_loss, use_ml_dct_loss, use_filter_loss,\
        use_pose_loss, use_pose_im_loss = use_losses

    # task weights: each loss is multiplied by the asosciated weight before summing for the final loss
    task_weights = [l[1] for l in losses]
    task_weights = torch.tensor(np.array(task_weights), dtype=torch.float, device=device, requires_grad=False)

    for i in range(len(use_losses)):
        if not use_losses[i]:
            task_weights.data[i] = 0.0
    # What operations are needed
    compute_interaction_matrices = use_pose_im_loss
    compute_rec = use_ae_loss

    # Model creation
    model = None
    if config['model_type'] == 'resnet':
        model = ResNetAEIMComputable(latent_dim=latent_dim, training_vae=False, **config['model_params'])
    elif config['model_type'] == 'efficientnet':
        model = EfficientNetAE(latent_dim)

    model = model.to(device)

    print(f'Encoder has {sum(p.numel() for p in model.encoder.parameters())} parameters')
    print('Training on', device)

    # Creating datasets
    dataset = StoredLookAtAEDataset(batches_per_epoch, batch_size, num_workers, config['generator_params'], look_at_parameters=config['dataset_params'], rz_max_deg=config['dataset_params']['rz_max_deg'], augmentation_factor=0.0)

    dl = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=dataset.get_worker_init_fn())

    val_dataset = StoredLookAtAEDataset(batches_per_epoch // 2, batch_size, num_workers, config['val_generator_params'], config['dataset_params'], rz_max_deg=config['dataset_params']['rz_max_deg'], augmentation_factor=0.0)
    val_dl = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=val_dataset.get_worker_init_fn())

    optimizer = Adam([{'params': model.parameters()}], lr=config['lr'], eps=1e-5, weight_decay=config['weight_decay'])

    lr_routine = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_plateau_params'])

    def model_forward(Is):
        '''
        Compute the required outputs for a set of image, given the model created above
        if compute_interaction_matrices: we perform forward propagation to find the interaction matrix associated to z
        if compute_rec: we use the decoder to compute the reconstruction
        returns:
            z: B x Z
            Irs : B x 1 x H x W the reconstructions
            Lz: B x Z x 6 the interaction matrix for z
            LIs: B x H x W x 6 the interaction matrix of the images (DVS)
        '''
        LIs = None
        if compute_interaction_matrices:
            with torch.no_grad():
                Idx, Idy = im_grads(Is.squeeze(1))
                LIs = Li_fn((Idx, Idy, Zinv))
        Is = Is[:, :, border:-border, border:-border].contiguous()
        Lz = None
        Irs  = None
        if not compute_interaction_matrices and not compute_rec:
            zI = model.forward_encode(Is)
        elif compute_interaction_matrices and not compute_rec:
            zI, Lz = model.forward_encode_with_interaction_matrix(Is, LIs)

        elif not compute_interaction_matrices and compute_rec:
            zI, Irs = model.forward(Is)
        else:
            zI, Irs, Lz = model.forward_with_interaction_matrix(Is, LIs)

        if len(zI.size()) > 2:
            if compute_interaction_matrices:
                zI, Lz = flattener.forward_with_interaction_matrix(zI, Lz)
            else:
                zI = flattener(zI)
        return zI, Irs, Lz, LIs

    # DVS Interaction matrix computer: required for the computation of LInn
    Li_fn = DVSInteractionMatrix(h, w,
                                generator_parameters['px'], generator_parameters['py'],
                                generator_parameters['u0'], generator_parameters['v0'],
                                border).to(device)
    im_grads = ImageGradients(border).to(device)
    # Use the average depth of the training set as input to the interaction matrix of DVS
    Zinv = 1 / Z

    callbacks = DispatchCallback([SaveModelCallback('val_Loss', 'min', model, optimizer,
                                    save_root / (model_name + '.pth'), save_root / (model_name + '_opt.pth'))])

    # Flatten latent space if required
    flattener = FlattenWrapper(Flatten()).to(device)

    # Build DCT-based reconstruction loss
    dct_ae_loss = None
    dct_loss_params = config['dct_loss_params']
    if dct_loss_params['type'] == 'zigzag':
        dct_ae_loss = make_ae_dct_loss_zigzag(train_image_size, dct_loss_params['zigzag_keep_count'], dct_loss_params['zigzag_inverse_weighting'], dct_loss_params['distance'], device)
    elif dct_loss_params['type'] == 'manhattan':
        dct_ae_loss = make_ae_dct_loss_manhattan(train_image_size, dct_loss_params['distance'], device)

    # AE loss dict
    ae_losses = {
        'mse': nn.MSELoss(),
        'bce': nn.BCELoss(),
        'dct': dct_ae_loss
    }
    ae_loss_fn = ae_losses[config['ae_loss_type']]

    # Filter regularization loss
    dct_filter_smoothing_loss_fn = make_dct_filter_smoothing(0.75, sizes=[3, 7], device=device)

    print(model)
    def add_noise_if_required(Is, training_denoising, noise_std):
        if training_denoising:
            out = Is + torch.randn(Is.size(), device=Is.device) * noise_std
        else:
            out = Is
        return out

    def do_iter(epoch, batch_idx, batch, is_train):
        should_do_log = not is_train and batch_idx == 0 and (epoch % (num_epochs // 10) == 0 or epoch == 1)

        if is_train:
            callbacks.on_batch_begin(batch_idx)

        Is, rs = batch[0].to(device), batch[1].to(device)
        Is = model.preprocess(Is)
        IsIn = add_noise_if_required(Is, False, 0.0) # Deactivated for now: allows training a denoising autoencoder
        if is_train:
            optimizer.zero_grad()

        z, Irs, Lz, _ = model_forward(IsIn)
        if Irs is not None:
            if Irs.size()[:2] != train_image_size: # If autoencoder outputs a smaller image, resize reconstruction to match input image
                Irs = F.interpolate(Irs, size=train_image_size, mode='bilinear')

        ae_loss, ml_dct_loss, filter_loss, pose_loss, pose_im_loss = [torch.tensor(0, device=device) for _ in range(len(use_losses))]

        # Compute losses

        if use_ae_loss:
            Ics = Is[:, :, border:-border, border:-border].contiguous()
            ae_loss = ae_loss_fn(Irs, Ics)

        if use_ml_dct_loss:
            Ics = Is[:, :, border:-border, border:-border].contiguous()
            ml_dct_loss = metric_learning_loss_dct_v2(z, Ics)

        if use_filter_loss:
            filter_loss = dct_filter_smoothing_loss_fn(model.get_encoder_convs())

        if use_pose_loss:
            t_dists, r_dists = get_se3_dist_matrices(rs.cpu().numpy())
            t_dists, r_dists = torch.from_numpy(t_dists).to(device), torch.from_numpy(r_dists).to(device)
            t_scale = 0.1 #10cm
            r_scale = np.radians(10.0) # 10 degree
            pose_loss = metric_learning_loss_log_ratio(z, t_dists, r_dists, t_scale, r_scale)

        if use_pose_im_loss:
            Lpbvs = torch.from_numpy(pose_interaction_matrix(rs.cpu().numpy())).to(device)
            assert Lpbvs.size() == Lz.size()
            Lpbvs_normed = Lpbvs / torch.norm(Lpbvs, dim=[-2, -1], keepdim=True)
            LInn_normed = Lz / torch.norm(Lz, dim=[-2, -1], keepdim=True)
            pose_im_loss = torch.mean(torch.norm(Lpbvs_normed - LInn_normed, p='fro', dim=(-2, -1)))



        loss_tensor = torch.cat([l.unsqueeze(0) for l in [ae_loss, ml_dct_loss, filter_loss, pose_loss, pose_im_loss]])
        assert torch.all(torch.isfinite(loss_tensor)), 'Some losses were NaN or Inf'
        loss = static_weighted_loss(task_weights, loss_tensor)
        metric_values = [l.squeeze(0).item() for l in loss_tensor] + [loss.item()]

        if is_train:
            metrics.new_values_train(metric_values)
            loss.backward()
            optimizer.step()
            callbacks.on_batch_end(batch_idx)

            print_tuple = (batch_idx + 1, len(dl), *metrics.get_train_values_as_display(), np.round(task_weights.detach().cpu().numpy(), decimals=6))
            print('{}/{}: AE: {}, ML DCT: {}, Filter loss: {}, Pose loss: {}, Pose IM loss: {}, Loss: {}, vars: {}'.format(*print_tuple), end='\r', sep='')
        else:
            metrics.new_values_val(metric_values)

        if should_do_log:
            # Add embedding to tensorboard
            Ics = IsIn[:, :, border:-border, border:-border].contiguous()
            Itb = model.unprocess(Ics)
            tb_writer.add_embedding(z, label_img=Itb / 255.0, global_step=epoch, tag=date_time)

            # Add first conv filters: Only for ResNet and EfficientNet for now
            c = None
            if config['model_type'] == 'resnet':
                c = model.encoder.op.conv1
            elif config['model_type'] == 'efficientnet':
                c = model.encoder.op._conv_stem
            w0 = c.op[1].op.weight if isinstance(c, SequentialWrapper) else c.op.weight
            w0 = w0.view(w0.size(0) * w0.size(1), 1, *w0.size()[2:])
            min_w0 = w0.view(w0.size(0), -1).min(dim=-1)[0].view(w0.size(0), 1, 1, 1)
            max_w0 = w0.view(w0.size(0), -1).max(dim=-1)[0].view(w0.size(0), 1, 1, 1)
            w0 = (w0 - min_w0) / (max_w0 - min_w0)
            tb_writer.add_image('first_filters', tv.utils.make_grid(w0), epoch)

            # Add input images to tensorboard
            IcsIn = IsIn[:, :, border:-border, border:-border].contiguous()
            Itb = model.unprocess(IcsIn)
            Itb = torch.clamp(Itb, 0, 255)
            tb_writer.add_image('originals', tv.utils.make_grid(Itb.type(torch.uint8)), epoch)
            if compute_rec: # Add reconstructions to tensorboard
                Irtb = model.unprocess(Irs)
                tb_writer.add_image('reconstructions', tv.utils.make_grid(Irtb.type(torch.uint8)), epoch)

            from mpl_toolkits.axes_grid1 import ImageGrid
            if compute_interaction_matrices: # Add plot of latent interaction matrices
                fig = plt.figure(figsize=(8, 64))
                bs = Lz.size(0)
                L_numpy = Lz.detach().cpu().numpy()
                vmin, vmax = L_numpy.min(), L_numpy.max()
                grid = ImageGrid(fig, 111, nrows_ncols=(bs, 1), axes_pad=(0.2, 0.2))
                for i, ax in enumerate(grid):
                    ax.matshow(L_numpy[i], vmin=vmin, vmax=vmax)
                plt.tight_layout()
                tb_writer.add_figure('Interaction matrices', fig, global_step=epoch)
                plt.clf()
            if use_pose_im_loss:
                fig = plt.figure(figsize=(8, 64))
                bs = Lz.size(0)
                L_numpy = Lz.detach().cpu().numpy()
                Lpbvs_numpy = Lpbvs.detach().cpu().numpy()
                vmin, vmax = L_numpy.min(), L_numpy.max()
                grid = ImageGrid(fig, 111, nrows_ncols=(bs, 2), axes_pad=(0.2, 0.2))
                for i, ax in enumerate(grid):
                    if i % 2 == 0:
                        ax.matshow(L_numpy[i // 2])
                    else:
                        ax.matshow(Lpbvs_numpy[i // 2])

                tb_writer.add_figure('Interaction matrices, Lz vs Lpbvs', fig, global_step=epoch)
                plt.clf()


    results_dir = save_root / 'results'
    results_dir.mkdir(exist_ok=False)
    print('Starting training...')

    if config['epochs_train_decoder_only'] > 0:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    for epoch in range(1, num_epochs + 1):
        model.train()
        if epoch - 1 == config['epochs_train_decoder_only']:
            print('Activating encoder')
            for parameter in model.encoder.parameters():
                parameter.requires_grad = True
        print('Epoch: ', epoch, '/', num_epochs, sep='')
        for batch_idx, batch in enumerate(iter(dl)):
            do_iter(epoch, batch_idx, batch, True)

        print('Running eval')
        dataset.on_epoch_end(epoch)
        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(iter(val_dl)):
                do_iter(epoch, batch_idx, batch, False)
        val_dataset.on_epoch_end(epoch)

        env_dict = metrics.values_as_dict()
        lr_routine.step(env_dict['val_Loss'])
        print(env_dict)
        metrics.write_to_tensorboard(epoch)
        callbacks.on_epoch_end(epoch, env_dict)
        metrics.reset()
        model.train()

    del model, dl, val_dl, optimizer


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
        save_dir = Path(project_conf['model_save_root_dir']) / 'aevs'

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
        scene_list = load_image_woof_paths(data_root / 'imagewoof2', 'train')
        scene_list_val = load_image_woof_paths(data_root / 'imagewoof2', 'val')
    elif images_dataset == ImagesDataset.Pipe:
        scene_list = [str(data_root / 'pipe.png')]
        scene_list_val = [str(data_root / 'pipe.png')]


    print(f'{len(scene_list)} to generate training data, {len(scene_list_val)} for validation')



    # Dataset parameters
    base_stds = np.asarray([0.1, 0.1, 10.0, 16.0])
    seed = 420
    Z = 0.5
    generator_parameters = {
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.35,
        'half_width_m': 0.5,
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
        'num_epochs': 50,
        'num_samples_per_epoch': 20 * 1000,
        'batch_size': 50,
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
            'look_from_half_ranges': [0.4, 0.4, 0.2],
            'rz_max_deg': 180.0
        },

        # Autoencoder parameters
        'model_type': 'resnet',
        'model_params': {
            'encoder_version': '18',
            'decoder_version': '18',
            'scale_decoder_wrt_latent_dim': False,
            'num_output_feature_maps': 1,
            'groups': 16,
            'bn_momentum': 0.1,
            'upsample':'nearest',
            'activation': 'relu',
            'use_wn_instead_of_bn': True,
            'replace_end_pooling': True,
            'stop_index': -1,
            'width_factor': 1.0,
            'last_decoder_activation': nn.Sigmoid(),
            'use_coord_conv': False,
            'Z_estimate': Z,
            'pretrained': True,
            'camera_parameters': dict({k: generator_parameters[k] for k in ('px', 'py', 'u0', 'v0')})
        },
        'losses': { # Legacy, only AE loss is used
            'pose_loss': (False, 0.0),
            'pose_interaction_matrix_loss': (False, 0.0),
            'ae': (True, 1.0),
            'metric_learning_dct': (False, 0.0),
            'filter_smoothing': (False, 0.0),
        },
        # Reconstruction loss params
        'ae_loss_type': 'dct',
        'dct_loss_params': {
            'type': 'manhattan',
            'distance': 'l1',
            'zigzag_keep_count': 256,
            'zigzag_inverse_weighting': False,
        },
        'epochs_train_decoder_only': 0,
        'cuda_reproducibility': False
    }
    from experiments_train import *
    runner = ExperimentTrainRunner(train, config)

    #runner(pose_exp, ae_loss_weight=1.0)
    #runner(ml_dct_exp)
    # runner(exp_multiple_model_seeds, base_name='resnet50_dct_loss', seed_count=3, start_seed=0)
    # runner(dataset_size_same_budget_exp, train_sizes=[200], seed_count=2, start_seed=1)
    # k = 1000
    # runner(width_exp, width_factors=[0.25, 0.5], seed_count=2, start_seed=3)

    def make_name(_dt, config):
        return f'pipe_model_v3_{config["latent_dim"]}'
    train(config, make_name)
    config['latent_dim'] = 128
    train(config, make_name)




