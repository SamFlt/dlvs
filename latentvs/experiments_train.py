import faulthandler
from matplotlib.pyplot import isinteractive
import matplotlib
import matplotlib.pyplot as plt
import cv2
# matplotlib.use('agg')
from datetime import datetime
from operator import itemgetter

import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from functools import partial

def get_seeds(overall_seed, seed_count):
    gnr = np.random.RandomState(overall_seed)
    data_seeds = gnr.choice(1000000, size=seed_count, replace=False)
    pt_seeds = gnr.choice(1000000, size=seed_count, replace=False)
    np_seeds =  gnr.choice(1000000, size=seed_count, replace=False)
    return data_seeds, pt_seeds, np_seeds

def make_net(starting_image_dims, block_count, layers_per_block, growth_rate, start_filters, compression_ratio,  latent_dim=None, max_filters=None, use_bn=False, pooling=lambda x, s: nn.AvgPool2d(s), global_pooling_size=3):
    assert block_count >= 1
    def make_conv(in_filters, out_filters, ks=3, padding=1, use_bn=False):
        c = [nn.Conv2d(in_filters, out_filters, kernel_size=ks, padding=padding)]
        if use_bn:
            c.append(nn.BatchNorm2d(out_filters))
        return c
    def make_block(in_filters, out_filters, pool_first):
        c = []
        if pool_first:
            c = c + [pooling(in_filters, 2)]
        c = c + make_conv(in_filters, out_filters, use_bn=use_bn)
        for i in range(1, layers_per_block):
            c = c + make_conv(out_filters, out_filters, use_bn=use_bn)
        return c
    p = make_block(1, start_filters, False)
    filters = start_filters
    for i in range(1, block_count):
        new_filters = growth_rate * filters
        if max_filters is not None:
            new_filters = min(max_filters, new_filters)
        p = p + make_block(filters, new_filters, True)
        filters = new_filters
    im_size = starting_image_dims[0] // ((block_count -1) * 2) *  starting_image_dims[1] // ((block_count -1) * 2)
    starting_im_size = np.prod(starting_image_dims)
    if compression_ratio is not None and latent_dim is None:
        required_bottleneck_size = compression_ratio * starting_im_size
        final_filters = filters
        bottleneck_size = final_filters * (im_size ** 2)
        while bottleneck_size > required_bottleneck_size and final_filters > 1:
            final_filters -= 1
            bottleneck_size = final_filters * (im_size ** 2)
    else:
        final_filters = latent_dim
        bottleneck_size = final_filters * (im_size ** 2)
    p = p + make_conv(filters, final_filters, ks=1, padding=0, use_bn=False)
    if global_pooling_size is not None and global_pooling_size > 0:
        p.append(pooling(final_filters, global_pooling_size))
    return nn.Sequential(*p), bottleneck_size / starting_im_size

def exp_activations(train, config):
    def make_name(date_time, config):
        return 'resnet_{}_{}'.format(config['latent_dim'], config['model_params']['activation'])
    for act in ['softplus', 'relu', 'leaky_relu_0.1', 'tanh']:
        c = config.copy()
        print('Training network with activation = {}'.format(act))
        c['model_params']['activation'] = act
        train(c, make_name)
def exp_activations_relu_leaky_linear(train, config):
    def make_name(date_time, config):
        return '{}_{}_{}'.format(date_time, config['latent_dim'], config['model_params']['activation_str'])
    for act in ['linear', 'leaky_relu_0.01', 'leaky_relu_0.1', 'leaky_relu_0.2', 'leaky_relu_0.5', 'leaky_relu_0.9']:
        c = config.copy()
        c['model_params']['activation_str'] = act
        train(c, make_name)
def exp_latent_size(train, config):
    dims = (config['generator_params']['h'], config['generator_params']['w'])
    if config['model_type'] == 'basic_ae':
        def make_name(date_time, config):
            return '{}_{}_{}'.format(date_time, config['latent_dim'], config['model_params']['activation_str'])
    else:
        def make_name(_date_time, config):
            return 'resnet_latent_{}_{}'.format(config['latent_dim'], config['model_params']['activation'])
    for size in [6, 16, 32, 64, 128, 256]:
        c = config.copy()
        c['latent_dim'] = size
        if c['model_type'] == 'basic_ae':
            base_net = make_net(dims, 6, 2, 2, 16, None, latent_dim=size, max_filters=512, global_pooling_size=7)[0]
            # base_net.add_module('end_pooling', nn.AvgPool2d(7))
            c['model_params']['downpath'] = base_net
        train(c, make_name)

def exp_linear_model_learning_rate(train, config):
    def make_name(date_time, config):
        return '{}_{}_lr_{}'.format(date_time, config['model_params']['activation_str'], config['lr'])
    for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
        c = config.copy()
        c['num_epochs'] = 5
        c['model_params']['activation_str'] = 'linear'
        c['lr'] = lr
        train(c, make_name)

def exp_wd(train, config):
    def make_name(_date_time, config):
        return 'resnet_{}_{}_wd_{}'.format(config['latent_dim'], config['model_params']['activation'], config['weight_decay'])
    for wd in [1e-6, 0, 1e-5, 1e-4, 1e-3, 1e-2]:
        c = config.copy()
        c['weight_decay'] = wd
        train(c, make_name)

def exp_mlp_depth(train, config):
    k = 2048
    min_k = 256
    def make_name(date_time, config):
        return '{}_{}_{}_hlayers_{}_min_k_{}'.format(date_time, config['model_params']['latent_dim'], k, len(config['model_params']['encoder_units']), min_k)
    for i in range(2, 10):
        c = config.copy()
        c['model_params']['encoder_units'] = [max(k // (2 ** j), min_k) for j in range(i)]
        c['model_params']['decoder_units'] = [max(k // (2 ** j), min_k) for j in range(i-1, -1, -1)]
        print(c['model_params'])
        train(c, make_name)


def exp_cnn_depth(train, config):
    pooling = nn.AvgPool2d
    use_bn = False
    layers_per_block = 2
    growth_rate = 2
    start_filters = 16
    compression_ratio = 0.2
    dims = (config['generator_params']['h'], config['generator_params']['w'])
    for i in range(3, 8):
        c = config.copy()
        model, ratio = make_net(dims, i, layers_per_block, growth_rate, start_filters, compression_ratio, None, use_bn=use_bn, pooling=pooling)
        c['model_params']['downpath'] = model
        def make_name(date_time, config):
            return 'cnn_{}_blocks_bn_{}_compression_ratio_{}_sf_{}_growth_{}'.format(i, use_bn, ratio, start_filters, growth_rate)
        train(c, make_name)

def exp_cnn_depth_lpb(train, config):
    pooling = lambda x, s: nn.AvgPool2d(s)
    use_bn = False
    number_of_blocks = 6
    growth_rate = 2
    start_filters = 16
    dims = (config['generator_params']['h'], config['generator_params']['w'])
    for i in range(1, 5):
        c = config.copy()
        model, _ = make_net(dims, number_of_blocks, i, growth_rate, start_filters, compression_ratio=None, latent_dim=c['latent_dim'], use_bn=use_bn, pooling=pooling)
        c['model_params']['downpath'] = model
        def make_name(date_time, config):
            return 'cnn_{}_blocks_lpb_{}_sf_{}_growth_{}'.format(number_of_blocks, i, start_filters, growth_rate)
        train(c, make_name)

def exp_cnn_contractive_vs_non_contractive(train, config):
    dims = (config['generator_params']['h'], config['generator_params']['w'])
    for use_contractive, weight in [(True, 1e-4), (True, 1e-3), (True, 1.0), (True, 0.01), (True, 0.1), (True, 10.0), (False, 0.0)]:
        c = config.copy()
        c['losses']['contractive_ae'] = (use_contractive, weight)
        print(c['losses'])
        print(c['loss_static_weights'])
        def make_name(date_time, config):
            return 'cnn_contractive_{}_weight_{}'.format(use_contractive, weight)

        c['model_params']['downpath'] = make_net(dims, 6, 2, 2, 16, None, 128, max_filters=512, global_pooling_size=7)[0]
        # c['model_params']['downpath'].add_module('end_pooling', nn.AvgPool2d(3))
        c['model_params']['activation_str'] = 'swish'
        train(c, make_name)

def exp_random_net_vs_trained(train, config):
    count = 5
    rs = np.random.RandomState(seed=23)
    np_seeds = rs.choice(10000, size=count, replace=False)
    torch_seeds = rs.choice(10000, size=count, replace=False)
    for i in range(count):
        def make_name(date_time, config):
            return 'random_net_{}'.format(i)
        c = config.copy()
        c['torch_seed'] = torch_seeds[i]
        c['np_seed'] = np_seeds[i]
        c['num_epochs'] = 0
        c['lr'] = 0 # make sure it can't train
        train(c, make_name)
    for i in range(count):
        def make_name_trained(date_time, config):
            return 'trained_net_{}'.format(i)
        c = config.copy()
        c['torch_seed'] = torch_seeds[i]
        c['np_seed'] = np_seeds[i]
        train(c, make_name_trained)
    
def exp_pooling(train, config):
    avg_pool = lambda xi, s: nn.AvgPool2d(s)
    max_pool = lambda xi, s: nn.MaxPool2d(s)
    depth_wise_conv = lambda xi, s: nn.Conv2d(xi, xi, kernel_size=s, stride=s, groups=xi)
    conv = lambda xi, s: nn.Conv2d(xi, xi, kernel_size=s, stride=s)
    dims = itemgetter('h', 'w')(config['generator_params'])
    # (conv, 'conv'), (depth_wise_conv, 'depthwise_conv'),
    for pooling_op, name in [(avg_pool, 'avg'), (max_pool, 'max')]:
        net = make_net(dims, 6, 2, 2, 16, None, 64, max_filters=512, use_bn=False, pooling=pooling_op, global_pooling_size=7)[0]
        c = config.copy()
        c['model_params']['downpath'] = net
        def make_name(date_time, config):
            return 'net_{}'.format(name)
        train(c, make_name)

def exp_dataset_size(train, config):
    target_iters = 20000
    def make_name(date, config):
        return 'resnet_{}_train_{}k'.format(config['latent_dim'], config['num_samples_per_epoch'] // 1000)
    for k in [5, 10, 20, 50, 100]:
        epochs = target_iters // (k * 1000 // config['batch_size'])
        c = config.copy()
        c['num_epochs'] = epochs
        c['num_samples_per_epoch'] = k * 1000
        train(c, make_name)

def exp_wn_vs_bn(train, config):
    def make_name(date_time, config):
        return '{}_{}_wn_{}'.format(config['model_type'], config['latent_dim'], config['model_params']['use_wn_instead_of_bn'])
    for use_wn in [False, True]:
        c = config.copy()
        c['model_params']['use_wn_instead_of_bn'] = use_wn
        train(c, make_name)

def exp_resnet_depth(train, config):
    def make_name(date_time, config):
        return '{}_{}_depthv2_{}'.format(config['model_type'], config['latent_dim'], config['model_params']['stop_index'])
    for index in range(8):
        c = config.copy()
        c['model_params']['stop_index'] = index
        train(c, make_name)

def exp_jacobian_reg(train, config):
    def make_name(date_time, config):
        return '{}_{}_jacobian_{}'.format(config['model_type'], config['latent_dim'], config['losses']['smoothing'][1])
    
    for weight in [1e-6, 0.0, 1e-5, 1e-4, 1e-3]:
        c = config.copy()
        c['lr'] = 1e-4
        c['model_params']['activation'] = 'softplus'
        c['batch_size'] = c['batch_size'] // 2
        c['num_epochs'] = c['num_epochs'] // 2
        c['losses']['smoothing'] = (weight > 0.0, weight)
        train(c, make_name)

def exp_pose_im_loss(train, config):
    def make_name(date_time, config):
        return '{}_{}_pose_im_loss_{}'.format(config['model_type'], config['latent_dim'], config['losses']['pose_interaction_matrix_loss'][1])
    
    for weight in [1e-3, 1e-4, 0.0]:
        c = config.copy()
        c['latent_dim'] = 6
        c['lr'] = 1e-4
        # c['model_params']['activation'] = 'softplus'
        c['batch_size'] = c['batch_size'] // 2
        c['num_epochs'] = c['num_epochs'] // 2
        c['losses']['pose_interaction_matrix_loss'] = (weight > 0.0, weight)
        train(c, make_name)

def exp_filter_smoothing(train, config):
    def make_name(date_time, config):
        return '{}_{}_smoothing_{}'.format(config['model_type'], config['latent_dim'], config['losses']['filter_smoothing'][1])
    
    for weight in [1.0, 5.0, 10.0, 1e-1, 1e-3, 1e-2, 0.0]:
        c = config.copy()
        c['losses']['filter_smoothing'] = (weight > 0.0, weight)
        train(c, make_name)

def exp_width(train, config):
    def make_name(date_time, config):
        return '{}_{}_width_{}'.format(config['model_type'], config['latent_dim'], config['model_params']['width_factor'])
    
    for factor in [0.125, 0.25, 0.5, 0.75, 1.0]:
        c = config.copy()
        c['model_params']['width_factor'] = factor
        train(c, make_name)

def exp_multiple_seeds(train, config, base_name, seed_count):
    data_seeds, pt_seeds, np_seeds = get_seeds(439, seed_count)
    for i in range(seed_count):
        def make_name(_dt, config):
            return '{}_seed_{}'.format(base_name, i)
        c = config.copy()
        c['generator_params']['base_seed'] = data_seeds[i]
        c['np_seed'] = np_seeds[i]
        c['torch_seed'] = pt_seeds[i]
        train(c, make_name)

def exp_multiple_model_seeds(train, config, base_name, seed_count, start_seed=0):
    _, pt_seeds, _ = get_seeds(439, seed_count)
    for i in range(seed_count):
        def make_name(_dt, config):
            return '{}_seed_{}'.format(base_name, start_seed + i)
        c = config.copy()
        c['torch_seed'] = pt_seeds[start_seed + i]
        train(c, make_name)

def exp_multiple_data_seeds(train, config, base_name, seed_count):
    data_seeds, _, np_seeds = get_seeds(439, seed_count)
    for i in range(seed_count):
        def make_name(_dt, config):
            return '{}_seed_{}'.format(base_name, i)
        c = config.copy()
        c['generator_params']['base_seed'] = data_seeds[i]
        c['np_seed'] = np_seeds[i]
        train(c, make_name)

def exp_keep_count(train, config):
    for kc in [64, 128, 256]:
        def make_name(_dt, config):
            return 'resnet_keep_count_{}'.format(kc)
        c = config.copy()
        c['ae_dct_loss_keep_count'] = kc
        train(c, make_name)

def exp_ratio_and_inverse_weighting(train, config):
    for kc, inverse in [(50, False), (50, True)]:
        def make_name(_dt, config):
            return 'resnet_keep_count_{}_weighting_{}'.format(kc, inverse)
        c = config.copy()
        c['ae_dct_loss_keep_count'] = kc
        c['ae_dct_loss_inverse_weighting'] = inverse
        
        train(c, make_name)

def exp_lr_dct_loss(train, config, lrs):
    def make_name(_dt, config):
        return 'resnet_dct_smaller_no_ortho_lr_{}'.format(config['lr'])
    for lr in lrs:
        c = config.copy()
        c['lr'] = lr
        train(c, make_name)

def multiscene_exp(train, config, data_root, scene_counts):
    from utils.datasets import load_image_net_images_paths
    for scene_count in scene_counts:
        c = config.copy()
        def make_name(_dt, config):
            return 'resnet_multiscene_dct_{}'.format(scene_count)
        scene_list = load_image_net_images_paths(data_root / 'imagenet_images', 0, scene_count, class_list=['car'])
        c['generator_params']['image_paths'] = scene_list
        c['val_generator_params']['image_paths'] = scene_list
        c['dataset_params']['look_at_half_ranges'] =  [0.1, 0.1]
        c['num_epochs'] = 50
        c['num_samples_per_epoch'] = 100 * 1000
        train(c, make_name)

def pose_exp(train, config, ae_loss_weight):
    c = config.copy()
    def make_name(_dt, config):
        return 'resnet_pose_{}'.format(c['latent_dim'])
    if ae_loss_weight > 0:
        c['losses']['ae'] = (True, ae_loss_weight)
    else:
        c['losses']['ae'] = (False, 0.0)
    c['losses']['pose_loss'] = (True, 1.0)
    train(c, make_name)

def ml_dct_exp(train, config):
    c = config.copy()
    def make_name(_dt, config):
        return 'resnet_metric_learning_dct_{}'.format(c['latent_dim'])
    c['losses']['metric_learning_dct'] = (True, 1.0)
    c['losses']['ae'] = (False, 0.0)
    train(c, make_name)

def dataset_size_same_epochs_exp(train, config, train_sizes, seed_count):
    data_seeds, _, np_seeds = get_seeds(439, seed_count)

    for size in train_sizes:
        for i in range(seed_count):
            c = config.copy()
            def make_name(_dt, config):
                return 'dataset_size_{}_seed_{}'.format(config['num_samples_per_epoch'], i)
            c['num_samples_per_epoch'] = size
            c['generator_params']['base_seed'] = data_seeds[i]
            c['np_seed'] = np_seeds[i]
            train(c, make_name)

def dataset_size_same_budget_exp(train, config, train_sizes, seed_count, start_seed=0):
    data_seeds, _, np_seeds = get_seeds(439, seed_count)
    iters_per_epoch = config['num_samples_per_epoch'] // config['batch_size']
    base_iters = iters_per_epoch * config['num_epochs']
    for size in train_sizes:
        for i in range(seed_count):
            c = config.copy()
            def make_name(_dt, config):
                return 'dataset_size_{}_same_budget_lr_{}_v2_seed_{}'.format(config['num_samples_per_epoch'], config['lr'], start_seed + i)
            c['num_samples_per_epoch'] = size
            c_num_epochs = base_iters // (c['num_samples_per_epoch'] // c['batch_size'])
            c['num_epochs'] = c_num_epochs
            c['lr_plateau_params']['patience'] = max(2, c_num_epochs // config['num_epochs'])
            c['generator_params']['base_seed'] = data_seeds[start_seed + i]
            # c['torch_seed'] = _pt_seeds[i]
            c['np_seed'] = np_seeds[start_seed + i]
            # c['model_params']['pretrained'] = False
            train(c, make_name)

def width_exp(train, config, width_factors, seed_count, start_seed=0):
    _, pt_seeds, np_seeds = get_seeds(439, seed_count)

    for width in width_factors:
        for i in range(seed_count):
            c = config.copy()
            def make_name(_dt, config):
                return 'resnet_width_{}_seed_{}'.format(config['model_params']['width_factor'], start_seed + i)
            c['model_params']['width_factor'] = width
            c['torch_seed'] = pt_seeds[start_seed + i]
            c['np_seed'] = np_seeds[start_seed + i]
            train(c, make_name)


class ExperimentTrainRunner():
    '''
    Wrapper for launching experiments
    An experiment is a function that has the signature (train_fn, config, **kargs) -> None
    with train_fn the function training a network, config is the dictionary of training parameters
    and kargs are the experiment-specific parameters

    The experiments are responsible for calling the train function (can be multiple times) with the correct inputs.
    '''
    def __init__(self, train_fn, config):
        self.train_fn = train_fn
        self.config = config
    def __call__(self, exp_fn, **kargs):
        exp_fn(self.train_fn, self.config.copy(), **kargs)
