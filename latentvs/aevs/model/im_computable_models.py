
import numpy as np
import torch
import torchvision as tv
from torchvision.models.resnet import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import *

import aevs.model.im_computable as ic
from aevs.model.resnet_im_computable import *
import aevs.model.im_op_wrapper as iow
import copy
import math
from models.init import *
from utils.custom_typing import *

class AEIMComputable(nn.Module):
    def __init__(self):
        super(AEIMComputable, self).__init__()

    # @property
    # def encoder(self):
    #     raise NotImplementedError
    # @property
    # def decoder(self):
    #     raise NotImplementedError

    def activation_str_to_act_and_init(self, activation_str):
        from efficientnet_pytorch.utils import MemoryEfficientSwish
        act_fn, act_init = None, None
        if activation_str == 'relu':
            act_init = relu_init
            act_fn = nn.ReLU(inplace=True)
        if activation_str == 'swish':
            act_fn = MemoryEfficientSwish()
            act_init = swish_init
        elif activation_str == 'softplus':
            act_fn = nn.Softplus()
            act_init = softplus_init
        elif activation_str == 'tanh':
            act_fn = nn.Tanh()
            act_init = tanh_init
        
        elif activation_str.startswith('leaky_relu'):
            leak = float(activation_str[len('leaky_relu_'):])
            act_fn = nn.LeakyReLU(leak, inplace=True)
            act_init = make_leaky_relu_init(leak)
        return act_fn, act_init
    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return z, r
    def forward_with_interaction_matrix(self, x, L):
        z, Ln = self.encoder.forward_with_interaction_matrix(x, L)
        r = self.decoder(z)
        return z, r, Ln
    
    def forward_encode_with_interaction_matrix(self, x, L):
        z, Ln = self.encoder.forward_with_interaction_matrix(x, L)
        return z, Ln
    
    def forward_encode(self, x):
        z = self.encoder(x)
        return z
    
    def forward_decode(self, z):
        return self.decoder(z)
        
    def preprocess(self, x):
        return x / 255.0
    def unprocess(self, x):
        return x * 255.0
    
class ResNetAEIMComputable(AEIMComputable):

    def __init__(self, encoder_version: str, decoder_version: str, training_vae: bool, 
                scale_decoder_wrt_latent_dim: bool, latent_dim: UnsignedInt, num_output_feature_maps: UnsignedInt,
                input_image_size, pretrained=True, groups=16, bn_momentum=None, upsample='bilinear',
                activation='relu', use_wn_instead_of_bn=True, use_sn_instead_of_bn=True, replace_end_pooling=False,
                last_decoder_activation=nn.Identity(), stop_index=-1,
                width_factor=1.0, use_coord_conv=False, Z_estimate=0.6, camera_parameters=None):
        """

        Args:
            encoder_version : The version of the encoder to use. 18|34|50
            decoder_version : The version of the decoder to use. 18|34
            training_vae: Whether we are training a VAE. If true, the latent size is multiplied by 2, as each component of the distribution has a mean and a variance output.
            scale_decoder_wrt_latent_dim: Whether to scale the decoder feature map count on the latent space size 
            latent_dim: The latent dimension, the size of the vector output of the decoder
            num_output_feature_maps: Number of output feature maps for the decoder
            pretrained (bool, optional): Whether to use a pretrained encoder. Defaults to True.
            groups (int, optional): Number of groups in the last block of the encoder and the first block of the decoder. Used if replace_end_pooling=True. Defaults to 16.
            bn_momentum ([type], optional): The batch normalization momentum in the encoder. Used if use_wn_instead_of_bn is False. Defaults to None.
            upsample (str, optional): Upsampling mode in the decoder. bilinear|nearest, or other options supported by Pytorch upsampling. Defaults to 'bilinear'.
            activation (str, optional): Activation function to use. Defaults to 'relu'.
            use_wn_instead_of_bn (bool, optional): Replace Batch norm with Weight Normalization. Only performed in the encoder. Defaults to True.
            replace_end_pooling (bool, optional): Replace the end average pooling, with a learned projection. Defaults to False.
            last_decoder_activation ([type], optional): Activation applied to the output of the decoder. Defaults to nn.Identity().
            stop_index (int, optional): Whether to deactivate the last n blocks of the encoder and the first n blocks of the decoder.
                                        Deactivated blocks keep the residual shortcut. Defaults to -1.
            width_factor (float, optional): Rescaling factor on the width of the network.
            use_coord_conv (bool, optional): Whether to add CoordConv to the network, feeding to the network the location of each pixel of the feature map. Defaults to False.
            Z_estimate (float, optional): An average estimate of the depth of the images, used when use_coord_conv==True. Defaults to 0.6.
            camera_parameters ([type], optional): The camera intrinsics parameters: A dictionary with keys px, py (the ratio between focal length and pixel size in m) 
                                                and u0,v0 (the location of the principal point in pixels). Used when use_coord_conv==True. Defaults to None.

        """               
        super(ResNetAEIMComputable, self).__init__()
        from models.common import ResNetDecoder

        version_to_builder = {
            '18': resnet18,
            '34': resnet34,
            '50': resnet50
        }
        version_to_output_features = {
            '18': 512,
            '34': 512,
            '50': 2048
        }
        final_conv_output_size = input_image_size
        for _ in range(5):
            print(final_conv_output_size)
            final_conv_output_size = math.ceil(final_conv_output_size[0] / 2), math.ceil(final_conv_output_size[1] / 2)
        print(final_conv_output_size)
        convert = lambda x: iow.OpIMWrapper.from_op(x)
        self.input_image_size = input_image_size
        self.encoder = ResNetIMWrapper(version_to_builder[encoder_version](pretrained=pretrained), latent_dim)
        self.activation = activation
        act_fn, act_init = self.activation_str_to_act_and_init(self.activation)
        
        if activation != 'relu':
            self.encoder.set_activation(act_fn, act_init)
            self.encoder.op.relu = iow.OpIMWrapper.from_op(act_fn)

        self.latent_dim = latent_dim * 2 if training_vae else latent_dim 
        self._change_first_conv(act_init)
        self.encoder.op.fc = iow.OpIMWrapper.from_op(nn.Linear(512, self.latent_dim, bias=True))
        
        if replace_end_pooling:
            self._replace_end_pooling(version_to_output_features[encoder_version], final_conv_output_size, groups, bn_momentum, act_fn, act_init, use_wn_instead_of_bn)
        # Initialise decoder
        self.decoder = ResNetDecoder(decoder_version, scale_decoder_wrt_latent_dim, replace_end_pooling, False,
                                    latent_dim, num_output_feature_maps, upsample, groups, final_conv_output_size, last_decoder_activation, start_index=8 - stop_index if stop_index >= 0 else -1) # TODO: start_index only valid for ResNet-18


        assert not (use_wn_instead_of_bn and use_sn_instead_of_bn), 'cannot replace BN with both spectral norm and weight norm'
        if use_wn_instead_of_bn: # Use WN or initialize BN with given momentum
            self.encoder.replace_bn_with_wn()
        elif use_sn_instead_of_bn:
            self.encoder.replace_bn_with_sn()
        else:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.momentum = bn_momentum
        if width_factor != 1.0: # Rescale width of network
            self.change_width(width_factor, act_init)
        
        self.stop_index = stop_index
        if self.stop_index >= 0:
            self._deactivate_end_blocks(stop_index)

        if use_coord_conv:
            def with_coord_conv(coords_layer, conv):
                new_conv = self._conv_with_coord_weights(conv.op if isinstance(conv, ic.OpWithInteractionMatrixComputable) else conv, act_init)
                return nn.Sequential(coords_layer, new_conv)
            # assert Z_estimate is not None and camera_parameters is not None
            from operator import itemgetter

            encoder = self.encoder.op

            decoder = self.decoder
            addcoords_base = ic.AddCoords(224, 224)
            addcoords_levels = [addcoords_base] + [addcoords_base.convert_for_downsampled_image(f) for f in [2, 4, 8, 16]]
            addcoords_levels = [ic.AddCoordsWrapperConstantZ(ac, Z_estimate) for ac in addcoords_levels]
            encoder.conv1 = convert(with_coord_conv(addcoords_levels[0], encoder.conv1))
            encoder.layer1.op[0].op.conv1 = convert(with_coord_conv(addcoords_levels[2], encoder.layer1.op[0].op.conv1))
            encoder.layer2.op[0].op.conv1 = convert(with_coord_conv(addcoords_levels[2], encoder.layer2.op[0].op.conv1))
            encoder.layer3.op[0].op.conv1 = convert(with_coord_conv(addcoords_levels[3], encoder.layer3.op[0].op.conv1))
            encoder.layer4.op[0].op.conv1 = convert(with_coord_conv(addcoords_levels[4], encoder.layer4.op[0].op.conv1))

            decoder.blocks[1][0].conv1 = with_coord_conv(addcoords_levels[4], decoder.blocks[1][0].conv1)
            decoder.blocks[2][0].conv1 = with_coord_conv(addcoords_levels[3], decoder.blocks[2][0].conv1)
            decoder.blocks[3][0].conv1 = with_coord_conv(addcoords_levels[2], decoder.blocks[3][0].conv1)
            decoder.blocks[4][0].conv1 = with_coord_conv(addcoords_levels[1], decoder.blocks[4][0].conv1)

    def forward_encode_controlled_depth(self, x, stop_index):
        z = self.encoder.forward_controlled_depth(x, stop_index)
        return z

    

    def _get_convs(self):
        return self.decoder._get_convs() + self.encoder._get_convs()

    def change_width(self, width_factor, init):
        self.change_conv_widths(self, width_factor, init)
        nin = math.ceil(self.encoder.op.fc.op.in_features * width_factor)
        l = self.encoder.op.fc.op.out_features
        self.encoder.op.fc.op = nn.Linear(nin, l, self.encoder.op.fc.op.bias is not None)
        init(self.encoder.op.fc.op)
        self.decoder.fc1 = nn.Linear(l, math.ceil(self.decoder.fc1.out_features * width_factor), self.decoder.fc1.bias is not None)

        
    def change_conv_widths(self, module, width_factor, init):
        for n, sm in module.named_children():
            if isinstance(sm, nn.Conv2d):
                nin = math.ceil(sm.in_channels * width_factor)
                nout = math.ceil(sm.out_channels * width_factor)
                setattr(module, n, nn.Conv2d(nin, nout, sm.kernel_size, sm.stride, sm.padding, sm.dilation, min(sm.groups, nin, nout), sm.bias is not None))
                nc = getattr(module, n)
                init(nc)
            elif isinstance(sm, nn.ConvTranspose2d):
                nin = math.ceil(sm.in_channels * width_factor)
                nout = math.ceil(sm.out_channels * width_factor)
                setattr(module, n, nn.ConvTranspose2d(nin, nout, sm.kernel_size, sm.stride, sm.padding, sm.output_padding, min(sm.groups, nin, nout), sm.bias is not None, sm.dilation))
                nc = getattr(module, n)
                init(nc)
            elif isinstance(sm, nn.BatchNorm2d):
                nout = math.ceil(sm.num_features * width_factor)
                setattr(module, n, nn.BatchNorm2d(nout, sm.eps, sm.momentum, sm.affine, sm.track_running_stats))
                init(getattr(module, n))
            else:
                self.change_conv_widths(sm, width_factor, init)
    
    def get_encoder_convs(self):
        return [c.weight for c in self.encoder.modules() if isinstance(c, nn.Conv2d)]

    def _conv_with_coord_weights(self, conv, init_coord_filters):
        conv_weights = conv.weight
        wout, win, wh, ww = conv_weights.size()
        coord_filters = nn.Parameter(torch.zeros(wout, 2, wh, ww), requires_grad=True)
        init_coord_filters(coord_filters)
        
        new_params = torch.cat((conv_weights, coord_filters), dim=1)
        new_conv = nn.Conv2d(win + 2, wout, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(new_params)
            new_conv.bias.copy_(conv.bias)
        return new_conv
    
    def _change_first_conv(self, act_init):
        resnet = self.encoder.op
        resnet.conv1 = iow.OpIMWrapper.from_op(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
        act_init(resnet.conv1.op)
    def _replace_end_pooling(self, input_num_features, input_feature_map_size, groups, bn_momentum, act_fn, act_init, use_wn_instead_of_bn):
        l = [nn.Conv2d(input_num_features, 512, kernel_size=input_feature_map_size, padding=0, groups=groups)]
        act_init(l[0])
        if use_wn_instead_of_bn:
            l[0] = weight_norm(l[0])
        else:
            l.append(nn.BatchNorm2d(512, eps=1e-5, momentum=bn_momentum, affine=True, track_running_stats=True))
        l.append(act_fn)
        final_block = iow.OpIMWrapper.from_op(nn.Sequential(*l))
        self.encoder.op.avgpool = final_block
    
    def _deactivate_end_blocks(self, stop_index):
        encoder = self.encoder.op
        layers = [encoder.layer1.op, encoder.layer2.op, encoder.layer3.op, encoder.layer4.op]
        for i in range(len(layers)): # Iterate on "resolution" blocks (each block has dimensionality reduction)
            for j in range(2): # Iterate on residual blocks
                current_index = i * 2 + j
                if current_index > stop_index:
                    if layers[i][j].op.downsample is not None: # dimension reduction: we keep only the skip connection with the downsampling
                        layers[i][j] = layers[i][j].op.downsample
                    else: #Nothing to be done, deactivate block
                        layers[i][j] = iow.OpIMWrapper.from_op(nn.Identity())

class ImComputableVGG(nn.Module):
    def __init__(self, version, end_pooling_size, latent_size, activation):
        super(ImComputableVGG, self).__init__()
        version_to_model = {
            '11': vgg11,
            '11_bn': vgg11_bn,
            '13': vgg13,
            '13_bn': vgg13_bn,
        }

        self.encoder = version_to_model[version](pretrained=True)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((end_pooling_size, end_pooling_size))
        self.version = version
        for i in range(len(list(self.encoder.features.children()))):
            m = self.encoder.features[i]
            if isinstance(m, nn.ReLU):
                self.encoder.features[i] = activation
        self.latent_dim = latent_size
        self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder.features = iow.OpIMWrapper.from_op(self.encoder.features)
        self.encoder.avgpool = iow.OpIMWrapper.from_op(self.encoder.avgpool)
        print(self.encoder.classifier)
        self.encoder.classifier[2] = nn.Identity()
        self.encoder.classifier[5] = nn.Identity()
        
        self.encoder.classifier[-1] = nn.Linear(4096, latent_size)
        self.encoder.classifier = iow.OpIMWrapper.from_op(self.encoder.classifier)
        self.flattener = iow.OpIMWrapper.from_op(ic.Flatten())
    def forward(self, x):
        return self.encoder(x)

    def forward_with_interaction_matrix(self, x, L):
        x, L = self.encoder.features.forward_with_interaction_matrix(x, L)
        x, L = self.encoder.avgpool.forward_with_interaction_matrix(x, L)
        x, L = self.flattener.forward_with_interaction_matrix(x, L)
        x, L = self.encoder.classifier.forward_with_interaction_matrix(x, L)
        return x, L


        
    
    
