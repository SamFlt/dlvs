import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils.custom_typing import UnsignedInt
from .CoordConv import AddCoords
from efficientnet_pytorch.utils import *
from efficientnet_pytorch.model import *

 
class EfficientNetDecoder(nn.Module):
    def __init__(self, bottleneck_size, out_channels, blocks_args, global_params, start_depthconv_groups, upsampling):
        super(EfficientNetDecoder, self).__init__()
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.start_depthconv_groups = start_depthconv_groups
        self.upsampling = upsampling
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        
        in_channels = round_filters(32, self._global_params)  # number of output channels
        self.last_bn = nn.BatchNorm2d(num_features=in_channels, momentum=bn_mom, eps=bn_eps)
        self.last_conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        # Build blocks
        self._blocks = nn.ModuleList([])
        self.upsamples = []
        image_size = [7, 7]
        for i, block_args in enumerate(reversed(self._blocks_args)):
            upsample = block_args.stride[0] > 1
            self.upsamples.append(upsample)
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.output_filters, self._global_params),
                output_filters=round_filters(block_args.input_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
                stride=1
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if upsample:
                image_size = image_size * 2
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
                self.upsamples.append(False)
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        for block in self._blocks:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    out_filters, _, h, w = m.weight.size()
                    fan_out = int(out_filters * h * w)
                    torch.nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0 / fan_out))
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    r = 1.0 / np.sqrt(m.weight.size()[1])
                    nn.init.uniform_(m.weight, -r, r)
        # Head
        out_channels = self._blocks_args[-1].output_filters  # output of final block
        in_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=[7, 7])
        self.first_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.first_bn = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._fc = nn.Linear(bottleneck_size, round_filters(1280, self._global_params))
        self._swish = MemoryEfficientSwish()
        if self.start_depthconv_groups > 0:
            self.start_dconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=7, groups=self.start_depthconv_groups, padding=0, bias=True)
    def forward(self, x):
        x = self._fc(x)
        x = self._swish(x)
        xsize = x.size()[-1]
        if self.start_depthconv_groups <= 0:
            x = F.interpolate(x.view(-1, xsize, 1, 1), size=[7,7])
        else:
            x = self.start_dconv(x.view(-1, xsize, 1, 1))
        x = self._swish(self.first_bn(self.first_conv(x)))
        for block, upsample in zip(self._blocks, self.upsamples):
            x = block(x)
            if upsample:
                x = F.interpolate(x, scale_factor=2, mode=self.upsampling)
        x = F.interpolate(x, scale_factor=2, mode=self.upsampling)
        x = self.last_conv(self.last_bn(x))
        return x


class ResNet18DAE(nn.Module):

    def __init__(self, latent_dim, use_coordconv, groups):
        super(ResNet18DAE, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.use_coordconv = use_coordconv
        self.coordconv = None
        self.latent_dim = latent_dim
        if not self.use_coordconv:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.coordconv = AddCoords(with_r=False)
        self.encoder.fc = nn.Linear(512, latent_dim * 2, bias=True)
        final_block = nn.Sequential(
            nn.Conv2d(512, latent_dim * 2, kernel_size=(7,7), padding=0, groups=groups),
            nn.BatchNorm2d(latent_dim * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.encoder.avgpool =  final_block
        self.decoder_content = ResNetDecoder(latent_dim, 1, 'bilinear', groups // 2)
        self.decoder_perturbations = ResNetDecoder(latent_dim, 1, 'bilinear', groups // 2)

        print(self.encoder)
        print('=' * 10)
        print(self.decoder_perturbations)
        
    def forward(self, x):
        if self.use_coordconv:
            x = self.coordconv(x)
        z_full = self.encoder(x)
        z_perturbations = z_full[:, :self.latent_dim]
        z_content = z_full[:, self.latent_dim:self.latent_dim * 2]

        r_content = self.decoder_content(z_content)
        r_perturbations = self.decoder_perturbations(z_perturbations)
        r = r_content + r_perturbations

        return z_content, z_perturbations, r_content, r_perturbations, r



class ResNetDecoder(nn.Module):
    def __init__(self, version: str, scale_on_latent_dim: bool, use_custom_first_block: bool, use_skip_inputs: bool, latent_dim: UnsignedInt,
                output_channels: UnsignedInt, upsample_mode, groups: UnsignedInt, reshape_size=(7,7), last_activation=nn.Identity(), start_index=-1):
        '''
        A ResNet-like decoder, mimicking the architecture of a ResNet encoder (same number of blocks), but instead of downsampling upsamples the features maps.
        version: version of the decoder to use: 18 to mimica Resnet-18 encoder, 34 to mimic a ResNet-34.
            The decoder uses the same number of blocks as the chosen version encoder, and the same block structure
        scale_on_latent_dim: Whether to choose a number of feature maps that is scaled on the size of the latent dimension
            for example if scale_on_latent_dim=True and latent_dim = 32, the 1st block will have 32 feature maps, the 2nd will have 16 and so on.
        use_custom_first_block: replace the basic upsampling operation (inverse of average pooling) by a learned upsampling ConvTranspose2d -> BN -> ReLU.
            If false all the feature pixels will have the same value after the resizing 
            (meaning that whether the encoder outputs positional information is kind of irrelevant, as the decoder as to relearn it)
        Use_skip_inputs: The forward function will accept as inputs the intermediate outputs of the encoder stages, transforming this decoder into a U-Net-like decoder.
        latent_dim: the dimension of the latent space (input)
        output_channels: The number of output feature maps: 1 for grayscale, 3 for RGB images etc.
        upsample_mode: the upsampling mode, bilinear or nearest
        groups: The number of groups for the first block, if use_custom_first_block==True.
        reshape_size: What size the feature map should be after the first block, when the latent space is converted back in a spatial representation
        last_activation: What activation to apply to the output image: use Identity when using non range-restricted inputs, Sigmoid when values are in [0, 1]
        start_index: if > 0, deactivates the first start_index blocks. For deactivated blocks, only the residual shortcut is kept.

        '''
        super(ResNetDecoder, self).__init__()
        self.upsample_mode = upsample_mode
        self.latent_dim = latent_dim
        self.scale_on_latent_dim = scale_on_latent_dim
        self.output_channels = output_channels
        self.upsampling = nn.Upsample(scale_factor=2, mode=self.upsample_mode)
        start_n_features = self.latent_dim if scale_on_latent_dim else 512
        self.fc1 = nn.Linear(self.latent_dim, start_n_features)
        self.relu = nn.ReLU(inplace=True)
        self.use_skip_inputs = use_skip_inputs
        # First block
        if use_custom_first_block:
            first_block = nn.Sequential(
                nn.ConvTranspose2d(start_n_features, start_n_features, kernel_size=reshape_size, padding=0, groups=groups),
                nn.BatchNorm2d(start_n_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )
        else:
            first_block = nn.Upsample(scale_factor=reshape_size[0], mode=self.upsample_mode)

        deconv = first_block
        # Number of blocks per stage in each version
        block_counts = {
            '18': [2, 2, 2, 2],
            '34': [3, 6, 4, 3]
        }[version]

        self.blocks = nn.ModuleList([
            deconv, # 7x7
            self.stage(start_n_features, start_n_features // 2, block_counts[0]), #14x14
            self.stage(start_n_features // 2, start_n_features // 4, block_counts[1]), # 28x28
            self.stage(start_n_features // 4, start_n_features // 8, block_counts[2]), # 56x56
            self.stage(start_n_features // 8, start_n_features // 8, block_counts[3]), # 112x112
        ])

        # Deactivate first n blocks
        if start_index > 0:
            for i in range(0, start_index):
                stage_index = 1 + i // 2
                if self.blocks[stage_index][i % 2].upsample is not None:
                    b = self.blocks[stage_index][i % 2]
                    if b.identity_proj is None:
                        self.blocks[stage_index][i % 2] = b.upsample
                    else:
                        self.blocks[stage_index][i % 2] = nn.Sequential(b.upsample, b.identity_proj)
                else:
                    self.blocks[stage_index][i % 2] = nn.Identity()
                print('Deactivating stage {}, block {}'.format(stage_index, i%2))
        
        # Create projections when using skip inputs
        if self.use_skip_inputs:
            ins = [256, 128, 64, 64]
            outs = [start_n_features // 2, start_n_features // 4, start_n_features // 8, start_n_features // 8]
            self.projs = nn.ModuleList()
            for ip, op in zip(ins, outs):
                conv = nn.Conv2d(ip, op, 1, bias=False)
                nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
                bn = nn.BatchNorm2d(op, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                relu = nn.ReLU(inplace=True)
                self.projs.append(nn.Sequential(conv, bn, relu))
        #Last convolution, activation
        self.last_conv = nn.Conv2d(start_n_features // 8, output_channels, kernel_size=7, padding=3)
        self.last_activation = last_activation
        self.activation_type = nn.ReLU

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m != self.last_conv:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight, gain=1.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        if self.use_skip_inputs:
            y = self.fc1(x[-1])
            y = self.relu(y)
            y = y.view(-1, y.size()[1], 1, 1)
            y = self.blocks[0](y)
            for b, xk, proj in zip(self.blocks[1:], reversed(x[:-1]), self.projs):
                y = b(y)
                y = y + proj(xk)
            y = self.upsampling(y)
            y = self.last_conv(y)
            y = self.last_activation(y)
            return y
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = x.view(-1, x.size()[1], 1, 1)
            for b in self.blocks:
                x = b(x)
            x = self.upsampling(x)
            x = self.last_conv(x)
            x = self.last_activation(x)
            return x
    
    def stage(self, inplanes, planes, num_blocks):
        '''
        Create a series of blocks for a single spatial resolution
        Start by creating a block with upsampling then creates blocks which preserve the spatial resolution
        '''
        layers = []
        layers.append(ResNetDecoderBasicBlock(inplanes, planes, upsample=self.upsampling))
        for n in range(1, num_blocks):
            layers.append(ResNetDecoderBasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def set_activation(self, activation, init_fn):
        '''
        Replace the activations of the network by another, and reinitializes the network weights to match the activation
        TODO: verify that this does not replace the last activation and the last conv layer weights, as they are separate from the other modules
        '''
        def recurse_fn(m):
            for mod_str, mod in m.named_children():
                if type(mod) == self.activation_type:
                    setattr(m, mod_str, activation)
                else:
                    init_fn(mod)
                    recurse_fn(mod)
        for k, v in self.named_children():
            recurse_fn(v)
        self.activation_type = type(activation)
    def _get_convs(self):
        '''returns the convolutional layers of the network'''
        convs = []
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                convs.append(module)
        return convs


class ResNet18Decoder(nn.Module):
    def __init__(self, scale_on_latent_dim, use_custom_first_block, use_skip_inputs, latent_dim,
                output_channels, upsample_mode, groups, reshape_size=(7,7), last_activation=nn.Identity(), start_index=-1):
        super(ResNet18Decoder, self).__init__()
        self.upsample_mode = upsample_mode
        self.latent_dim = latent_dim
        self.scale_on_latent_dim = scale_on_latent_dim
        self.output_channels = output_channels
        self.upsampling = nn.Upsample(scale_factor=2, mode=self.upsample_mode)
        start_n_features = self.latent_dim if scale_on_latent_dim else 512
        self.fc1 = nn.Linear(self.latent_dim, start_n_features)
        self.relu = nn.ReLU(inplace=True)
        self.use_skip_inputs = use_skip_inputs
        if use_custom_first_block:
            first_block = nn.Sequential(
                nn.ConvTranspose2d(start_n_features, start_n_features, kernel_size=reshape_size, padding=0, groups=groups),
                nn.BatchNorm2d(start_n_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )
        else:
            first_block = nn.Upsample(scale_factor=reshape_size[0], mode=self.upsample_mode)
        deconv = first_block
        block_counts = {
            '18': [2, 2, 2, 2],
            '34': [3, 6, 4, 3]
        }['18']

        self.blocks = nn.ModuleList([
            deconv, # 7x7
            self.stage(start_n_features, start_n_features // 2, block_counts[0]), #14x14
            self.stage(start_n_features // 2, start_n_features // 4, block_counts[1]), # 28x28
            self.stage(start_n_features // 4, start_n_features // 8, block_counts[2]), # 56x56
            self.stage(start_n_features // 8, start_n_features // 8, block_counts[3]), # 112x112
        ])
        if start_index > 0:
            for i in range(0, start_index):
                stage_index = 1 + i // 2
                if self.blocks[stage_index][i % 2].upsample is not None:
                    b = self.blocks[stage_index][i % 2]
                    if b.identity_proj is None:
                        self.blocks[stage_index][i % 2] = b.upsample
                    else:
                        self.blocks[stage_index][i % 2] = nn.Sequential(b.upsample, b.identity_proj)
                else:
                    self.blocks[stage_index][i % 2] = nn.Identity()
                print('Deactivating stage {}, block {}'.format(stage_index, i%2))
        if self.use_skip_inputs:
            ins = [256, 128, 64, 64]
            outs = [start_n_features // 2, start_n_features // 4, start_n_features // 8, start_n_features // 8]
            self.projs = nn.ModuleList()
            for ip, op in zip(ins, outs):
                conv = nn.Conv2d(ip, op, 1, bias=False)
                nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
                bn = nn.BatchNorm2d(op, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                # bn = nn.GroupNorm(8, op, eps=1e-5, affine=True)
                relu = nn.ReLU(inplace=True)
                self.projs.append(nn.Sequential(conv, bn, relu))
        
        self.last_conv = nn.Conv2d(start_n_features // 8, output_channels, kernel_size=7, padding=3)
        self.activation_type = nn.ReLU
        self.last_activation = last_activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m != self.last_conv:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight, gain=1.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        if self.use_skip_inputs:
            y = self.fc1(x[-1])
            y = self.relu(y)
            y = y.view(-1, y.size()[1], 1, 1)
            y = self.blocks[0](y)
            for b, xk, proj in zip(self.blocks[1:], reversed(x[:-1]), self.projs):
                y = b(y)
                y = y + proj(xk)
            y = self.upsampling(y)
            y = self.last_conv(y)
            y = self.last_activation(y)
            return y
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = x.view(-1, x.size()[1], 1, 1)
            for b in self.blocks:
                x = b(x)
            x = self.upsampling(x)
            x = self.last_conv(x)
            x = self.last_activation(x)
            return x
    def stage(self, inplanes, planes, num_blocks):
        layers = []
        layers.append(ResNetDecoderBasicBlock(inplanes, planes, upsample=self.upsampling))
        for n in range(1, num_blocks):
            layers.append(ResNetDecoderBasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def set_activation(self, activation, init_fn):
        def recurse_fn(m):
            for mod_str, mod in m.named_children():
                if type(mod) == self.activation_type:
                    setattr(m, mod_str, activation)
                else:
                    init_fn(mod)
                    recurse_fn(mod)
        for k, v in self.named_children():
            recurse_fn(v)
        self.activation_type = type(activation)
    def _get_convs(self):
        convs = []
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                convs.append(module)
        return convs

class ResNetDecoderBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=None):
        super(ResNetDecoderBasicBlock, self).__init__()
        from functools import partial
        conv3x3 = partial(lambda inp, outp: nn.Conv2d(inp, outp, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        bn = partial(lambda p: nn.BatchNorm2d(p, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # gn = partial(lambda p: nn.GroupNorm(8, p, eps=1e-5, affine=True))
        # print('Using GN')
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.upsample = upsample
        self.identity_proj = None
        if inplanes != planes:
            self.identity_proj = nn.Sequential(nn.Conv2d(inplanes, planes, (1, 1)), nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    def forward(self, x):
        identity = x
        out = x
        if self.identity_proj is not None:
            identity = self.identity_proj(identity)
        if self.upsample is not None:
            identity = self.upsample(identity)
            out = self.upsample(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    def _get_convs(self):
        return [self.conv1, self.conv2]

