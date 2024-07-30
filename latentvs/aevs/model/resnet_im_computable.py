import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils import weight_norm
from aevs.model.im_computable import OpWithInteractionMatrixComputable, ReLUWrapper
import aevs.model.im_op_wrapper as iow

def fuse_conv_bn(conv, bn):
    '''Taken from the Pytorch repo, but it requires the modules to be in eval'''
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv

class BasicBlockIMWrapper(OpWithInteractionMatrixComputable):
    def __init__(self, op):
        super(BasicBlockIMWrapper, self).__init__(op)
        convert = lambda o: iow.OpIMWrapper.from_op(o)
        self.op.conv1 = convert(self.op.conv1)
        self.op.bn1 = convert(self.op.bn1)
        self.op.relu = convert(self.op.relu)
        self.op.conv2 = convert(self.op.conv2)
        self.op.bn2 = convert(self.op.bn2)
        if self.op.downsample is not None:
            self.op.downsample = convert(self.op.downsample)
    def forward_with_interaction_matrix(self, x, L):
        identity = x
        Lin = L.clone()
        out, L = self.op.conv1.forward_with_interaction_matrix(x, L)
        out, L = self.op.bn1.forward_with_interaction_matrix(out, L)
        out, L = self.op.relu.forward_with_interaction_matrix(out, L)

        out, L = self.op.conv2.forward_with_interaction_matrix(out, L)
        out, L = self.op.bn2.forward_with_interaction_matrix(out, L)

        if self.op.downsample is not None:
            identity, Lin = self.op.downsample.forward_with_interaction_matrix(x, Lin)

        out += identity
        L += Lin
        out, L = self.op.relu.forward_with_interaction_matrix(out, L)
        return out, L
    def replace_bn_with_wn(self):
        self.op.conv1.op = weight_norm(fuse_conv_bn(self.op.conv1.op, self.op.bn1.op))
        self.op.conv2.op = weight_norm(fuse_conv_bn(self.op.conv2.op, self.op.bn2.op))
        if self.op.downsample is not None:
            self.op.downsample.op[0].op = weight_norm(fuse_conv_bn(self.op.downsample.op[0].op, self.op.downsample.op[1].op))
            self.op.downsample.op[1] = iow.OpIMWrapper.from_op(nn.Identity())

        self.op.bn1 = iow.OpIMWrapper.from_op(nn.Identity())
        self.op.bn2 = iow.OpIMWrapper.from_op(nn.Identity())
    def replace_bn_with_sn(self):
        from torch.nn.utils import spectral_norm
        self.op.conv1.op = spectral_norm(fuse_conv_bn(self.op.conv1.op, self.op.bn1.op))
        self.op.conv2.op = spectral_norm(fuse_conv_bn(self.op.conv2.op, self.op.bn2.op))
        if self.op.downsample is not None:
            self.op.downsample.op[0].op = spectral_norm(fuse_conv_bn(self.op.downsample.op[0].op, self.op.downsample.op[1].op))
            self.op.downsample.op[1] = iow.OpIMWrapper.from_op(nn.Identity())

        self.op.bn1 = iow.OpIMWrapper.from_op(nn.Identity())
        self.op.bn2 = iow.OpIMWrapper.from_op(nn.Identity())


class BottleneckIMWrapper(OpWithInteractionMatrixComputable):
    def __init__(self, op):
        super(BottleneckIMWrapper, self).__init__(op)
        convert = lambda o: iow.OpIMWrapper.from_op(o)

        self.op.conv1 = convert(self.op.conv1)
        self.op.bn1 = convert(self.op.bn1)
        self.op.conv2 = convert(self.op.conv2)
        self.op.bn2 = convert(self.op.bn2)
        self.op.conv3 = convert(self.op.conv3)
        self.op.bn3 = convert(self.op.bn3)
        self.op.relu = convert(self.op.relu)
        if self.op.downsample is not None:
            self.op.downsample = convert(self.op.downsample)

    def forward_with_interaction_matrix(self, x, L):
        identity = x
        Lin = L
        out, L = self.op.conv1.forward_with_interaction_matrix(x, L)
        out, L = self.op.bn1.forward_with_interaction_matrix(out, L)
        out, L = self.op.relu.forward_with_interaction_matrix(out, L)

        out, L = self.op.conv2.forward_with_interaction_matrix(out, L)
        out, L = self.op.bn2.forward_with_interaction_matrix(out, L)
        out, L = self.op.relu.forward_with_interaction_matrix(out, L)

        out, L = self.op.conv3.forward_with_interaction_matrix(out, L)
        out, L = self.op.bn3.forward_with_interaction_matrix(out, L)

        if self.op.downsample is not None:
            identity, Lin = self.op.downsample.forward_with_interaction_matrix(x, Lin)

        out += identity
        L += Lin
        out, L = self.op.relu.forward_with_interaction_matrix(out, L)

        return out, L
    def replace_bn_with_wn(self):
        self.op.conv1.op = weight_norm(fuse_conv_bn(self.op.conv1.op, self.op.bn1.op))
        self.op.conv2.op =  weight_norm(fuse_conv_bn(self.op.conv2.op, self.op.bn2.op))
        self.op.conv3.op =  weight_norm(fuse_conv_bn(self.op.conv3.op, self.op.bn3.op))

        self.op.bn1 = iow.OpIMWrapper.from_op(nn.Identity())
        self.op.bn2 = iow.OpIMWrapper.from_op(nn.Identity())
        self.op.bn3 = iow.OpIMWrapper.from_op(nn.Identity())

class ResNetIMWrapper(OpWithInteractionMatrixComputable):

    def __init__(self, op: nn.Module, latent_dim):
        super(ResNetIMWrapper, self).__init__(op)
        from aevs.model.im_computable import Flatten
        self.latent_dim = latent_dim
        convert = lambda o: iow.OpIMWrapper.from_op(o)
        self.op.conv1 = convert(self.op.conv1)
        self.op.bn1 = convert(self.op.bn1)
        self.op.relu = convert(self.op.relu)
        self.op.maxpool = convert(self.op.maxpool)
        self.op.layer1 = convert(self.op.layer1)
        self.op.layer2 = convert(self.op.layer2)
        self.op.layer3 = convert(self.op.layer3)
        self.op.layer4 = convert(self.op.layer4)
        self.op.avgpool = convert(self.op.avgpool)
        self.op.fc = convert(self.op.fc)
        self.op.flatten = convert(Flatten())
        self.activation_type = ReLUWrapper
    def forward_with_interaction_matrix(self, x, L):

        x, L = self.op.conv1.forward_with_interaction_matrix(x, L)
        x, L = self.op.bn1.forward_with_interaction_matrix(x, L)
        x, L = self.op.relu.forward_with_interaction_matrix(x, L)
        x, L = self.op.maxpool.forward_with_interaction_matrix(x, L)

        # for i, b in zip(range(self.block_stop_im), self.blocks[:self.block_stop_im]):
        #     x, L = b.forward_with_interaction_matrix(x, L)
        x, L = self.op.layer1.forward_with_interaction_matrix(x, L)
        x, L = self.op.layer2.forward_with_interaction_matrix(x, L)
        x, L = self.op.layer3.forward_with_interaction_matrix(x, L)
        x, L = self.op.layer4.forward_with_interaction_matrix(x, L)

        x, L = self.op.avgpool.forward_with_interaction_matrix(x, L)
        x, L = self.op.flatten.forward_with_interaction_matrix(x, L)
        x, L = self.op.fc.forward_with_interaction_matrix(x, L)

        return x, L
    def forward_encode(self, x):
        return self.op(x)
        # x = self.op.conv1(x)
        # x = self.op.bn1(x)
        # x = self.op.relu(x)
        # x = self.op.maxpool(x)

        # x = self.op.layer1(x)
        # x = self.op.layer2(x)
        # x = self.op.layer3(x)
        # x = self.op.layer4(x)

        # print(x.size())
        # x = self.op.avgpool(x)
        # print(x.size())
        # x = self.op.flatten(x)
        # print(x.size())
        # x = self.op.fc(x)

        # return x
    def forward_encode_with_interaction_matrix(self, x, L):
        return self.forward_with_interaction_matrix(x, L)

    # def forward(self, x):
    #     x = self.op.conv1(x)
    #     x = self.op.bn1(x)
    #     x = self.op.relu(x)
    #     print(x.size())
    #     x = self.op.maxpool(x)
    #     print(x.size())
    #     # for i, b in zip(range(self.block_stop_im), self.blocks[:self.block_stop_im]):
    #     #     x, L = b.forward_with_interaction_matrix(x, L)
    #     x = self.op.layer1(x)
    #     x = self.op.layer2(x)
    #     x = self.op.layer3(x)
    #     x = self.op.layer4(x)

    #     x = self.op.avgpool(x)
    #     x = self.op.flatten(x)
    #     x = self.op.fc(x)

    #     return x
    def set_activation(self, activation, init_fn):
        import sys
        a = iow.OpIMWrapper.from_op(activation)
        def recurse_fn(m):
            for mod_str, mod in m.named_children():

                if type(mod) == self.activation_type:
                    setattr(m, mod_str, a)
                else:
                    init_fn(mod)
                    recurse_fn(mod)
        for _k, v in self.op.named_children():
            recurse_fn(v)
        self.activation_type = type(a)
    def replace_bn_with_wn(self):
        self.op.conv1.op = weight_norm(fuse_conv_bn(self.op.conv1.op, self.op.bn1.op))
        self.op.bn1 = iow.OpIMWrapper.from_op(nn.Identity())
        for m in [self.op.layer1, self.op.layer2, self.op.layer3, self.op.layer4]:
            for block in m.op.children():
                block.replace_bn_with_wn()
    def replace_bn_with_sn(self):
        from torch.nn.utils import spectral_norm
        self.op.conv1.op = spectral_norm(fuse_conv_bn(self.op.conv1.op, self.op.bn1.op))
        self.op.bn1 = iow.OpIMWrapper.from_op(nn.Identity())
        for m in [self.op.layer1, self.op.layer2, self.op.layer3, self.op.layer4]:
            for block in m.op.children():
                block.replace_bn_with_wn()

    def _get_convs(self):
        convs = []
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                convs.append(m)
        return convs
    def preprocess(self, x):
        return x / 255.0
    def unprocess(self, x):
        return x * 255.0

iow.OpIMWrapper.register_wrappers({
    BasicBlock: BasicBlockIMWrapper,
    Bottleneck: BottleneckIMWrapper,
    ResNet: ResNetIMWrapper
})