from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import Conv2dStaticSamePadding, efficientnet
from aevs.model.im_computable import Conv2DWrapper, Flatten, OpWithInteractionMatrixComputable
from aevs.model.im_computable_models import AEIMComputable, fuse_conv_bn
import aevs.model.im_op_wrapper as iow
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
from typing import *
from models.common import ResNetDecoder
from models.init import linear_init, relu_init
class EfficientNetAE(AEIMComputable):
    def __init__(self, latent_size):
        super(EfficientNetAE, self).__init__()
        self.latent_dim = latent_size
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=latent_size,
                                                     in_channels=1)
        self.decoder = self.decoder = ResNetDecoder(False, True, False,
                                                        self.latent_dim, 1, 'bilinear', 16, (7, 7),
                                                        nn.Sigmoid(), start_index=-1)
        relu_init(self.encoder._conv_stem)
        linear_init(self.encoder._fc)
        self._fuse_conv_and_bn()
        
        self.encoder = iow.OpIMWrapper.from_op(self.encoder)
        

    def _fuse_conv_and_bn(self):
        was_training = self.training
        self.eval()
        def fuse(conv, bn):
            return weight_norm(fuse_conv_bn(conv, bn)), nn.Identity()
        e = self.encoder
        e._conv_stem, e._bn0 = fuse(e._conv_stem, e._bn0)
        e._conv_head, e._bn1 = fuse(e._conv_head, e._bn1)
        blocks: List[MBConvBlock] = e._blocks
        for b in blocks:
            if hasattr(b, '_expand_conv'):
                b._expand_conv, b._bn0 = fuse(b._expand_conv, b._bn0)
            b._depthwise_conv, b._bn1 = fuse(b._depthwise_conv, b._bn1)
            b._project_conv, b._bn2 = fuse(b._project_conv, b._bn2)
        if was_training:
            self.train()

class EfficientNetImComputable(OpWithInteractionMatrixComputable):
    def __init__(self, op: EfficientNet):
        super(EfficientNetImComputable, self).__init__(op)
        convert = iow.OpIMWrapper.from_op

        self.op._conv_stem = convert(self.op._conv_stem)
        self.op._bn0 = convert(self.op._bn0)
        self.op._swish = convert(self.op._swish)
        self.stem_seq = convert(nn.Sequential(self.op._conv_stem,
                                            self.op._bn0,
                                            self.op._swish))
        self.op._conv_head = convert(self.op._conv_head)
        self.op._bn1 = convert(self.op._bn1)
        self.head_seq = convert(nn.Sequential(self.op._conv_head,
                                            self.op._bn1,
                                            self.op._swish))

        self.op._avg_pooling = convert(self.op._avg_pooling)
        self.op._fc = convert(self.op._fc)
        self.flatten = convert(Flatten())
        self.latent_seq = convert(nn.Sequential(self.op._avg_pooling,
                                            self.flatten,
                                            self.op._fc))

        for i in range(len(self.op._blocks)):
            self.op._blocks[i] = convert(self.op._blocks[i])
    def forward_with_interaction_matrix(self, x, L):
        assert not self.training
        # print('=========')
        # print(L.mean(), L.std())
        x, L = self.stem_seq.forward_with_interaction_matrix(x, L)
        # print(L.mean(), L.std())
        for block in self.op._blocks:
            x, L = block.forward_with_interaction_matrix(x, L)
            # print(x.mean(), L.mean(), L.std())
        
        x, L = self.head_seq.forward_with_interaction_matrix(x, L)
        x, L = self.latent_seq.forward_with_interaction_matrix(x, L)
        return x, L

    
            

class Conv2dStaticSamePaddingWrapper(OpWithInteractionMatrixComputable):
    def __init__(self, op: Conv2dStaticSamePadding):
        super(Conv2dStaticSamePaddingWrapper, self).__init__(op)
    def forward_with_interaction_matrix(self, x, L):
        from models.im_computable import permute_im_to_image_rep_if_required
        b, c, h, w = x.size()
        z = self.forward(x)
        L = permute_im_to_image_rep_if_required(x.size(), L).view(b * 6, c, h, w)
        L = self.op.static_padding(L)
        _, zc, zh, zw = z.size()
        Ln = F.conv2d(L, self.op.weight,
                    stride=self.op.stride,
                    padding=self.op.padding,
                    dilation=self.op.dilation,
                    groups=self.op.groups).view(b, 6, zc, zh, zw)
        return z, Ln

class MBConvBlockWrapper(OpWithInteractionMatrixComputable):
    def __init__(self, op: MBConvBlock):
        super(MBConvBlockWrapper, self).__init__(op)
        convert = lambda o: iow.OpIMWrapper.from_op(o)
        op: MBConvBlock = self.op
        op._swish = convert(op._swish)

        if op._block_args.expand_ratio != 1:
            op._expand_conv = convert(op._expand_conv)
            op._bn0 = convert(op._bn0)
            self.expand_seq = convert(nn.Sequential(op._expand_conv, op._bn0, op._swish))
        
        
        op._depthwise_conv = convert(op._depthwise_conv)
        op._bn1 = convert(op._bn1)
        self.depth_seq = convert(nn.Sequential(op._depthwise_conv, op._bn1, op._swish))
        if op.has_se:
            self.se_pooling = convert(nn.AdaptiveAvgPool2d(1))
            op._se_reduce = convert(op._se_reduce)
            op._se_expand = convert(op._se_expand)
            self._sigmoid = convert(nn.Sigmoid())
            self.se_seq =  convert(nn.Sequential(self.se_pooling, op._se_reduce, op._swish, op._se_expand, self._sigmoid))
        op._project_conv = convert(op._project_conv)
        op._bn2 = convert(op._bn2)
        self.proj_seq = convert(nn.Sequential(op._project_conv, op._bn2))
            
    def forward(self, x, drop_connect_rate):
        return self.op(x, drop_connect_rate=drop_connect_rate)

    def forward_with_interaction_matrix(self, x, L):
        assert not self.training
        x_orig, L_orig = x, L
        
        if self.op._block_args.expand_ratio != 1:
            x, L = self.expand_seq.forward_with_interaction_matrix(x, L)
        x, L = self.depth_seq.forward_with_interaction_matrix(x, L)

        # Squeeze and Excitation
        if self.op.has_se:
            x_squeezed, L_squeezed = self.se_seq.forward_with_interaction_matrix(x, L)
            x, L = x_squeezed * x, L * L_squeezed
        x, L = self.proj_seq.forward_with_interaction_matrix(x, L)
        

        # Skip connection and drop connect
        input_filters, output_filters = self.op._block_args.input_filters, self.op._block_args.output_filters
        if self.op.id_skip and self.op._block_args.stride == 1 and input_filters == output_filters:
            x, L = x + x_orig, L + L_orig  # skip connection
        
        return x, L
        

iow.OpIMWrapper.register_wrappers({
    Conv2dStaticSamePadding: Conv2dStaticSamePaddingWrapper,
    MBConvBlock: MBConvBlockWrapper,
    EfficientNet: EfficientNetImComputable
}) 



