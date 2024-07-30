import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import models
from .CoordConv import AddCoords
from efficientnet_pytorch import EfficientNet
from functools import partial

class Siame_se3(nn.Module):
    def __init__(self, efficient_net_version='efficientnet-b3', siamese_bottleneck=None,
                relu_leak=0.0, hidden_layer_units_list=[1024, 256],
                use_skip_connection=False, use_group_norm=False,
                drop_connect_rate=0.0, dropout=0.0,
                return_embeddings=False):
        super(Siame_se3, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.feature_extractor = EfficientNet.from_pretrained(efficient_net_version)
        self.feature_extractor._global_params = self.feature_extractor._global_params._replace(drop_connect_rate=drop_connect_rate)
        self.feature_extractor.set_swish(memory_efficient=True)
        self.dropout = dropout
        self.use_skip_connection = use_skip_connection
        self.relu_leak = relu_leak
        
        self.siamese_bottleneck = siamese_bottleneck
        if self.siamese_bottleneck is not None:
            self.siamese_fc = nn.Linear(1536, self.siamese_bottleneck, bias=False)
        
        self.return_embeddings = return_embeddings
        relu_init = partial(lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        leaky_relu_init = partial(lambda x: nn.init.kaiming_normal_(x, a=relu_leak, nonlinearity='leaky_relu'))
        
        init_kaiming = relu_init if relu_leak else leaky_relu_init

        # for m in self.feature_extractor.children():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.momentum = 0.01
        def bn_to_gn(module):
            new_modules = {}
            for name, m in module._modules.items():
                if isinstance(m, nn.BatchNorm2d):
                    x = m.num_features
                    g = 32
                    while x % g > 0:
                        g -= 1
                    print('groups:', g)
                    module._modules[name] = nn.GroupNorm(g, x)
                if isinstance(m, nn.Module):
                    bn_to_gn(m)
        if use_group_norm:
            print('Using groupnorm')
            bn_to_gn(self.feature_extractor)
        self.fcs = []
        end_units_dict = {
            'efficientnet-b0': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
        }
        sub_units = end_units_dict[efficient_net_version] if self.siamese_bottleneck is None else self.siamese_bottleneck
        # self.bn_sub = nn.BatchNorm1d(sub_units, momentum=0.01, affine=True, track_running_stats=True)
        previous_units = sub_units
        if self.use_skip_connection:
            self.residual_proj = nn.Linear(previous_units, hidden_layer_units_list[-1])
            init_kaiming(self.residual_proj.weight)
        for units in hidden_layer_units_list:
            self.fcs.append(nn.Linear(previous_units, units, bias=True))
            previous_units = units
        for i, f in enumerate(self.fcs):
            self.add_module("fc_{}".format(i), f)
            init_kaiming(f.weight)
        self.velocity_fc = nn.Linear(previous_units, 6, bias=False)
        #nn.init.normal_(self.velocity_fc.weight, mean=0.0, std=0.1)
    def _get_activation(self):
        relu_fn = partial(lambda x: F.relu(x, inplace=True))
        leaky_relu = partial(lambda x: F.leaky_relu(x, negative_slope=self.relu_leak, inplace=True))
        
        return relu_fn if self.relu_leak == 0.0 else leaky_relu
    def forward(self, x):
        activation = self._get_activation()
        I, Id = x
        If = self.feature_extractor.extract_features(I)
        Idf = self.feature_extractor.extract_features(Id)
        units = If.size()[1]
        If = nn.AdaptiveAvgPool2d(1)(If).view(-1, units)
        Idf = nn.AdaptiveAvgPool2d(1)(Idf).view(-1, units)
        if self.siamese_bottleneck:
            If = self.siamese_fc(If)
            Idf = self.siamese_fc(Idf)
        features = If - Idf
        features = F.dropout(features, p=self.dropout, training=self.training, inplace=False)
        # features = self.bn_sub(features)
        first_features = features

        for f in self.fcs:
            features = f(features)
            features = activation(features)
            features = F.dropout(features, p=self.dropout, training=self.training, inplace=False)
        if self.use_skip_connection:
            proj = self.residual_proj(first_features)
            proj = activation(proj)
            # proj = F.dropout(proj, p=self.dropout, training=self.training, inplace=False)
            velocity = self.velocity_fc(features.add_(proj))
        else:
            velocity = self.velocity_fc(features)
        if self.return_embeddings:
            return velocity, If, Idf
        else:
            return velocity
        
    def forward_from_features(self, x):
        activation = self._get_activation()
        If, Idf = x
        features = If - Idf
        # features = self.bn_sub(features)
        first_features = features
        for f in self.fcs:
            features = f(features)
            features = activation(features)
            features = F.dropout(features, p=self.dropout, training=self.training, inplace=False)
        if self.use_skip_connection:
            proj = self.residual_proj(first_features)
            proj = activation(proj)
            proj = F.dropout(proj, p=self.dropout, training=self.training, inplace=False)
            velocity = self.velocity_fc(features.add_(proj))
        else:
            velocity = self.velocity_fc(features)
        return velocity
    def normalize(self, images):
        return ((images / 255.0) - self.mean) / self.std
        
    def _apply(self, fn):
        super(Siame_se3, self)._apply(fn)
        self.mean = fn(self.mean)
        self.std = fn(self.std)
        return self

class Siame_se3Inception(nn.Module):
    def __init__(self, hidden_layer_units_list=[1024, 256], dropout=0.0):
        super(Siame_se3Inception, self).__init__()
        from torchvision.models import inception_v3
        self.feature_extractor = inception_v3(pretrained=True, aux_logits=True)
        self.feature_extractor.AuxLogits.fc = nn.Identity()
        self.dropout = dropout
        # self.feature_extractor.set_swish(memory_efficient=True)
        # for m in self.feature_extractor.children():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.momentum = 0.001
        
        self.fcs = []
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        previous_units = 2048 + 768
        for units in hidden_layer_units_list:
            self.fcs.append(nn.Linear(previous_units, units, bias=True))
            previous_units = units
        for i, f in enumerate(self.fcs):
            self.add_module("fc_{}".format(i), f)
        self.velocity_fc = nn.Linear(previous_units, 6, bias=False)

    def _extractor_forward(self, x):
        #https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html#inception_v3 , modified a bit to always use logits
        inception = self.feature_extractor
        # N x 3 x 299 x 299
        x = inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = inception.aux_logits
        if aux_defined:
            aux = inception.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, p=self.dropout, training=inception.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x, aux
    def forward(self, x):
        I, Id = x
        
        If, If_aux = self._extractor_forward(I)
        Idf, Idf_aux = self._extractor_forward(Id)
        If = torch.cat((If, If_aux), dim=-1)
        Idf = torch.cat((Idf, Idf_aux), dim=-1)

        features = If - Idf
        for f in self.fcs:
            features = f(features)
            features = F.relu(features, inplace=True)
            features = F.dropout(features, p=self.dropout, training=self.training, inplace=False)
        velocity = self.velocity_fc(features)
        return velocity
    def forward_from_features(self, x):
        If, Idf = x
        features = If - Idf
        for f in self.fcs:
            features = f(features)
            features = F.relu(features, inplace=True)
        velocity = self.velocity_fc(features)
        return velocity
    def normalize(self, images):
        return ((images / 255.0) - self.mean) / self.std
        
    def _apply(self, fn):
        super(Siame_se3Inception, self)._apply(fn)
        self.mean = fn(self.mean)
        self.std = fn(self.std)
        return self