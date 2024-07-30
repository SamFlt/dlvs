import torch
from torch import nn
import numpy as np

from aevs.model.im_computable import OpWithInteractionMatrixComputable, im_is_in_image_rep
from aevs.model.im_op_wrapper import OpIMWrapper
from aevs.model.im_computable_models import ResNetAEIMComputable

class PoseEmbedder(nn.Module):
    '''
    Module that embeds a 6D pose to a latent space of dimension projection_size
    '''
    def __init__(self, projection_size, hidden_counts=[32, 64]):
        super(PoseEmbedder, self).__init__()
        current_nodes = 6
        self.embedder_layers = []

        for count in hidden_counts:
            layer = nn.Linear(current_nodes, count)
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            current_nodes = count
            self.embedder_layers.append(layer)
            self.embedder_layers.append(nn.ReLU(inplace=True))

        self.embedder_layers.append(nn.Linear(current_nodes, projection_size))
        self.embedder = OpIMWrapper.from_op(nn.Sequential(*self.embedder_layers))

    def forward(self, p):
        '''
        Compute the latent representations for a set of poses
        '''
        zp = self.embedder(self.preprocess(p))
        return zp
        # return zp / torch.norm(zp, p=2, dim=-1, keepdim=True)

    def forward_with_interaction_matrix(self, p, Lp):
        '''
        Compute the latent representations and the latent interaction matrices for a set of poses
        '''
        return self.embedder.forward_with_interaction_matrix(self.preprocess(p), self.preprocess_im(Lp))

    def preprocess(self, p):
        return p

    def preprocess_im(self, Lp):
        return Lp

class SineActivation(nn.Module):
    def __init__(self, omega, trainable=False):
        super(SineActivation, self).__init__()
        self.omega = nn.Parameter(torch.tensor(omega), requires_grad=trainable)

    def forward(self, x):
        return torch.sin(self.omega * x)

class SineIMWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.op(x)
        omega = self.op.omega
        ci = 1 if im_is_in_image_rep(L) else -1

        L *= omega * torch.cos(omega * x).unsqueeze(ci)
        return z, L
OpIMWrapper.register_wrappers({
    SineActivation: SineIMWrapper
})

class PoseEmbedderSIREN(nn.Module):
    '''
    Module that embeds a 6D pose to a latent space of dimension projection_size.
    This model uses Sine activations
    '''
    def __init__(self, projection_size, hidden_counts, omega=30.0):
        super(PoseEmbedderSIREN, self).__init__()
        def init_weights(l: nn.Linear, a: SineActivation, is_first: bool):
            in_features = l.in_features
            omega = a.omega
            with torch.no_grad():
                if is_first:
                    l.weight.uniform_(-1 / in_features,
                                                1 / in_features)
                else:
                    l.weight.uniform_(-np.sqrt(6 / in_features) / omega,
                                                np.sqrt(6 / in_features) / omega)


        current_nodes = 6
        self.embedder_layers = []

        for i, count in enumerate(hidden_counts):
            layer = nn.Linear(current_nodes, count)
            activation = SineActivation(omega if i == 0 else 1.0)
            init_weights(layer, activation, i == 0)
            nn.init.zeros_(layer.bias)
            current_nodes = count
            self.embedder_layers.append(layer)
            self.embedder_layers.append(activation)

        self.embedder_layers.append(nn.Linear(current_nodes, projection_size))
        self.embedder = OpIMWrapper.from_op(nn.Sequential(*self.embedder_layers))



    def forward(self, p):
        zp = self.embedder(self.preprocess(p))
        return zp

    def forward_with_interaction_matrix(self, p, Lp):
        return self.embedder.forward_with_interaction_matrix(self.preprocess(p), self.preprocess_im(Lp))

    def preprocess(self, p):
        return p

    def preprocess_im(self, Lp):
        return Lp



class ImageEncoder(nn.Module):
    '''
    Image encoder for MLVS.
    Projects a grayscale image on a latent space
    '''
    def __init__(self, projection_size, encoder_version='18', width_factor=0.5, use_coord_conv=False):
        super(ImageEncoder, self).__init__()
        self.encoder = ResNetAEIMComputable(encoder_version, '18', False, False, projection_size, 1, (224, 224),
                        use_wn_instead_of_bn=True, use_sn_instead_of_bn=False, replace_end_pooling=True, width_factor=width_factor, use_coord_conv=use_coord_conv).encoder

    def forward(self, x):
        zi = self.encoder(x)
        return zi

    def preprocess(self, x):
        return x / 255.0
    def unprocess(self, x):
        return x * 255.0