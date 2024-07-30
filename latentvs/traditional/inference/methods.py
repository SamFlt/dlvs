import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

from traditional.model import DVS as DVSpt
from traditional.model import DVSInteractionMatrix, ImageGradients
from generator import DCTBatchServo


from inference.interaction_matrix_mixer import *
from inference.utils import *
from inference.optimizers import *
from utils.custom_typing import *
from collections import namedtuple
import yaml
from inference import io
from inference.methods import VSMethod

class DVSMethod(VSMethod):
    def __init__(self, Z, border, arguments, generator_parameters, optimizer=None, mixer=DesiredInteractionMatrix()):
        self.border = border
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            self.optimizer = LevenbergMarquardtOptimizer(0.01, 1000, 0.999, 0.0, self.device)
        else:
            self.optimizer = optimizer


        self.Zinv = torch.tensor(1.0 / Z, requires_grad=False, device=self.device)
        self.dvs = DVSpt(arguments.h, arguments.w, generator_parameters['px'], generator_parameters['py'],
                                        generator_parameters['u0'], generator_parameters['v0'], self.border).to(self.device)
        self.Li_fn = DVSInteractionMatrix(arguments.h, arguments.w,
                                        generator_parameters['px'], generator_parameters['py'],
                                        generator_parameters['u0'], generator_parameters['v0'],
                                        self.border).to(self.device)
        self.im_grads = ImageGradients(self.border).to(self.device)
        self.mixer = mixer
    def name(self):
        return 'dvs'
    def on_end_iter(self, iter_idx):
        self.optimizer.on_iter(iter_idx)

    def compute_vc(self, current_images, desired_images, iter, run_indices=None, _=None):
        assert run_indices is None, 'Run indices != None (early stopping) not yet supported'
        Li = None
        b = self.border
        if self.mixer.requires_Li():
            Li = self.compute_features_and_interaction_matrix(current_images)[1]
        L = self.mixer.compute_final_L(Li, self.Lid)
        I = current_images[:, b:-b,b:-b].contiguous()
        Id = desired_images[:, b:-b, b:-b].contiguous()
        vcs = self.optimizer(L, (I - Id).view(I.size(0), -1))

        # vcs = self.dvs((current_images, desired_images, self.Zinv, self.mu))
        ssd = torch.sum((I - Id) ** 2, dim=(1, 2)).unsqueeze(-1)
        return vcs, ssd.cpu().numpy(), []
    def error_shape(self):
        return 1
    def on_new_batch(self, batch_idx, Id_processed, _):
        self.optimizer.reset()
        self.mixer.reset()
        if self.mixer.requires_Lid():
            self.Lid = self.compute_features_and_interaction_matrix(Id_processed)[1]
        else:
            self.Lid = None
        return []
    def process_image(self, images):
        with torch.no_grad():
            img = to_gray(images)
            img = torch.tensor(img, requires_grad=False, device=self.device)
            img = img.float()
            return img
    def compute_features(self, images):
        b = self.border
        return images[:, b:-b,b:-b].contiguous()
    def compute_features_and_interaction_matrix(self, I_processed):
        b = self.border
        Idx, Idy = self.im_grads(I_processed.squeeze(1))
        Li = self.Li_fn((Idx, Idy, self.Zinv))
        return I_processed[:, b:-b,b:-b].contiguous(), Li
    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        return lambda args: DVSMethod(
                io.get_or_default('Z', node, defaults),
                io.get_or_default('border', node, defaults),
                args,
                io.get_or_default('camera_settings', node, defaults),
                io.optimizer_from_yaml(io.get_or_default('optimizer', node, defaults), device),
                io.interaction_matrix_mixer_from_yaml(io.get_or_default('interaction_matrix_mixer', node, defaults)),
            )
class PCAMethod(VSMethod):
    def __init__(self, Z, folder, border, ncomponents, arguments, generator_parameters, optimizer=None, mixer=DesiredInteractionMatrix()):
        self.border = border
        self.ncomponents = ncomponents
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        if optimizer is None:
            self.optimizer = LevenbergMarquardtOptimizer(0.01, 1000, 0.999, 0.0, self.device)
        else:
            self.optimizer = optimizer
        self.Zinv = torch.tensor(1.0 / Z, requires_grad=False, device=self.device)
        self.Li_fn = DVSInteractionMatrix(arguments.h, arguments.w,
                                        generator_parameters['px'], generator_parameters['py'],
                                        generator_parameters['u0'], generator_parameters['v0'],
                                        self.border).to(self.device)
        self.im_grads = ImageGradients(self.border).to(self.device)
        self.mixer = mixer

        np_proj = np.load(str(folder / 'components.npy'))
        assert self.ncomponents <= len(np_proj), '# of selected components should be < to # of saved components'
        np_proj = np_proj[:self.ncomponents]
        self.projection = torch.tensor(np_proj, requires_grad=False, device=self.device).float()
        self.projection_perm = torch.transpose(self.projection, 0, 1)

        np_mean = np.load(str(folder / 'mean.npy'))
        self.mean = torch.tensor(np_mean, requires_grad=False, device=self.device).float()

    def name(self):
        return 'PCA_vs_{}'.format(self.ncomponents)
    def on_end_iter(self, iter_idx):
        self.optimizer.on_iter(iter_idx)

    def compute_vc(self, current_images, desired_images, iter, run_indices=None):
        Li = None

        if self.mixer.requires_Li():
            projc, Li = self.compute_features_and_interaction_matrix(current_images)
        else:
            projc = self.compute_features(current_images)
        Lid, projd = self.Lid, self.projd
        if run_indices is not None:
            Lid, projd = self.Lid[run_indices], self.projd[run_indices]
        L = self.mixer.compute_final_L(Li, Lid)

        error = projc - projd
        vcs = self.optimizer(L, error)
        return vcs, error.cpu().numpy(), []
    def error_shape(self):
        return self.ncomponents
    def on_new_batch(self, batch_idx, Id_processed):
        self.optimizer.reset()
        self.mixer.reset()
        if self.mixer.requires_Lid():
            self.projd, self.Lid = self.compute_features_and_interaction_matrix(Id_processed)
        else:
            self.projd = self.compute_features(Id_processed)
            self.Lid = None
        return []
    def process_image(self, images):
        with torch.no_grad():
            img = to_gray(images)
            img = torch.tensor(img, requires_grad=False, device=self.device)
            img = img.float()
            return img
    def compute_features(self, images):
        I = images[:, self.border:-self.border, self.border:-self.border].contiguous()
        I = I.view(I.size(0), -1)
        I = I - self.mean
        proj = torch.matmul(I, self.projection_perm)
        return proj
    def compute_features_and_interaction_matrix(self, I_processed):
        Idx, Idy = self.im_grads(I_processed.squeeze(1))
        Li = self.Li_fn((Idx, Idy, self.Zinv))
        proj = self.compute_features(I_processed)
        Lpca = torch.matmul(self.projection, Li)
        return proj, Lpca

    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        root_pca_path = Path(globals['pca_folder'])
        return lambda args: PCAMethod(
            io.get_or_default('Z', node, defaults),
            root_pca_path / node['weights_path'],
            io.get_or_default('border', node, defaults),
            node['num_components'],
            args,
            io.get_or_default('camera_settings', node, defaults),
            io.optimizer_from_yaml(io.get_or_default('optimizer', node, defaults), device),
            io.interaction_matrix_mixer_from_yaml(io.get_or_default('interaction_matrix_mixer', node, defaults)),
        )

class DCTVS(VSMethod):
    def __init__(self, Z, border, batch_size, ncoeffs, arguments, generator_parameters, mu, iterGN, lambdaGN, muFactorIter, useESM):
        self.border = border
        self.ncoeffs = ncoeffs
        self.use_esm = useESM
        self.dct_vs = DCTBatchServo(batch_size, arguments.h,
                                    self.border, ncoeffs,
                                    1.0, Z,
                                    generator_parameters['px'], generator_parameters['py'],
                                    generator_parameters['u0'], generator_parameters['v0'], mu, iterGN, lambdaGN, muFactorIter, self.use_esm)
    def name(self):
        return 'dct_{}_{}'.format(self.ncoeffs, 'ESM' if self.use_esm else 'Lid')

    def compute_vc(self, current_images, desired_images, iter, run_indices):
        res = self.dct_vs(np.ascontiguousarray(current_images), np.ascontiguousarray(desired_images))
        vcs = torch.tensor(np.array([r.v() for r in res]))
        error = np.array([r.error() for r in res])
        return vcs, error, []
    def error_shape(self):
        return self.ncoeffs
    def on_new_batch(self, batch_idx, Id_processed):
        self.dct_vs.reset()
        return []
    def process_image(self, images):
        img = to_gray(images)
        return img.astype(np.uint8)
    def compute_features_and_interaction_matrix(self, I_processed):
        res = self.dct_vs.compute_features_and_interaction_matrix(len(I_processed), I_processed)
        s = np.array([r.current_features() for r in res])
        Ls = np.array([r.interaction_matrix() for r in res])
        return torch.tensor(s), torch.tensor(Ls)

    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        opt_node = node['optimizer']
        im = io.interaction_matrix_mixer_from_yaml(io.get_or_default('interaction_matrix_mixer', node, defaults)),
        assert not isinstance(im, CurrentInteractionMatrix), 'Current interaction matrix not supported for DCTVS'
        return lambda args: DCTVS(
            io.get_or_default('Z', node, defaults),
            io.get_or_default('border', node, defaults),
            defaults['batch_size'],
            node['num_components'],
            args,
            io.get_or_default('camera_settings', node, defaults),
            opt_node['mu'],
            opt_node['iter_gauss_newton'],
            opt_node['lambda_gauss_newton'],
            opt_node['mu_factor'],
            isinstance(im, AverageCurrentAndDesiredInteractionMatrices)
        )


for name, cls in [('DVS', DVSMethod), ('PCAVS', PCAMethod), ('DCTVS', DCTVS)]:
    io.model_builders[name] = cls.make_method_from_yaml
