
import torch
from torch import nn
from enum import Enum
from pathlib import Path

from traditional.model import DVSInteractionMatrix, ImageGradients
from aevs.model.im_computable import L2Normalize, L2NormalizeWrapper, permute_im_to_vec_rep_if_required_minimal_checks

from inference.interaction_matrix_mixer import *
from inference.utils import *
from inference.optimizers import *
from inference.methods import VSMethod
from utils.custom_typing import *
import yaml
from inference import io


class Weighting(Enum):
    Identity = 1,
    DecoderError = 2


class NNIMVS(VSMethod):
    def __init__(self, Z: PositiveNonZeroFloat, model_path: Path, name: str, generator_parameters: GeneratorParameters,
                vs_arguments: VSArguments, border: UnsignedInt = 10,
                do_decode: bool =True, optimizer: Optional[Optimizer] = None, interaction_matrix_mixer: InteractionMatrixMixer = DesiredInteractionMatrix(),
                weighting: Weighting = Weighting.Identity, device: str = 'cuda'):
        self.device = device
        self.model: nn.Module = torch.load(str(model_path), map_location=torch.device(self.device))
        self.vs_arguments = vs_arguments
        self.model.eval()
        self._name = name
        self.border = border
        self.weighting = weighting
        if optimizer is None:
            self.optimizer = LevenbergMarquardtOptimizer(0.01, 1000, 0.995, 0.0, self.device)
        else:
            self.optimizer = optimizer


        self.Zinv = torch.tensor(1.0 / Z, requires_grad=False, device=self.device)
        self.Li_fn = DVSInteractionMatrix(vs_arguments.h, vs_arguments.w,
                                        generator_parameters['px'], generator_parameters['py'],
                                        generator_parameters['u0'], generator_parameters['v0'],
                                        self.border).to(self.device)
        self.im_grads = ImageGradients(self.border).to(self.device)
        self.do_decode = do_decode
        self.interaction_matrix_mixer = interaction_matrix_mixer

        self.forward_with_im_fn = self.model.forward_encode_with_interaction_matrix
        self.forward_encode_fn = self.model.forward_encode
        from torch.nn.utils.spectral_norm import remove_spectral_norm
        for module in self.model.modules():
            try:
                remove_spectral_norm(module)
            except ValueError:
                pass
            
    def on_end_iter(self, iter_idx) -> None:
        self.optimizer.on_iter(iter_idx)
    def name(self) -> str:
        return self._name
    def error_shape(self) -> UnsignedInt:
        return self.model.latent_dim
    def on_new_batch(self, batch_idx, Id_processed, _) -> None:
        self.interaction_matrix_mixer.reset()
        self.optimizer.reset()
        with torch.no_grad():
            b = self.border
            Id_cropped = Id_processed[:, :, b:-b, b:-b].contiguous()
            if self.interaction_matrix_mixer.requires_Lid():
                Idx, Idy = self.im_grads(Id_processed.squeeze(1))
                interaction_matrix_image = self.Li_fn((Idx, Idy, self.Zinv))

                if self.do_decode:
                    self.Id_z, Lnn = self.forward_with_im_fn(Id_cropped, interaction_matrix_image)
                    Id_rec = self.model.forward_decode(self.Id_z)
                else:
                    self.Id_z, Lnn = self.forward_with_im_fn(Id_cropped, interaction_matrix_image)
                    # self.Id_z, Lnn = L2NormalizeWrapper(L2Normalize()).forward_with_interaction_matrix(self.Id_z, Lnn)
                    Id_rec = torch.zeros_like(Id_cropped)
                    # print(Id_processed.size(), Id_rec.size())
                self.interaction_matrix = Lnn
                
                if self.weighting == Weighting.DecoderError:
                    with torch.enable_grad():
                        self.Id_z.requires_grad = True
                        rec = self.model.forward_decode(self.Id_z)
                        if rec.size()[2:] != Id_cropped.size()[2:]:
                            rec = torch.nn.functional.interpolate(rec, size=Id_cropped.size()[2:], mode='bilinear')
                        bce_loss = torch.nn.BCELoss()
                        loss = bce_loss(rec, Id_cropped)
                        grads = torch.autograd.grad(loss, self.Id_z)[0]
                        abs_grads = torch.abs(grads)
                        abs_grads_inv = 1.0 / abs_grads
                        abs_grads_inv_sum = torch.sum(abs_grads_inv, dim=-1, keepdim=True)
                        self.desired_weighting = abs_grads_inv / abs_grads_inv_sum
                        self.desired_weighting *= abs_grads.size(-1)
            else:
                self.Id_z = self.forward_encode_fn(Id_cropped)
                if self.do_decode:
                    assert False
                Id_rec = torch.zeros_like(Id_cropped)
                self.interaction_matrix = None
            ops = []
            self.Id_z = self.Id_z.view(self.Id_z.size()[0], -1)
            if self.interaction_matrix is not None:
                matrix_size = self.interaction_matrix.size()
                if len(matrix_size) == 5:
                    self.interaction_matrix = self.interaction_matrix.view(matrix_size[0], 6, -1).permute(0, 2, 1)
                ops = [('save_array', (['interaction_matrix.txt'] * self.interaction_matrix.size()[0], self.interaction_matrix.cpu().numpy()), False)]
            # with torch.no_grad():
            #     s = torch.svd(self.interaction_matrix, compute_uv=False)[1][0]
            #     print(s[0] / s[-1])

                # print(torch.mean(abs_grads), torch.std(abs_grads))
                # print(grads)
                # print(grads.size())

            if self.do_decode:
                desired_images_decoded = self.model.unprocess(Id_rec).cpu().numpy()[:, 0]
                Id = self.model.unprocess(Id_processed[:, 0, self.border:-self.border, self.border:-self.border]).cpu().numpy()
                return ops + [('save_rebuilt_desired_image', desired_images_decoded, True),
                        ('save_desired_reconstruction_error_image', (Id, desired_images_decoded), True)]
            else:
                return ops
    def process_image(self, images) -> ImageTorchArray:
        with torch.no_grad():
            img = to_gray(images)
            img = torch.tensor(img, requires_grad=False, device=self.device)
            img = self.model.preprocess(img).unsqueeze(1).float()
            return img
    def compute_vc(self, I_processed, _Id_processed, _iter, run_indices=None, _=None):
        with torch.no_grad():
            b = self.border
            Lnn = None
            I_cropped = I_processed[:, :, b:-b, b:-b].contiguous()
            current_weighting = None
            if self.interaction_matrix_mixer.requires_Li():
                Ix, Iy = self.im_grads(I_processed.squeeze(1))
                interaction_matrix_image = self.Li_fn((Ix, Iy, self.Zinv))
                if self.do_decode:
                    I_z, I_rec, Lnn = self.forward_with_im_fn(I_cropped, interaction_matrix_image)
                else:
                    I_z, Lnn = self.forward_with_im_fn(I_cropped, interaction_matrix_image)
                    # I_z, Lnn = L2NormalizeWrapper(L2Normalize()).forward_with_interaction_matrix(I_z, Lnn)
                    I_rec = torch.zeros_like(I_processed)
                Lnn = permute_im_to_vec_rep_if_required_minimal_checks(Lnn)
                if self.weighting == Weighting.DecoderError:
                    with torch.enable_grad():
                        I_z.requires_grad = True
                        rec = self.model.forward_decode(I_z)
                        if rec.size()[2:] != I_cropped.size()[2:]:
                            rec = torch.nn.functional.interpolate(rec, size=I_cropped.size()[2:], mode='bilinear')
                        bce_loss = torch.nn.BCELoss()
                        loss = bce_loss(rec, I_cropped)
                        grads = torch.autograd.grad(loss, I_z)[0]
                        abs_grads = torch.abs(grads)
                        abs_grads_inv = 1.0 / abs_grads
                        abs_grads_inv_sum = torch.sum(abs_grads_inv, dim=-1, keepdim=True)
                        current_weighting = abs_grads_inv / abs_grads_inv_sum
                        current_weighting *= abs_grads.size(-1)
            else:
                if self.do_decode:
                    I_z = self.model(I_cropped)
                    I_rec = self.forward_decode(I_z)
                else:
                    I_z = self.forward_encode_fn(I_cropped)
                    I_rec = torch.zeros_like(I_cropped)

            I_z = I_z.view(I_z.size()[0], -1)
            desired_IM = self.interaction_matrix
            Id_z = self.Id_z
            if run_indices is not None:
                desired_IM = self.interaction_matrix[run_indices]
                Id_z = self.Id_z[run_indices]
            L = self.interaction_matrix_mixer.compute_final_L(Lnn, desired_IM)
            # print(np.linalg.matrix_rank(L.cpu().numpy()))
            current_images_decoded = self.model.unprocess(I_rec).cpu().numpy()[:, 0]
            error = (I_z - Id_z).view(Id_z.size()[0], -1)
            if self.weighting == Weighting.DecoderError:
                final_weighting = torch.zeros_like(I_z)
                if self.interaction_matrix_mixer.requires_Li():
                    final_weighting += current_weighting
                if self.interaction_matrix_mixer.requires_Lid():
                    final_weighting += self.desired_weighting
                if self.interaction_matrix_mixer.requires_Lid() and self.interaction_matrix_mixer.requires_Li():
                    final_weighting /= 2.0
                final_weighting = torch.diag_embed(final_weighting)
                # print(final_weighting)
                vc = self.optimizer(L, error, final_weighting)
            else:
                vc = self.optimizer(L, error)

            return vc, error.cpu().numpy(), [('save_rebuilt_current_image',  ([iter for _ in range(I_processed.size()[0])], current_images_decoded), True)]
    def compute_features(self, I_processed):
        b = self.border
        return self.model.forward_encode(I_processed[:, :, b:-b, b:-b].contiguous())
    def compute_features_and_interaction_matrix(self, I_processed):
        b = self.border
        Idx, Idy = self.im_grads(I_processed.squeeze(1))
        Li = self.Li_fn((Idx, Idy, self.Zinv))
        return self.model.forward_encode_with_interaction_matrix(I_processed[:, :, b:-b, b:-b].contiguous(), Li)

    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        root_nnimvs_path = Path(globals['nnimvs_models_folder'])

        return lambda args: NNIMVS(
            io.get_or_default('Z', node, defaults),
            root_nnimvs_path / node['model_path'],
            node['name'],
            io.get_or_default('camera_settings', node, defaults),
            args,
            io.get_or_default('border', node, defaults),
            io.get_or_default_val('do_decode', node, False),
            io.optimizer_from_yaml(io.get_or_default('optimizer', node, defaults), device),
            io.interaction_matrix_mixer_from_yaml(io.get_or_default('interaction_matrix_mixer', node, defaults)),
            io.get_or_default_val('weighting', node, Weighting.Identity),
            device
        )

io.model_builders['NNIMVS'] = NNIMVS.make_method_from_yaml
