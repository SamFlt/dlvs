import yaml
import torch
import torch.nn.functional as F

from inference.interaction_matrix_mixer import *
from inference.utils import *
from inference.optimizers import *
from utils.custom_typing import *

from inference import io



class VSMethod():
    def __init__(self):
        pass
    def name(self) -> str:
        pass
    def process_image(self, images: RawRGBImageArray) -> ImageTorchArray:
        pass
    def on_new_batch(self, batch_idx: UnsignedInt, Id_processed: ImageTorchArray, desired_pose=None) -> None:
        pass
    def on_end_iter(self, iter_idx: UnsignedInt) -> None:
        pass
    def on_iter_begin(self, iter_idx: UnsignedInt) -> None:
        pass
    def compute_vc(self, current_images: ImageTorchArray, desired_images: ImageTorchArray, iter, run_indices=None, current_pose=None) -> Tuple[VelocityArray, VSErrorArray, LoggingActions]:
        pass
    def error_shape(self) -> UnsignedInt:
        pass
    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        pass


class SiameseModelVelocity(VSMethod):
    def __init__(self, model_path, name, use_color, border, _generator_parameters, _vs_arguments):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.use_color = use_color
        self.border = border
        self.model = torch.load(str(model_path), map_location=torch.device(self.device))
        self.model.eval()
        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp
        print('#Parameters = ', get_n_params(self.model))
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#Parameters 2 =', count_parameters(self.model))
        self._name = name
    def process_image(self, images):
        img = images[:, self.border:-self.border, self.border:-self.border]
        with torch.no_grad():
            if self.use_color:
                img = torch.tensor(img, requires_grad=False, device=self.device).permute((0, 3, 1, 2))
            else:
                img = to_gray(img)
                img = torch.tensor(img, requires_grad=False, device=self.device).unsqueeze(1).expand(-1, 3, -1, -1)
            img = self.model.normalize(img).float()
            return img
    def error_shape(self):
        return 1
    def name(self):
        return self._name
    def on_new_batch(self, batch_idx, Id_processed):
        with torch.no_grad():
            self.Id_features = self.model.feature_extractor.extract_features(Id_processed)
            self.Id_features = F.adaptive_avg_pool2d(self.Id_features, (1, 1)).squeeze(-2).squeeze(-1)
            return []
    def compute_vc(self, current_images, desired_images, _iter):
        with torch.no_grad():
            I_features = self.model.feature_extractor.extract_features(current_images)
            I_features = F.adaptive_avg_pool2d(I_features, (1, 1)).squeeze(-2).squeeze(-1)
            error = I_features - self.Id_features
            error = torch.sum(error, dim=1, keepdim=True).cpu().numpy()
            vc = self.model.forward_from_features((I_features, self.Id_features))
            return vc, error, []

class TruePBVS(VSMethod):
    def __init__(self):
        self.cdTo = None
    def name(self) -> str:
        return 'PBVS'
    def process_image(self, images: RawRGBImageArray) -> ImageTorchArray:
        return images
    def on_new_batch(self, batch_idx: UnsignedInt, Id_processed: ImageTorchArray, desired_pose=None) -> None:
        assert desired_pose is not None
        orcd = desired_pose
        oTcd = batch_to_homogeneous_transform_with_axis_angle(orcd[:, :3], orcd[:, 3:])
        self.cdTo = batch_homogeneous_inverse(oTcd)
        return []
    def on_end_iter(self, iter_idx: UnsignedInt) -> None:
        pass
    def on_iter_begin(self, iter_idx: UnsignedInt) -> None:
        pass
    def compute_vc(self, current_images: ImageTorchArray, desired_images: ImageTorchArray, _iter, run_indices=None, current_pose=None) -> Tuple[VelocityArray, VSErrorArray, LoggingActions]:
        orc = current_pose
        oTc = batch_to_homogeneous_transform_with_axis_angle(orc[:, :3], orc[:, 3:])
        cdTc = np.matmul(self.cdTo, oTc)
        cdrc = batch_to_pose_vector(cdTc)

        vc = np.empty((cdTc.shape[0], 6))
        vc[:, :3] = (cdTc[:, :3, :3].transpose(0, 2, 1) @ cdrc[:, :3, None])[..., 0]
        vc[:, 3:] = cdrc[:, 3:]
        return torch.from_numpy(-vc), cdrc, []
    def error_shape(self) -> UnsignedInt:
        return 6
    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        return lambda args: TruePBVS()

io.model_builders['PBVS'] = TruePBVS.make_method_from_yaml
