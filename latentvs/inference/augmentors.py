import torch
from torchvision.transforms import Compose

from utils.custom_typing import RawRGBImageArray
from utils.torchvision_transforms import ColorJitter, RandomErasing

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class Augmentor():
    def __init__(self):
        pass
    def __call__(self, I: RawRGBImageArray) -> RawRGBImageArray:
        raise NotImplementedError

class IdentityAugmentor(Augmentor):
    def __init__(self):
        pass
    def __call__(self, I: RawRGBImageArray) -> RawRGBImageArray:
        return I


class TorchAugmentor(Augmentor):
    def __init__(self, composer: Compose):
        super().__init__()
        self.composer = composer
    def __call__(self, I: RawRGBImageArray) -> RawRGBImageArray:
        Itorch = torch.from_numpy(I).float().clone() / 255
        Itorch = Itorch.permute(0, 3, 1, 2)
        print(Itorch.size())
        for i in range(Itorch.size(0)):
            Itorch[i] = self.composer(Itorch[i])
        Ip = Itorch.permute(0, 2, 3, 1) * 255
        
        return Ip.cpu().numpy()

def str_to_augmentor(t) -> Augmentor:
    if t is None or t == 'None':
        return IdentityAugmentor()
    if t == 'full':
        augmentation = Compose([
            RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ColorJitter(brightness=0.6, contrast=0.4),
            AddGaussianNoise(0.0, 0.05),
        ])
        return TorchAugmentor(augmentation)
    assert False, 'Unknown Augmentation string'