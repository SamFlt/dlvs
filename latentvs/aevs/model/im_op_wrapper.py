import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn



class OpIMWrapper():
    # type_to_wrapper = {}
    type_to_wrapper = {
        # nn.Tanh: ic.TanhWrapper,
        # nn.ReLU: ic.ReLUWrapper,
        # nn.LeakyReLU: ic.LeakyReLUWrapper,
        # nn.Linear: ic.LinearWrapper,
        # nn.AvgPool2d: ic.PoolWrapper,
        # nn.AdaptiveAvgPool2d: ic.PoolWrapper,
        # nn.MaxPool2d: ic.PoolWrapper,
        # nn.Conv2d: ic.Conv2DWrapper,
        # nn.BatchNorm2d: ic.BatchNorm2dWrapper,
        # Flatten: ic.FlattenWrapper,
        # nn.Sequential: ic.SequentialWrapper,
    }
    @staticmethod
    def from_op(op):
        if type(op) not in OpIMWrapper.type_to_wrapper:
            assert False, 'Tried to use an operation for which interaction matrix is not defined: ' + str(op)
        return OpIMWrapper.type_to_wrapper[type(op)](op)
    @staticmethod
    def register_wrapper(typ, wraptyp):
        OpIMWrapper.type_to_wrapper[typ] = wraptyp
    @staticmethod
    def register_wrappers(d):
        OpIMWrapper.type_to_wrapper.update(d)
    