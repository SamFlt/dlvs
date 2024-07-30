import torch
import torch.nn as nn
import numpy as np
def tanh_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def softplus_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def relu_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def make_leaky_relu_init(leak):
    def init(m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', a=leak)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', a=leak)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init
def linear_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sin_weight_init(m, w0):
    c = 6
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
    bound = np.sqrt(c / fan_in) / w0
    nn.init.uniform_(m.weight, -bound, bound)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
def sin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        sin_weight_init(m, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
def swish_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)