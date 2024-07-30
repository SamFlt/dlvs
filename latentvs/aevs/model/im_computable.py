import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import aevs.model.im_op_wrapper as iow
from efficientnet_pytorch.utils import MemoryEfficientSwish, Swish


def permute_im_to_image_rep_if_required(xs, L):
    assert len(xs) == 4
    xb, xc, xh, xw = xs
    if len(L.size()) == 3 and L.size() == (xb, xc * xh * xw, 6):
        return L.permute(0, 2, 1).contiguous().view(xb, 6, xc, xh, xw)
    return L
def permute_im_to_vec_rep_if_required(xs, L):
    assert len(xs) == 2
    if len(L.size()) == 5 and L.size()[:2] == (xs[0], 6):
        return L.view(xs[0], 6, -1).permute(0, 2, 1).contiguous()
    return L
def permute_im_to_vec_rep_if_required_minimal_checks(L):
    b = L.size()[0]
    if len(L.size()) == 5 and L.size()[1] == 6:
        return L.view(b, 6, -1).permute(0, 2, 1).contiguous()
    return L

def im_is_in_image_rep(L):
    return len(L.size()) == 5 and L.size()[1] == 6
def im_is_in_vec_rep(L):
    return len(L.size()) == 3 and L.size()[2] == 6

class OpWithInteractionMatrixComputable(nn.Module):
    def __init__(self, op):
        super(OpWithInteractionMatrixComputable, self).__init__()
        self.op = op
    def forward(self, x):
        return self.op(x)

    def forward_with_interaction_matrix(self, x, L):
        pass

class LinearWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        L = permute_im_to_vec_rep_if_required(x.size(), L)
        Ln = torch.matmul(self.op.weight, L)
        return z, Ln
class IdentityWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        return x, L
class ReLUWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= (torch.gt(z, 0.0) * 1.0).unsqueeze(ci)
        return z, L

class LeakyReLUWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= (torch.ge(z, 0.0) * 1.0 + torch.lt(z, 0.0) * self.op.negative_slope).unsqueeze(ci)
        return z, L

class TanhWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        derivative = 1.0 - z ** 2
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= (derivative).unsqueeze(ci)
        return z, L

class SoftplusWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= (torch.sigmoid(x)).unsqueeze(ci)
        return z, L

class Sin(nn.Module):
    def __init__(self, w0=1.0):
        super(Sin, self).__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
class SinWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= torch.cos(self.op.w0 * x).unsqueeze(ci)
        return z, L

class SwishWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        sig_x = torch.sigmoid(x)
        dx = z + sig_x * (1 - z)
        # dx = (sig_x * (1 + x * (1 - sig_x)))
        ci = 1 if im_is_in_image_rep(L) else -1
        L *= dx.unsqueeze(ci)
        return z, L

class SigmoidWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.forward(x)
        ci = 1 if im_is_in_image_rep(L) else -1
        dx = z * (1 - z)
        L *= dx.unsqueeze(ci)
        return z, L

class Conv2DWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        b, c, h, w = x.size()
        L = permute_im_to_image_rep_if_required(x.size(), L)
        z = self.forward(x)
        _, zc, zh, zw = z.size()
        # Ln = self.op(L.view(b * 6, c, h, w)).reshape(b, 6, zc, zh, zw) # Not correct, as it uses bias
        
        Ln = F.conv2d(L.view(b * 6, c, h, w), self.op.weight,
                    stride=self.op.stride, padding=self.op.padding,
                    dilation=self.op.dilation,
                    groups=self.op.groups).reshape(b, 6, zc, zh, zw)
        return z, Ln
class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()
    def forward(self, x):
        n = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / n
class L2NormalizeWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        B, N = x.size()
        L = permute_im_to_vec_rep_if_required_minimal_checks(L)
        n = torch.norm(x, p=2, dim=-1, keepdim=True).view(-1, 1)
        xnormed = x / n
        dzdx = torch.diag_embed(xnormed)
        
        
        Ln = torch.bmm(dzdx, L)
        return xnormed, Ln
class BatchNorm2dWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.op(x)
        xs = x.size()
        L = permute_im_to_image_rep_if_required(x.size(), L)
        if not self.training:
            sqrt_var = torch.sqrt(self.op.running_var + self.op.eps).view(1, 1, -1, 1, 1)
            L /= sqrt_var
            if self.op.affine:
                L *= self.op.weight.view(1, 1, -1, 1, 1)

        else:
            print('BN train Interaction matrix forward: UNTESTED DO NOT USE')
            mu = torch.mean(x, (0, 2, 3), keepdim=True)
            var = torch.mean((x - mu) ** 2, (0, 2, 3), keepdim=True)
            varpeps = var + self.op.eps
            sqrtvareps = torch.sqrt(varpeps)
            dsqrtdvar = 1 / (2 * sqrtvareps)
            dmudr = torch.mean(L, (0, 3, 4)) # mean interaction matrix across batches and width height
            dxmmudr = L - dmudr.view(1, 6, -1, 1, 1)
            assert dmudr.size() == (6, xs[1])
            dvardr = 2 * torch.mean(dxmmudr, (0, 3, 4))
            dsqrtvardr = dvardr * dsqrtdvar.view((1, -1))#torch.matmul(dvardr, dsqrtdvar.view((-1, 1)))
            assert dsqrtvardr.size() == (6, xs[1]), dsqrtvardr.size()

            disqrtvardr = (-1.0 / (varpeps.view((1, -1)))) * dsqrtvardr

            dstatsdr = (x - mu).unsqueeze(1) * disqrtvardr.view(1, 6, xs[1], 1, 1) + (1.0 / sqrtvareps).view(1,1, xs[1], 1, 1) * dxmmudr
            L = dstatsdr * self.op.weight.view(1, 1, -1, 1, 1)
        return z, L
class PoolWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.op(x)
        L = permute_im_to_image_rep_if_required(x.size(), L)
        b, c, h, w = x.size()
        b, zc, zh, zw = z.size()
        Ln = self.op(L.view(b * 6, c, h, w)).view(b, 6, zc, zh, zw)
        return z, Ln

class MaxPool2DWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z, indices = F.max_pool2d(x, kernel_size=self.op.kernel_size, stride=self.op.stride,
                                    padding=self.op.padding, dilation=self.op.dilation,
                                    return_indices=True, ceil_mode=self.op.ceil_mode)
        L = permute_im_to_image_rep_if_required(x.size(), L)
        b, c, h, w = x.size()
        b, zc, zh, zw = z.size()
        indices = indices.unsqueeze(1).expand(-1, 6, -1, -1, -1).view(b, 6, zc, zh * zw)
        Ln = torch.gather(L.view(b, 6, zc, h * w), -1, indices).view(b, 6, zc, zh, zw)
        # Ln = self.op(L.view(b * 6, c, h, w)).view(b, 6, zc, zh, zw)
        return z, Ln

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class FlattenWrapper(OpWithInteractionMatrixComputable):
    def forward_with_interaction_matrix(self, x, L):
        z = self.op(x)
        L = permute_im_to_vec_rep_if_required(z.size(), L)
        return z, L

class SequentialWrapper(OpWithInteractionMatrixComputable):
    def __init__(self, op):
        super(SequentialWrapper, self).__init__(op)
        for i in range(len(self.op)):
            if not isinstance(self.op[i], OpWithInteractionMatrixComputable):
                self.op[i] = iow.OpIMWrapper.from_op(self.op[i])

    def forward_with_interaction_matrix(self, x, L):
        out = x
        for module in self.op:
            out, L = module.forward_with_interaction_matrix(out, L)
        return out, L

class AddCoordsWithIntrinsics(nn.Module):
    '''
    Add Camera coordinates (in meters) to the channels of a tensor
    x = (u - u0) / px
    y = (v - v0) / py
    with u, v in pixels
    See Coord conv
    '''
    def __init__(self, px, py, u0, v0, h, w):
        super(AddCoordsWithIntrinsics, self).__init__()
        self.py = py
        self.px = px
        self.u0 = u0
        self.v0 = v0
        self.h = h
        self.w = w
    def forward(self, x):
        b, c ,h ,w = x.size()
        assert h == self.h and w == self.w, 'Mismatch between expected image size and actual image size, expected {}, got {}'.format((self.h, self.w), (h,w))
        print('This looks wrong!!! check conversion from pixel to meters')
        import sys; sys.exit(0)
        yy = torch.arange(-self.v0, -self.v0 + h, dtype=torch.float, device=x.device) / self.py
        xx = torch.arange(-self.u0, -self.u0 + w, dtype=torch.float, device=x.device) / self.px

        xx = xx.view((1, 1, 1, w)).repeat((b, 1, h, 1))
        yy = yy.view((1, 1, h, 1)).repeat((b, 1, 1, w))

        res = torch.cat((x, xx, yy), dim=1)
        return res
    def convert_for_downsampled_image(self, downsample_factor):
        return AddCoordsWithIntrinsics(self.px / downsample_factor,
                        self.py / downsample_factor,
                        self.u0 / downsample_factor,
                        self.v0 / downsample_factor,
                        self.h // downsample_factor,
                        self.w // downsample_factor)
class AddCoords(nn.Module):
    '''
    Add Camera coordinates (in meters) to the channels of a tensor
    x = (u - u0) / px
    y = (v - v0) / py
    with u, v in pixels
    See Coord conv
    '''
    def __init__(self, h, w):
        super(AddCoords, self).__init__()
        
        self.h = h
        self.w = w
    def forward(self, x):
        b, _ ,h ,w = x.size()
        assert h == self.h and w == self.w, 'Mismatch between expected image size and actual image size, expected {}, got {}'.format((self.h, self.w), (h,w))
        yy = torch.arange(-1.0, 1.0, step=(2.0/h), dtype=torch.float, device=x.device, requires_grad=False)
        xx = torch.arange(-1.0, 1.0, step=(2.0/w), dtype=torch.float, device=x.device, requires_grad=False)

        xx = xx.view((1, 1, 1, w)).repeat((b, 1, h, 1))
        yy = yy.view((1, 1, h, 1)).repeat((b, 1, 1, w))

        res = torch.cat((x, xx, yy), dim=1)
        return res
    def convert_for_downsampled_image(self, downsample_factor):
        return AddCoords(self.h // downsample_factor, self.w // downsample_factor)
class AddCoordsBaseWrapper(OpWithInteractionMatrixComputable):
    '''
    Base wrapper for AddCoords
    '''
    def __init__(self, op):
        super(AddCoordsBaseWrapper, self).__init__(op)
    def forward_with_interaction_matrix(self, x, L):
        b,_,h,w = x.size()
        z = self.op.forward(x)
        L = permute_im_to_image_rep_if_required(x.size(), L)
        Lz = torch.cat((L, torch.zeros((b, 6, 2, h, w), device=L.device)), dim=2)
        return z, Lz

class AddCoordsIntrinsicsBaseWrapper(OpWithInteractionMatrixComputable):
    '''
    Base wrapper for AddCoords
    When computing the interaction matrix,
    expects as input a tuple with the features and the depth (can be a scalar or a tensor) in it.
    The depth is required to compute the Interaction matrices related to the points
    '''
    def __init__(self, op):
        super(AddCoordsIntrinsicsBaseWrapper, self).__init__(op)
    def forward_with_interaction_matrix(self, x_depth, L):
        x, depth = x_depth
        b,c,h,w = x.size()
        L = permute_im_to_image_rep_if_required(x.size(), L)

        x = self.op.forward(x)
        xs = x[0, -2]
        ys = x[0, -1]
        Zinv = 1 / depth
        xsys = xs * ys
        if isinstance(Zinv, float):
            Zinv = torch.ones_like(xs, device=xs.device) * Zinv
        zeros = torch.zeros_like(xs, device=xs.device)
        Lx = [
            -Zinv, zeros, xs * Zinv, xsys, -(1 + xs ** 2), ys
        ]
        Ly = [
            zeros, -Zinv, ys * Zinv, 1 + ys ** 2, -xsys, -xs
        ]
        Lx = torch.cat([Lxi.unsqueeze(0) for Lxi in Lx], dim=0).view(1, 6, 1, h, w).repeat(b, 1, 1, 1, 1)
        Ly = torch.cat([Lyi.unsqueeze(0) for Lyi in Ly], dim=0).view(1, 6, 1, h, w).repeat(b, 1, 1, 1, 1)
        Lres = torch.cat((L, Lx, Ly), dim=2) # Concatenate wrt to channels
        return x, Lres
class AddCoordsWrapperConstantZ(AddCoordsIntrinsicsBaseWrapper):
    def __init__(self, op, Z):
        super(AddCoordsWrapperConstantZ, self).__init__(op)
        self.Z = Z
    def forward_with_interaction_matrix(self, x, L):
        return super(AddCoordsWrapperConstantZ,self).forward_with_interaction_matrix((x, self.Z), L)


iow.OpIMWrapper.register_wrappers({
    nn.Tanh: TanhWrapper,
    nn.ReLU: ReLUWrapper,
    nn.LeakyReLU: LeakyReLUWrapper,
    nn.Softplus: SoftplusWrapper,
    nn.Identity: IdentityWrapper,
    MemoryEfficientSwish: SwishWrapper,
    Swish: SwishWrapper,
    nn.Sigmoid: SigmoidWrapper,
    nn.Linear: LinearWrapper,
    nn.AvgPool2d: PoolWrapper,
    nn.AdaptiveAvgPool2d: PoolWrapper,
    nn.MaxPool2d: MaxPool2DWrapper,
    nn.Conv2d: Conv2DWrapper,
    L2Normalize: L2NormalizeWrapper,
    nn.BatchNorm2d: BatchNorm2dWrapper,
    Flatten: FlattenWrapper,
    nn.Sequential: SequentialWrapper,
    Sin: SinWrapper
})



