import torch
from torch import nn
import numpy as np

class DVSInteractionMatrix(nn.Module):
    def __init__(self, h, w, px, py, u0, v0, border):
        super(DVSInteractionMatrix, self).__init__()
        self.px = px
        self.py = py
        self.u0 = u0
        self.v0 = v0
        self.border = border
        self.h = h
        self.w = w
        self.cropped_h = h - self.border * 2
        self.cropped_w = w - self.border * 2
        us = torch.arange(self.border, w - self.border, dtype=torch.float)
        vs = torch.arange(self.border, h - self.border, dtype=torch.float)
        xs = (us - self.u0) / self.px
        ys = (vs - self.v0) / self.py
        self.xs = nn.Parameter(xs.repeat(self.cropped_h), requires_grad=False) # 0, 1, 2, 0, 1, 2
        self.ys = nn.Parameter(ys.repeat_interleave(self.cropped_w), requires_grad=False) # 000, 111, 222..
        
    def forward(self, x):
        Ix, Iy, Zinv = x
        Ix = Ix.view(-1, self.cropped_h * self.cropped_w) * self.px
        Iy = Iy.view(-1, self.cropped_h * self.cropped_w) * self.py
        xs = self.xs
        ys = self.ys
        xsys = xs * ys
        l = [
            Ix * Zinv,
            Iy * Zinv,
            -(xs * Ix + ys * Iy) * Zinv,
            -Ix * xsys - (1.0 + ys ** 2) * Iy,
            (1.0 + xs ** 2) * Ix + Iy * xsys,
            Iy * xs - Ix * ys
        ]
        for i in range(len(l)):
            l[i] = l[i].unsqueeze(-1)
        Ls = torch.cat(l,-1)
        return Ls
class DVSError(nn.Module):
    def __init__(self):
        super(DVSError, self).__init__()
    def forward(self, x):
        I, Id = x
        s = I.size()
        Ivec = I.view(-1, s[1] * s[2])
        Idvec = Id.view(-1, s[1] * s[2])
        return (Ivec - Idvec)

class ImageGradients(nn.Module):
    def __init__(self, border):
        super(ImageGradients, self).__init__()
        self.border = border
        assert self.border > 3
        f = np.array([-112.0, -913.0, -2047.0, 0.0, 2047.0, 913.0, 112.0]) / 8418.0 # Taken from visp
        self.x_filter = nn.Parameter(torch.from_numpy(f.reshape((1, 1, 1, 7))).float(), requires_grad=False)
        self.y_filter = nn.Parameter(torch.from_numpy(f.reshape((1, 1, 7, 1))).float(), requires_grad=False)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        Ix = nn.functional.conv2d(x[:, :, self.border:-self.border,self.border-3:-(self.border-3)], self.x_filter, bias=None)
        Iy = nn.functional.conv2d(x[:, :, self.border-3:-(self.border-3), self.border:-self.border], self.y_filter, bias=None)
        Ix = Ix[:, 0].contiguous()
        Iy = Iy[:, 0].contiguous()
        return Ix, Iy


class DVS(nn.Module):
    def __init__(self, h, w, px, py, u0, v0, border):
        super(DVS, self).__init__()
        self.h = nn.Parameter(torch.tensor(h), requires_grad=False)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=False)
        self.border = nn.Parameter(torch.tensor(border), requires_grad=False)
        self.px = nn.Parameter(torch.tensor(px), requires_grad=False)
        self.py = nn.Parameter(torch.tensor(py), requires_grad=False)
        self.u0 = nn.Parameter(torch.tensor(u0), requires_grad=False)
        self.v0 = nn.Parameter(torch.tensor(v0), requires_grad=False)
        
        self.spatial_gradient = ImageGradients(border)
        self.interaction_matrix = DVSInteractionMatrix(self.h, self.w, self.px, self.py, self.u0, self.v0, border)
        self.dvs_error = DVSError()
        self.eye_6 = nn.Parameter(torch.eye(6), requires_grad=False)
        
    def forward(self, x):
        Is, Ids, Zinv, mu = x
        if len(Zinv.size()) == 3:
            Zinv = Zinv[:, self.border:-self.border, self.border:-self.border].contiguous()
            Zinv = Zinv.view((-1, (self.h - self.border * 2) * (self.w - self.border * 2)))
        Idx, Idy = self.spatial_gradient(Ids)
        # Idx = Idx * self.px
        # Idy = Idy * self.py
        Is = Is[:, self.border:-self.border, self.border:-self.border].contiguous()
        Ids = Ids[:, self.border:-self.border, self.border:-self.border].contiguous()
        
        Lsd = self.interaction_matrix((Idx, Idy, Zinv))
        # print(Lsd)
        LsdT = torch.transpose(Lsd, 1, 2)
        Hsd = torch.bmm(LsdT, Lsd)
        
        diags_vectors = torch.diagonal(Hsd, dim1=-2, dim2=-1)
        eye_repeated = self.eye_6.unsqueeze(0).repeat((Hsd.size()[0], 1, 1))
        diagHsd = eye_repeated * diags_vectors.unsqueeze(1)
        
        H = ((mu.view(-1, 1, 1) * diagHsd) + Hsd).inverse()
        
        e = self.dvs_error([Is, Ids])
        vc = -torch.matmul(torch.matmul(H, LsdT), e.unsqueeze(-1)).squeeze(-1)
        return vc
