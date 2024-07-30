import torch
import numpy as np
def qr_inverse(L: torch.Tensor) -> torch.Tensor:
    Q, R = torch.qr(L, some=False)

    Qt = Q.transpose(1, 2)
    Rinv = torch.inverse(R)

    return torch.bmm(Rinv, Qt)
def lu_inverse(L: torch.Tensor) -> torch.Tensor:
    L, U = torch.lu()


def lm_opt(Lz, e, mu):
    with torch.no_grad():
        LzT = torch.transpose(Lz, 1, 2)
        Hz = torch.bmm(LzT, Lz)
        diags_vectors = torch.diagonal(Hz, dim1=-2, dim2=-1)
        eye_repeated = torch.eye(6, device=Lz.device).unsqueeze(0).repeat((Hz.size()[0], 1, 1))
        diagHz = eye_repeated * diags_vectors.unsqueeze(1)
        
        Htmp = ((mu.view(-1, 1, 1) * diagHz) + Hz)
        # H = torch.from_numpy(np.linalg.pinv(Htmp)).to(Lz.device)
        H = torch.pinverse(Htmp).to(Lz.device)
        #H = qr_inverse(Htmp).to(Lz.device)
        # H = ((mu.view(-1, 1, 1) * diagHz) + Hz).inverse()
        # print(torch.svd(torch.matmul(H, LzT), compute_uv=False)[1])
        vc = -torch.matmul(torch.matmul(H, LzT), e.unsqueeze(-1)).squeeze(-1)
        return vc

def weighted_lm_opt(Lz, e, mu, weighting):
    LzT = torch.transpose(Lz, 1, 2)
    Hz = torch.bmm(weighting, Lz)
    Hz = torch.bmm(LzT, Hz)
    diags_vectors = torch.diagonal(Hz, dim1=-2, dim2=-1)
    eye_repeated = torch.eye(6, device=Lz.device).unsqueeze(0).repeat((Hz.size()[0], 1, 1))
    diagHz = eye_repeated * diags_vectors.unsqueeze(1)
    H = torch.pinverse((mu.view(-1, 1, 1) * diagHz) + Hz)
    e = torch.bmm(weighting, e.unsqueeze(-1))
    vc = -torch.matmul(torch.matmul(H, LzT), e).squeeze(-1)
    return vc
    
def linear_opt(Lz, e):
    Lzt = torch.pinverse(Lz)
    # print(torch.svd(Lz[0])[1])
    vc = -torch.matmul(Lzt, e.unsqueeze(-1)).squeeze(-1)
    return vc
def weighted_linear_opt(Lz, e, weighting):
    WLz = torch.bmm(weighting, Lz)
    WLzt = torch.transpose(WLz, 1, 2)

    e = torch.bmm(weighting, e.unsqueeze(-1))
    vc = -torch.matmul(WLzt, e).squeeze(-1)
    return vc

class Optimizer():
    '''
    Optimizer class for visual servoing. It will be used to compute the velocity at each timestep of servoing.
    It takes as input the interaction matrix (dz/dr) and the error
    '''
    def __init__(self):
        pass
    def reset(self):
        pass
    def __call__(self, L, e, W=None):
        pass
    def on_iter(self, iter):
        pass

class LinearOptimizer(Optimizer):
    '''
    Basic optimizer which solves:
    vc = -L^+ e
    with L^+ the pseudo inverse of L, the interaction matrix
    '''
    def __call__(self, L, e, W=None):
        if W is None:
            return linear_opt(L, e)
        else:
            return weighted_linear_opt(L, e, W)

class LevenbergMarquardtOptimizer(Optimizer):
    '''
    Levenberg Marquardt based optimization.
    Used in the paper in DVS, DCT-VS and PCA-VS
    '''
    def __init__(self, mu_initial, iterGN, mu_factor, mu_min, device):
        self.device = device
        self.mu_initial = mu_initial
        self.iterGN = iterGN
        self.mu_factor = mu_factor
        self.mu_min = torch.tensor(mu_min, requires_grad=False, device=self.device)
        self.reset()
    def reset(self):
        self.mu = torch.tensor(self.mu_initial, requires_grad=False, device=self.device)

    def __call__(self, L, e, W=None):
        if W is None:
            return lm_opt(L, e, self.mu)
        else:
            return weighted_lm_opt(L, e, self.mu, W)
    def on_iter(self, iter):
        if iter > self.iterGN:
            self.mu = torch.max(self.mu_min, self.mu * self.mu_factor)

class AdaptiveLevenbergMarquardtOptimizer(Optimizer):
    '''
    Levenberg Marquardt based optimization.
    Used in the paper in DVS, DCT-VS and PCA-VS
    '''
    def __init__(self, mu_initial, mu_up_factor, mu_down_factor, device):
        self.device = device
        self.mu_initial = mu_initial
        self.mu_up_factor = mu_up_factor
        self.mu_down_factor = mu_down_factor
        self.reset()
    def reset(self):
        self.mu = None
        self.previous_e = None

    def __call__(self, L, e, W=None):
        if self.mu is None:
            self.mu = torch.tensor([self.mu_initial for _ in range(e.size(0))], requires_grad=False, device=self.device)
            self.previous_e = e.detach().clone()
        else:
            essd = torch.sum(e, dim=1) ** 2
            pessd = torch.sum(self.previous_e, dim=1) ** 2
            error_lt = torch.lt(essd, pessd)
            error_gt = torch.gt(essd, pessd)

            self.mu = self.mu + (self.mu * (self.mu_up_factor - 1.0)) * error_gt - (self.mu * (1.0 - self.mu_down_factor)) * error_lt
        res = lm_opt(L, e, self.mu) if W is None else weighted_lm_opt(L, e, self.mu, W) 
        return res
    def on_iter(self, iter):
        pass
