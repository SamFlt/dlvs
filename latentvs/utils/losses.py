import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np

from utils.custom_typing import *



# Multi task learning losses

def mtl_loss_fn(task_weights, task_losses):
    sigma2s = torch.exp(-task_weights)
    weights = 1.0 / (2.0 * sigma2s)
    objectives_loss = (weights * task_losses).sum()
    reg_loss = torch.sqrt(sigma2s).prod().log()
    return objectives_loss + reg_loss

def static_weighted_loss(task_weights: List[float], task_losses: Tensor):
    '''
    Combine multiple losses into a single one, by performing a weighted sum
    the final loss is: sum_i task_losses[i] * task_weights[i]
    '''
    return torch.sum(task_losses * task_weights)

# utils


# Metric learning

def metric_learning_loss(zs, t_dists, r_dists, t_range, r_range, scale_clip=4):
    """ Metric learning loss that tries to match the distance in latent space with the a proxy loss of SE(3).
    The loss on SE(3) is defined as d_se3(t1, t2, r1, r2) = ||t1 - t2||_2  / t_range + ||r1 - r2||_2  / r_range
    with t the translation part and r the axis/angle representation of the rotation.

    Args:
        zs (torch.tensor): NxD The latent representation associated to the poses
        t_dists (torch.tensor): An NxN matrix representing the translation distances between the poses
        r_dists (torch.tensor): An NxN matrix representing the rotation distances between the poses
        t_range (torch.tensor): A scaling factor for translation, in meters (if t_range = 0.1m then 0.1m = 1 for the loss scale)
        r_range (torch.tensor): A scaling factor for rotation, in radians
        scale_clip (int, optional): A clipping factor to not consider samples where the distance is too large. Defaults to 4.
            If t_range = 0.1m and r_range = rad(10°) and scale_clip = 2, samples with more than 0.2m or 20° (or a mix of the two) distance will be ignored in the loss.

    Returns:
        torch.tensor: A scalar, the loss value
    """
    bs = zs.size()[0]
    z_dists = torch.pdist(zs)
    indices = torch.triu_indices(bs, bs, 1)
    W = t_dists[indices[0], indices[1]] / t_range + r_dists[indices[0], indices[1]] / r_range
    return torch.mean(((z_dists - W) ** 2) * torch.lt(W, scale_clip))

def metric_learning_loss_by_pair(zs, zds, t_dists, r_dists, t_range, r_range):
    """Same as metric_learning_loss, but instead of working on the whole batch, pairs of associated samples are compared.

    Args:
        zs (torch.tensor): A BxD tensor, the latent representation associated to the "current" poses
        zds (torch.tensor): A BxD tensor, the latent representation associated to the "desired" poses
        t_dists (torch.tensor): A B tensor, translation distances
        r_dists (torch.tensor): A B tensor, rotation distances
        t_range (float): Translation scaling parameter
        r_range (float): Rotation scaling parameter

    Returns:
        torch.tensor: the loss value
    """
    z_dists = F.pairwise_distance(zs, zds, p=2.0)
    W = t_dists / t_range + r_dists / r_range
    # return torch.mean((z_dists - W) ** 2 / (W + 1e-7))
    return F.l1_loss(z_dists, W)

def metric_learning_loss_dct(zs, Is):
    """Metric learning loss that tries to match the distance in the latent space with the distance in the weighted dct space

    We first compute the dct representation of the images Fs = dct_2d(Is)
    Then, we rescale them with an importance matrix M, with M_i,j = 1 / (i + j + 1)
    Then we compute the L2 distances in latent space d(z_i, z_j) and in dct space d(M * F_i, M * F_j)
    for all pairs i,j (with j > i, ie we take the upper triangular part of the distance matrix)
    Finally, we minimise the L2 difference between d(z_i, z_j), d(M * F_i, M * F_j)
    Args:
        zs (torch.tensor): a B x N tensor, containing the latent representations associated to Is
        Is (torch.tensor): a B x 1 x H x W tensor, the raw (preprocessed) images

    Returns:
        torch.tensor: the loss value
    """
    import torch_dct

    Is_dct = torch_dct.dct_2d(Is)
    h, w = Is.size()[-2:]
    M = make_weighting_mask_manhattan((h, w), Is.device)
    Is_dct = (Is_dct * M).contiguous()
    z_dists = torch.pdist(zs)
    dct_dists = torch.pdist(Is_dct.view(-1, h * w))
    return F.mse_loss(z_dists, dct_dists)

def metric_learning_loss_dct_v2(zs, Is):
    """Metric learning loss that tries to match the distance in the latent space with the distance in the weighted dct space
    The weighting is represented by a mask M of dimension HxW
    We first compute the dct representation of the images Fs = dct_2d(Is)
    Then we compute the L2 distances in latent space d(z_i, z_j) and in dct space ||M * (F_i -F_j||_2
    for all pairs i,j (with j > i, ie we take the upper triangular part of the distance matrix)
    Finally, we minimise the L2 difference between d(z_i, z_j), ||M * (F_i -F_j||_2
    Args:
        zs (torch.tensor): a B x N tensor, containing the latent representations associated to Is
        Is (torch.tensor): a B x 1 x H x W tensor, the raw (preprocessed) images

    Returns:
        torch.tensor: the loss value
    """
    import torch_dct


    Is_dct = torch_dct.dct_2d(Is)
    h, w = Is.size()[-2:]
    b = Is.size(0)
    M = make_weighting_mask_manhattan((h, w), Is.device)
    dct_dists = []
    for i in range(b):
        sample_i = Is_dct[i].unsqueeze(0)
        sample_js = Is_dct[i+1:]
        ijs_loss = torch.mean(M * (sample_i - sample_js) ** 2, dim=(1, 2, 3))
        dct_dists.append(ijs_loss)
        # for j in range(i + 1, b):
        #     ij_loss = torch.mean(M * (Is_dct[i] - Is_dct[j]) ** 2)
        #     dct_dists.append(ij_loss.unsqueeze(0))
    dct_dists = torch.cat(dct_dists, 0)
    z_dists = torch.pdist(zs)
    return F.mse_loss(z_dists, dct_dists)

def metric_learning_loss_ratio(zs, t_dists, r_dists, t_range, r_range):
    """A scale invariant metric learning loss on SE(3):
    for 4 samples, i,j,k,l: minimise the difference between ratios ||d_z(z_i, z_j) / d_z(z_k, z_l) - d_p(p_i, p_j)/d_p(p_k, p_l)||_2
    where z are latent representations, p are poses and d_z and d_p are distance functions.

    Args:
        zs (torch.tensor): B x D Latent representations
        t_dists (torch.tensor): B x B matrix, Translation distances
        r_dists (torch.tensor): B x B matrix, Rotation distances
        t_range (float): translation scaling factor
        r_range (float): rotation scaling factor

    Returns:
        torch.tensor: loss value
    """
    z_dists = torch.cdist(zs, zs)
    r, c = z_dists.size()
    tri_indices = torch.triu_indices(r, c, offset=1, device=zs.device) # avoid computing the same thing twice, as d[i][j] = d[j][i], also d[i][i] = 0 so it's useless
    z_dists = z_dists[tri_indices[0], tri_indices[1]]
    se3_dists = t_dists / t_range + r_dists / r_range
    se3_dists = se3_dists[tri_indices[0], tri_indices[1]]
    z_dists = z_dists.view((-1,1))
    se3_dists = se3_dists.view((-1, 1))
    eps = 1e-7
    z_dists_ratios = torch.matmul((z_dists + eps).pow(-1), torch.transpose(z_dists, 0, 1))
    se3_dists_ratios = torch.matmul((se3_dists + eps).pow(-1), torch.transpose(se3_dists, 0, 1))
    return F.mse_loss(z_dists_ratios, se3_dists_ratios)

def metric_learning_loss_log_ratio(zs, t_dists, r_dists, t_range, r_range):
    """A scale invariant metric learning loss on SE(3):
    Taken from: "Deep Metric Learning Beyond Binary Supervision" https://arxiv.org/pdf/1904.09626.pdf
    In the paper, they use an anchor a and two other samples i,j. In here we compare ratios between all pairs (see below)
    for 4 samples, i,j,k,l: minimise the difference between the log ratios ||log(d_z(z_i, z_j) / d_z(z_k, z_l)) - log(d_p(p_i, p_j)/d_p(p_k, p_l))||_2
    where z are latent representations, p are poses and d_z and d_p are distance functions.

    Args:
        zs (torch.tensor): B x D Latent representations
        t_dists (torch.tensor): B x B matrix, Translation distances
        r_dists (torch.tensor): B x B matrix, Rotation distances
        t_range (float): translation scaling factor
        r_range (float): rotation scaling factor

    Returns:
        torch.tensor: loss value
    """
    z_dists = torch.cdist(zs, zs)
    r, c = z_dists.size()
    tri_indices = torch.triu_indices(r, c, offset=1, device=zs.device) # avoid computing the same thing twice, as d[i][j] = d[j][i], also d[i][i] = 0 so it's useless
    z_dists = z_dists[tri_indices[0], tri_indices[1]]
    se3_dists = t_dists / t_range + r_dists / r_range
    se3_dists = se3_dists[tri_indices[0], tri_indices[1]]
    z_dists = z_dists.view((-1,1))
    se3_dists = se3_dists.view((-1, 1))
    eps = 1e-7
    z_dists_log_ratios = torch.log(torch.matmul((z_dists + eps).pow(-1), torch.transpose(z_dists, 0, 1)))
    se3_dists_log_ratios = torch.log(torch.matmul((se3_dists + eps).pow(-1), torch.transpose(se3_dists, 0, 1)))
    return F.mse_loss(z_dists_log_ratios, se3_dists_log_ratios)

def batch_log_ratio_loss(batch_size, z_dists, gt_dists):
    eps = 1e-6
    losses = [] # losses for each anchor
    o =  batch_size - 1 # number of connected samples for an anchor
    ai = 0
    log_z_dists = torch.log(z_dists + eps)
    log_gt_dists = torch.log(gt_dists + eps)
    def compute_ratio(log_dists):
        count = log_dists.size(0)
        log_dists = log_dists.unsqueeze(-1)
        log_dists_rep = log_dists.repeat(count, 1)
        return log_dists_rep.t() - log_dists_rep
    for _ in range(batch_size - 1):
        log_z_dists_a = log_z_dists[ai:ai + o]
        log_gt_dists_a = log_gt_dists[ai:ai + o]

        z_log_ratios = compute_ratio(log_z_dists_a)
        gt_log_ratios = compute_ratio(log_gt_dists_a)

        log_ratio_losses_a = (z_log_ratios - gt_log_ratios) ** 2
        log_ratio_losses_a = log_ratio_losses_a.view((-1,))

        losses.append(log_ratio_losses_a)
        ai = ai + o
        o = o - 1
    losses = torch.cat(losses, 0)
    loss = losses.mean()
    return loss

def metric_learning_loss_log_ratio_v2(zs, t_dists, r_dists, t_range, r_range):
    """A scale invariant metric learning loss on SE(3):
    Taken from: "Deep Metric Learning Beyond Binary Supervision" https://arxiv.org/pdf/1904.09626.pdf
    for 3 samples, a,i,j: minimise the difference between the log ratios ||log(d_z(z_a, z_i) / d_z(z_a, z_j)) - log(d_p(p_a, p_i)/d_p(p_a, p_j))||_2
    where z are latent representations, p are poses and d_z, d_p are distance functions.

    Args:
        zs (torch.tensor): B x D Latent representations
        t_dists (torch.tensor): B x B matrix, Translation distances
        r_dists (torch.tensor): B x B matrix, Rotation distances
        t_range (float): translation scaling factor
        r_range (float): rotation scaling factor

    Returns:
        torch.tensor: loss value
    """
    batch_size = zs.size(0)
    z_dists = torch.pdist(zs)
    tri_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=zs.device)
    se3_dists = t_dists / t_range + r_dists / r_range
    se3_dists = se3_dists[tri_indices[0], tri_indices[1]]
    se3_dists = se3_dists.view((-1,))
    return batch_log_ratio_loss(batch_size, z_dists, se3_dists)

def metric_learning_loss_log_ratio_split_latents(zs, t_dists, r_dists, t_scale, r_scale):
    batch_size = zs.size(0)
    z_dim = zs.size(1)
    zs_t = zs[:, :z_dim//2]
    zs_r = zs[:, z_dim//2:]
    z_t_dists = torch.pdist(zs_t)
    z_r_dists = torch.pdist(zs_r)
    tri_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=zs.device)

    t_dists = t_dists[tri_indices[0], tri_indices[1]].view((-1,))
    r_dists = r_dists[tri_indices[0], tri_indices[1]].view((-1,))
    return batch_log_ratio_loss(batch_size, z_t_dists, t_dists) * t_scale + batch_log_ratio_loss(batch_size, z_r_dists, r_dists) * r_scale

def dct_log_ratio_loss(zs, Is):
    import torch_dct
    batch_size = zs.size(0)
    z_dists = torch.pdist(zs)
    Is_dct = torch_dct.dct_2d(Is)
    h, w = Is.size()[-2:]
    b = Is.size(0)
    M = make_weighting_mask_manhattan((h, w), Is.device)
    dct_dists = []
    for i in range(b):
        sample_i = Is_dct[i].unsqueeze(0)
        sample_js = Is_dct[i+1:]
        ijs_loss = torch.mean(M * (sample_i - sample_js) ** 2, dim=(1, 2, 3))
        dct_dists.append(ijs_loss)
    dct_dists = torch.cat(dct_dists, 0)
    return batch_log_ratio_loss(batch_size, z_dists, dct_dists)





