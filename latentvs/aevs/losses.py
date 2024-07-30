import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np

from utils.custom_typing import *


def make_weighting_mask_manhattan(size, device):
    row = torch.arange(size[0], dtype=torch.float, device=device).unsqueeze(-1)
    col = torch.arange(size[1], dtype=torch.float, device=device).unsqueeze(0)
    manhattan = row + col
    w = 1.0 / (manhattan + 1)
    return w.view(1, 1, *w.size())

def smoothing_filter_loss(conv_weights_list):
    """Regularization loss, that seeks to smooth the learned filters by minimising their spatial gradients
    In this version, the filters are divided by the norm of the weights to make it insensitive to the weight scale

    Args:
        conv_weights_list (List[torch.nn.Parameter]): The convolution filters on which to apply the loss

    Returns:
        torch.tensor: The loss value
    """
    device = conv_weights_list[0].device
    loss = torch.tensor(0.0, requires_grad=True, device=device)
    dy_filter = torch.tensor([[-1], [1]], requires_grad=False, device=device)
    dy_filter = dy_filter.view(1, 1, *dy_filter.size())
    dx_filter = torch.tensor([[-1, 1]], requires_grad=False, device=device)
    dx_filter = dx_filter.view(1, 1, *dx_filter.size())
    filter_count = 0
    for weight in conv_weights_list:
        if weight.size(-2) == 1 or weight.size(-1) == 1:
            continue
        inp, oup = weight.size(1), weight.size(0)
        w_norms = torch.norm(weight, dim=[-2, -1], keepdim=True).detach_()
        filter_count += inp * oup
        applied_dx_filter = (dx_filter / w_norms).view(1, inp * oup, *dx_filter.size()[-2:])
        applied_dy_filter = (dy_filter / w_norms).view(1, inp * oup, *dy_filter.size()[-2:])
        w_input = weight.view(1, inp * oup, *weight.size()[-2:])
        dx_grads = nn.functional.conv2d(w_input, applied_dx_filter)
        dy_grads = nn.functional.conv2d(w_input, applied_dy_filter)
        weight_loss = (torch.sum(dx_grads ** 2) + torch.sum(dy_grads ** 2)) / (2.0)
        loss = loss + weight_loss
    return loss / filter_count

def smoothing_filter_loss_v2(conv_weights_list):
    """Regularization loss, that seeks to smooth the learned filters by minimising their spatial gradients
    In this version, The weights are normalized before applying the spatial gradient filters

    Args:
        conv_weights_list (List[torch.nn.Parameter]): The convolution filters on which to apply the loss

    Returns:
        torch.tensor: The loss value
    """
    device = conv_weights_list[0].device

    dy_filter = torch.tensor([[-1], [1]], requires_grad=False, device=device, dtype=torch.float)
    dy_filter = dy_filter.view(1, 1, *dy_filter.size())
    dx_filter = torch.tensor([[-1, 1]], requires_grad=False, device=device, dtype=torch.float)
    dx_filter = dx_filter.view(1, 1, *dx_filter.size())
    loss_tensors = []
    for weight in conv_weights_list:
        if weight.size(-2) == 1 or weight.size(-1) == 1:
            continue
        inp, oup = weight.size(1), weight.size(0)
        w_input = weight.view(inp * oup, 1, *weight.size()[-2:]) # CO * CI X 1 X H X W

        w_norms = torch.norm(w_input, p=2, dim=[-3,-2, -1], keepdim=True).detach_() # CO * CI
        w_input = w_input / w_norms # Normalize filters: we want to be norm independent!
        dx_grads = nn.functional.conv2d(w_input, dx_filter) # CO * CI X 1 X H x W - 1
        dy_grads = nn.functional.conv2d(w_input, dy_filter) # CO * CI X 1 X H - 1 x W
        dx_grad_sum = torch.sum(dx_grads ** 2,  dim=[-3, -2, -1]) # CO * CI
        dy_grad_sum = torch.sum(dy_grads ** 2,  dim=[-3, -2, -1]) # CO * CI
        weight_loss = (dx_grad_sum + dy_grad_sum) / (2.0)
        loss_tensors.append(weight_loss)

    return torch.mean(torch.cat(loss_tensors, dim=0))


def smoothing_filter_loss_v3(conv_weights_list):
    """Regularization loss, that seeks to smooth the learned filters by minimising their spatial gradients
    In this version, the filters are by the (average norm of the output + average norm of input)
    average of output for a CI x CO x H x W = mean(dim=0)

    Args:
        conv_weights_list (List[torch.nn.Parameter]): The convolution filters on which to apply the loss

    Returns:
        torch.tensor: The loss value
    """
    device = conv_weights_list[0].device

    dy_filter = torch.tensor([[-1], [1]], requires_grad=False, device=device, dtype=torch.float)
    dy_filter = dy_filter.view(1, 1, *dy_filter.size())
    dx_filter = torch.tensor([[-1, 1]], requires_grad=False, device=device, dtype=torch.float)
    dx_filter = dx_filter.view(1, 1, *dx_filter.size())
    loss_tensors = []
    for weight in conv_weights_list:
        if weight.size(-2) == 1 or weight.size(-1) == 1:
            continue
        inp, oup = weight.size(1), weight.size(0)
        w_filter_norms = torch.norm(weight, p=2, dim=[2, 3], keepdim=True)
        out_avg_norms = torch.mean(w_filter_norms, dim=[1], keepdim=True)
        in_avg_norms = torch.mean(w_filter_norms, dim=[0], keepdim=True)
        w_reweighting = (out_avg_norms + in_avg_norms) / 2.0
        w_reweighting = w_reweighting.view(inp * oup, 1, 1, 1)
        w_input = weight.view(inp * oup, 1, *weight.size()[-2:]) # CO * CI X 1 X H X W

        w_input = w_input / w_reweighting # Normalize filters: we want to be norm independent!
        dx_grads = nn.functional.conv2d(w_input, dx_filter) # CO * CI X 1 X H x W - 1
        dy_grads = nn.functional.conv2d(w_input, dy_filter) # CO * CI X 1 X H - 1 x W
        dx_grad_sum = torch.sum(dx_grads ** 2,  dim=[-3, -2, -1]) # CO * CI
        dy_grad_sum = torch.sum(dy_grads ** 2,  dim=[-3, -2, -1]) # CO * CI
        weight_loss = (dx_grad_sum + dy_grad_sum) / (2.0)
        loss_tensors.append(weight_loss)

    return torch.mean(torch.cat(loss_tensors, dim=0))


def make_dct_filter_smoothing(remove_ratio, sizes=[3, 7], device='cuda'):
    """Regularization loss, that seeks to smooth the learned filters by minimising a weighted version of the frequency representation of the filters
    The weighting places more emphasis on the high frequencies which should be closer to zero (in order to get a smooth filter). The high frequencies are in the lower right part of the matrix

    Args:
        conv_weights_list (List[torch.nn.Parameter]): The convolution filters on which to apply the loss
        sizes (List[float]): kernel sizes on which the DCT decomposition and the minimisation will be applied
        device (str): The device on which to perform the computation

    Returns:
        torch.tensor: The loss value
    """
    def make_zigzag_mask_inverse(size):
            M = np.zeros(size)
            c = size[0] * size[1]
            def get_diagonal_indices(start_pos):
                i, j = start_pos
                m = min(i, size[1] - j -1)
                ri = list(range(i, i - m - 1, -1))
                rj = list(range(j, j + m + 1))
                return ri, rj


            reverse = False
            indices_i, indices_j = [], []
            for i in range(0, size[0]):
                indi, indj = get_diagonal_indices((i, 0))
                if reverse:
                    indi.reverse()
                    indj.reverse()
                reverse = not reverse
                indices_i.extend(indi)
                indices_j.extend(indj)
            for j in range(1, size[0]):
                indi, indj = get_diagonal_indices((size[0] -1, j))
                if reverse:
                    indi.reverse()
                    indj.reverse()
                reverse = not reverse
                indices_i.extend(indi)
                indices_j.extend(indj)

            weight = None
            to_be_removed = int(remove_ratio * c)
            weight = np.array([1 if i >= (c - to_be_removed) else 0 for i in range(c)])
            M[(indices_i, indices_j)] = weight
            M = torch.from_numpy(M).to(device).view(1, 1, *size)
            return M
    masks = { x: make_zigzag_mask_inverse((x, x)) for x in sizes}
    def dct_filter_smoothing(conv_weights_list, masks):
        import torch_dct
        loss_tensors = []
        for weight in conv_weights_list:
            if weight.size(-2) == 1 or weight.size(-1) == 1:
                continue

            mask = masks[weight.size()[-2]]
            dct_w = torch_dct.dct_2d(weight)
            weight_loss = torch.sum((dct_w * mask) ** 2).view((1))
            loss_tensors.append(weight_loss)
        return torch.mean(torch.cat(loss_tensors, dim=0))
    return lambda conv_weights: dct_filter_smoothing(conv_weights, masks)

def make_ae_dct_loss_zigzag(size, keep_count, use_inverse_weighting, device):
    """Autoencoder loss "factory", that creates a loss minimising the difference between the reconstruction and target images, in the frequential space.
    This minimisation is weighted by a mask M that follows a zigzag pattern: more importance is given to the lower frequencies (upper left) and the higher frequencies are discarded

    Args:
        size ((unsigned, unsigned)): The size of the image and reconstruction
        keep_count (unsigned): Number of elements of the zigzag pattern to keep for the loss minimisation: if keep_count = 100, all the weights associated to elements of the zigzag with index > 100 will be set to zero
        use_inverse_weighting (bool): whether to scale the elements of the zigzag pattern by their position. If True, M_i,j = 1 / (zigzag_index(i, j) + 1). Otherwise, M_i,j = 1. In both cases, if zigzag_index(i,j) > keep_count then M_i,j = 0
        device (str: cuda|cpu): The device
    """
    def make_zigzag_mask(size, keep_count):
        M = np.zeros(size)
        c = size[0] * size[1]
        def get_diagonal_indices(start_pos):
            i, j = start_pos
            m = min(i, size[1] - j -1)
            ri = list(range(i, i - m - 1, -1))
            rj = list(range(j, j + m + 1))
            return ri, rj


        reverse = False
        indices_i, indices_j = [], []
        for i in range(0, size[0]):
            indi, indj = get_diagonal_indices((i, 0))
            if reverse:
                indi.reverse()
                indj.reverse()
            reverse = not reverse
            indices_i.extend(indi)
            indices_j.extend(indj)
        for j in range(1, size[0]):
            indi, indj = get_diagonal_indices((size[0] -1, j))
            if reverse:
                indi.reverse()
                indj.reverse()
            reverse = not reverse
            indices_i.extend(indi)
            indices_j.extend(indj)

        weight = None
        if use_inverse_weighting:
            weight = np.array([1/(i+1) if i < keep_count else 0 for i in range(c)])
        else:
            weight = np.array([1 if i < keep_count else 0 for i in range(c)])
        M[(indices_i, indices_j)] = weight
        M = torch.from_numpy(M).to(device).view(1, 1, *size)
        return M
    mask = make_zigzag_mask(size, keep_count)
    def ae_dct_loss(rec, target, mask):
        import torch_dct

        target_dct = torch_dct.dct_2d(target, norm='ortho')
        rec_dct = torch_dct.dct_2d(rec, norm='ortho')
        freq_squared_diff = (rec_dct - target_dct) ** 2

        lower_freqs_diff = freq_squared_diff * mask
        higher_freqs_rec = torch.masked_select(rec_dct, mask == 0.0) ** 2
        return torch.mean(lower_freqs_diff) + torch.mean(higher_freqs_rec)
    return lambda rec, target: ae_dct_loss(rec, target, mask)

def make_ae_dct_loss_manhattan(size, distance, device):
    """Autoencoder loss "factory", that creates a loss minimising the difference between the reconstruction and target images, in the frequential space.
    This minimisation is weighted by a mask M that follows a hamming pattern: The higher the L1 distance of (i,j) with (0, 0) the less weight is given to it. M_0,0 = 1

    Args:
        size ((unsigned, unsigned)): The size of the image and reconstruction
        distance: (str: l1|l2): the loss function to use when comparing the reconstruction and target images.
        device (str: cuda|cpu): The device
    """
    mask = make_weighting_mask_manhattan(size, device)
    distance_fn = None
    if distance == 'l2':
        distance_fn = lambda x,y: (x - y) ** 2
    elif distance == 'l1':
        distance_fn = lambda x,y: torch.abs(x - y)
    def ae_dct_loss(rec, target, mask, distance_fn):
        import torch_dct
        target_dct = torch_dct.dct_2d(target)
        rec_dct = torch_dct.dct_2d(rec)
        freq_diff = distance_fn(rec_dct, target_dct)
        weighted_diff = freq_diff * mask
        return torch.mean(weighted_diff)
    return lambda rec, target: ae_dct_loss(rec, target, mask, distance_fn)