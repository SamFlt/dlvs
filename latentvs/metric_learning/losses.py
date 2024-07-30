import torch

def loss_distance_image_pose(zI, zP):
    '''
    Loss that minimises the distance between latent representations of images and their associated poses.
    For Images Ij and Ik acquired at poses rj, rk, the distance between Ij and rk should be equal to the distance between rj and rk.
    This loss is backpropagated to both image and pose representations, which leads to coadaptation.
    zI:
        A B x P x Z tensor, where B is batch size, P is the number of images for a single pose and Z is the latent space dimension
    zP:
        A B x Z tensor, the poses' latent representations

    returns a Torch scalar
    '''
    B, P, Z = zI.size()
    zP = zP.view(B, 1, Z).repeat(1, P, 1).view(B * P, Z)
    zI = zI.view(B * P, Z)
    ip_dists = torch.cdist(zI, zP, p=2)
    pp_dists = torch.cdist(zP, zP, p=2)
    return torch.nn.functional.mse_loss(ip_dists, pp_dists)


def image_invariance_loss(zI):
    '''
    minimize the distance between latent representations of images that are acquired at the same pose.
    zIs is 3 dimensional: B x P x Z
    '''
    return torch.mean(torch.cdist(zI, zI, p=2) ** 2)

def metric_learning_loss(zP, dist_matrix):
    '''
    Loss that enforces the distance in latent space to match a ground truth distance (dist_matrix)
    '''
    z_dist = torch.cdist(zP.unsqueeze(0), zP.unsqueeze(0), p=2)[0]
    return torch.nn.functional.mse_loss(z_dist, dist_matrix)
