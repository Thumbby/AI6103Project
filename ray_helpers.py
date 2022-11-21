import torch
import numpy as np


def compute_rays(H, W, K, c2w, if_np=False):
    '''Given an image HxW and the camera parameters K and c2w, compute the set of rays corresponding to the pixels.

    Augments:
        H: image height by pixel
        W: image width by pixel
        K: camera intrinsic matrix, mapping pixel coordinates to camera 3D coordinates
        c2w: camera extrinsic matrix, mapping camera coordinates to world coordinates
        if_np: if true, change return type from tensor to numpy

    Returns:
        r_ori: origins of the rays in world coord
        r_dir: directions of the rays in world coord
    '''
    if not if_np:
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        i = i.transpose(0, 1)
        j = j.transpose(0, 1)
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

        # translate from camera coord system to world coord system
        r_dir = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        r_ori = c2w[:3, -1].expand(r_dir.shape)
        return r_ori, r_dir
    else:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)

        # translate from camera coord system to world coord system
        r_dir = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        r_ori = np.broadcast_to(c2w[:3, -1], np.shape(r_dir))
        return r_ori, r_dir


def compute_pdf(W, det, sample_num):

    W = W + 1e-6  # avoid dividing by 0
    pdf = W / torch.sum(W, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros(cdf[..., :1].size()), cdf], -1)
    if det:
        u = torch.linspace(0., 1., steps=sample_num)
        u = u.expand(list(cdf.shape[:-1]) + [sample_num])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [sample_num])
    return cdf, u


def compute_inverted_cdf(cdf, mid, u):

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros((inds - 1).size(), dtype=inds.dtype), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones(inds.size(), dtype=inds.dtype), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    mid_g = torch.gather(mid.unsqueeze(1).expand(matched_shape), 2, inds_g)
    return cdf_g, mid_g, u


def fine_sample(mid, W, sample_num, det=False):
    '''Implement the second step of the Hierachical sampling strategy: leveraging the outputs
    of the coarse sampling and conduct fine sampling

    W: weights of each segmentation along the ray, computed by coarse sampling
    sample_num: number of samples used by fine sampling
    det: if True, use determined uniform samples, else use stratified random sampling 
    '''

    cdf, u = compute_pdf(W, det, sample_num)

    cdf_g, mid_g, u = compute_inverted_cdf(cdf, mid, u)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones(denom.size(), dtype=denom.dtype), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = mid_g[..., 0] + t * (mid_g[..., 1] - mid_g[..., 0])

    return samples
