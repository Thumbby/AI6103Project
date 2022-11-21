import os
import numpy as np
import imageio
import time
import torch
from tqdm import tqdm
from collections import defaultdict

from embedder import *
from model import *
from ray_helpers import *

# set the random seed
np.random.seed(0)

to_8_bytes = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def batchify(fc, chunk):
    # Create a version of function that can be used on small batches
    if chunk is None:
        return fc

    def ret(inputs):
        return torch.cat([fc(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fc, embed_fc, embeddirs_fc, netchunk=1024 * 64):
    # Prepares inputs and applies network function.
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fc(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fc(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fc, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def raw2outputs(raw, integration_time, r_dir, raw_noise_std=0, white_background=False):
    '''To transform the result of prediction.
    Args:
        raw: The prediction result.
        integration_time: Times for integration.
        r_dir: The list of each ray's direction.
    Returns:
        rgb_map: The predicted RGB of each ray.
        disp_map: The disparity map.
        r_weight: Sum of the weight along each ray.
        weight_each_color: Weights assigned to each sampled color.
        distance_map: The predicted distance to object.
    '''

    raw2density = lambda raw, dists, act_fc=F.relu: 1. - torch.exp(-act_fc(raw) * dists)

    dists = integration_time[..., 1:] - integration_time[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(r_dir[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    density = raw2density(raw[..., 3] + noise, dists)
    weight_each_color = density * torch.cumprod(
        torch.cat([torch.ones((density.shape[0], 1)), 1. - density + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weight_each_color[..., None] * rgb, -2)

    distance_map = torch.sum(weight_each_color * integration_time, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(distance_map), distance_map / torch.sum(weight_each_color, -1))
    r_weight = torch.sum(weight_each_color, -1)

    if white_background:
        rgb_map = rgb_map + (1. - r_weight[..., None])

    return rgb_map, disp_map, r_weight, weight_each_color, distance_map


def render_batch(r_batch,
                 network_fc,
                 network_query_fc,
                 N_samples,
                 include_raw=False,
                 lindisp=False,
                 perturb=0.,
                 N_importance=0,
                 network_fine=None,
                 white_background=False,
                 raw_noise_std=0., ):
    '''The volumetric rendering progress
    Args:
        r_batch: The list of batch_size.
        network_query_fc: The function for passing queries to network_fc.
        N_samples: The number of different times to sample along each ray.
        include_raw: Include model's raw if True, unprocessed predictions.
        lindisp: Whether sample linearly in inverse depth rather than in depth.
        perturb: 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
        N_importance: The number of additional times to sample along each ray, only passed to network_fine.
        network_fine: fine-tune network with same spec as network_fc.
        white_background: Whether ssume a white background.
    #Returns:
        rgb_map: The predicted RGB of each ray.
        disp_map: The disparity map.
        r_weight: Sum of the weight along each ray.
        raw: The raw predictions from model.
        distance_std: Standard deviation of distances along ray for each sample.
    '''

    N_rays = r_batch.shape[0]
    r_ori, r_dir = r_batch[:, 0:3], r_batch[:, 3:6]
    viewdirs = r_batch[:, -3:] if r_batch.shape[-1] > 8 else None
    bounds = torch.reshape(r_batch[..., 6:8], [-1, 1, 2])
    nearest_distance, farthest_distance = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        integration_time = nearest_distance * (1. - t_vals) + farthest_distance * (t_vals)
    else:
        integration_time = 1. / (1. / nearest_distance * (1. - t_vals) + 1. / farthest_distance * (t_vals))

    integration_time = integration_time.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (integration_time[..., 1:] + integration_time[..., :-1])
        upper = torch.cat([mids, integration_time[..., -1:]], -1)
        lower = torch.cat([integration_time[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(integration_time.shape)

        integration_time = lower + (upper - lower) * t_rand

    pts = r_ori[..., None, :] + r_dir[..., None, :] * integration_time[..., :, None]

    raw = network_query_fc(pts, viewdirs, network_fc)
    rgb_map, disp_map, r_weight, weight_each_color, distance_map = raw2outputs(raw, integration_time, r_dir,
                                                                               raw_noise_std, white_background)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, r_weight

        z_vals_mid = .5 * (integration_time[..., 1:] + integration_time[..., :-1])
        z_samples = fine_sample(z_vals_mid, weight_each_color[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        integration_time, _ = torch.sort(torch.cat([integration_time, z_samples], -1), -1)
        pts = r_ori[..., None, :] + r_dir[..., None, :] * integration_time[..., :, None]

        run_fn = network_fc if network_fine is None else network_fine

        raw = network_query_fc(pts, viewdirs, run_fn)

        rgb_map, disp_map, r_weight, weight_each_color, distance_map = raw2outputs(raw, integration_time, r_dir,
                                                                                   raw_noise_std, white_background)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'r_weight': r_weight}
    if include_raw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb_0'] = rgb_map_0
        ret['disp_0'] = disp_map_0
        ret['acc_0'] = acc_map_0
        ret['distance_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = defaultdict(list)
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_batch(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_and_reshape(chunk, kwargs, rays, sh):

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    k_extract = ['rgb_map', 'disp_map', 'r_weight']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_dict, ret_list


def render_image(H, W, K, chunk=1024 * 32, rays=None, c2w=None,
                 nearest_distance=0., farthest_distance=1., c2w_staticcam=None,
                 **kwargs):
    '''Render each ray
    Args:
        H: Height of image in pixels.
        W: Width of image in pixels.
        K: The camera intrinsic matrix, the shape is [3, 3].
        focal: Focal length of pinhole camera.
        chunk: The maximum number of rays to process simultaneously.
        rays: The origin and direction for each ray in batch.
        c2w: Camera-to-world transformation matrix.
        nearest_distance: The nearest distance for a ray.
        farthest_distance: The farthest distance for a ray.
        c2w_staticcam: Whether use the transformation matrix for camera.
    Returns:
        rgb_map: The predicted RGB of each ray.
        disp_map: The disparity map.
        r_weight: Sum of the weight along each ray.
        extras: The dict with everything returned by render_rays().
    '''

    if c2w is not None:
        r_ori, r_dir = compute_rays(H, W, K, c2w)
    else:
        r_ori, r_dir = rays

    # provide ray directions as input
    viewdirs = r_dir
    if c2w_staticcam is not None:
        # special case to visualize effect of viewdirs
        r_ori, r_dir = compute_rays(H, W, K, c2w_staticcam)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = r_dir.shape  # [..., 3]

    # Generate the ray batch
    r_ori = torch.reshape(r_ori, [-1, 3]).float()
    r_dir = torch.reshape(r_dir, [-1, 3]).float()

    nearest_distance, farthest_distance = nearest_distance * torch.ones_like(
        r_dir[..., :1]), farthest_distance * torch.ones_like(r_dir[..., :1])
    rays = torch.cat([r_ori, r_dir, nearest_distance, farthest_distance], -1)
    rays = torch.cat([rays, viewdirs], -1)

    ret_dict, ret_list = render_and_reshape(chunk, kwargs, rays, sh)
    return ret_list + [ret_dict]


def render_images(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    '''render a set of images given a set of render views

    render_poses: a set of camera views to render
    K: intrinsic camera matrix
    render_kwargs: arguments that are directly pass to the render() function
    savedir: path to save rendered images
    '''
    H, W, focal = hwf

    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render_image(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to_8_bytes(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps
