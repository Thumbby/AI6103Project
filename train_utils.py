import os
import numpy as np
import imageio
import csv
import torch

from embedder import *
from model import *
from ray_helpers import *
from train_utils import *
from dataset.load_blender import load_blender_data
from render import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(0)

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def data_loader(args):
    K = None
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    nearest_distance = 2.
    farthest_distance = 6.
    if args.white_background:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]
    return K, farthest_distance, hwf, i_test, i_train, i_val, images, nearest_distance, poses, render_poses


def cast_intrinsics_matrix(K, hwf):
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
    return H, K, W, hwf


def logging(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    return basedir, expname


def random_ray_batching(H, K, W, i_train, images, poses):
    print('get rays')
    rays = np.stack([compute_rays(H, W, K, p, if_np=True) for p in poses[:, :3, :4]], 0)
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:, None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    return rays_rgb


def sample_ray_with_all(N_rand, i_batch, rays_rgb):
    batch = rays_rgb[i_batch:i_batch + N_rand]
    batch = torch.transpose(batch, 0, 1)
    batch_rays, target_s = batch[:2], batch[2]
    i_batch += N_rand
    if i_batch >= rays_rgb.shape[0]:
        print("Shuffle data after an epoch!")
        rand_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        i_batch = 0
    return batch_rays, target_s


def sample_ray_with_one(H, K, N_rand, W, args, i, i_train, images, poses, start):
    img_i = np.random.choice(i_train)
    target = images[img_i]
    target = torch.Tensor(target).to(device)
    pose = poses[img_i, :3, :4]
    if N_rand is not None:
        rays_o, rays_d = compute_rays(H, W, K, torch.Tensor(pose))

        if i < args.precrop_iters:
            dH = int(H // 2 * args.precrop_frac)
            dW = int(W // 2 * args.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
            if i == start:
                print(
                    f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                 -1)

        coords = torch.reshape(coords, [-1, 2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]
    return batch_rays, target_s


def short_circuit(K, args, basedir, expname, hwf, i_test, images, render_kwargs_test, render_poses, start):
    with torch.no_grad():
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_images(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return


def save_testset(K, args, basedir, expname, hwf, i, i_test, images, poses, render_kwargs_test):
    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', poses[i_test].shape)
    with torch.no_grad():
        render_images(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                      gt_imgs=images[i_test], savedir=testsavedir)
    print('Saved test set')


def save_weight_ckpt(basedir, expname, global_step, i, optimizer, render_kwargs_train):
    path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
    torch.save({
        'global_step': global_step,
        'network_fn_state_dict': render_kwargs_train['network_fc'].state_dict(),
        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print('Saved checkpoints at', path)


def render_video(K, args, basedir, expname, hwf, i, render_kwargs_test, render_poses):
    with torch.no_grad():
        rgbs, disps = render_images(render_poses, hwf, K, args.chunk, render_kwargs_test)
    print('Done, saving', rgbs.shape, disps.shape)
    moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)


def record_result(basedir, expname, i, loss, psnr, time_end, time_start):
    result_path = os.path.join(basedir, expname, 'result.csv')
    if os.path.exists(result_path) is False:
        with open(result_path, 'w+', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ['Iter', 'Loss', 'PSNR', 'Time_Cost']
            csv_write.writerow(csv_head)
    if os.path.exists(result_path):
        with open(result_path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            result_row = [i, loss.item(), psnr.item(), time_end - time_start]
            csv_write.writerow(result_row)
