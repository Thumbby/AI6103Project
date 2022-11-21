import os
import numpy as np
import time
import torch
from tqdm import tqdm, trange

from embedder import *
from model import *
from ray_helpers import *
from train_utils import *
from configuration import *
from render import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(0)

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def create_nerf(args):
    # Instantiate NeRF's MLP model.
    embed_fc, input_channel = get_embedder(args.multires, args.i_embed)

    input_channel_views = 0
    embeddirs_fc = None
    embeddirs_fc, input_channel_views = get_embedder(args.multires_views, args.i_embed)
    output_channel = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(args.netdepth, args.netwidth,
                 pos_in=input_channel, views_in=input_channel_views, skip_conn=skips,
                 act_fc=args.activation).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(args.netdepth_fine, args.netwidth_fine,
                          pos_in=input_channel, skip_conn=skips,
                          views_in=input_channel_views,
                          act_fc=args.activation).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fc = lambda inputs, viewdirs, network_fc: run_network(inputs, viewdirs, network_fc,
                                                                        embed_fc=embed_fc,
                                                                        embeddirs_fc=embeddirs_fc,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fc': network_query_fc,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fc': model,
        'white_background': args.white_background,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train():
    time_start = time.time()
    time_end = time.time()

    parser = config_parser()
    args = parser.parse_args()

    K, farthest_distance, hwf, i_test, i_train, i_val, images, nearest_distance, poses, render_poses = data_loader(args)

    H, K, W, hwf = cast_intrinsics_matrix(K, hwf)

    if args.render_test:
        render_poses = np.array(poses[i_test])

    basedir, expname = logging(args)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    # create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.N_iters))

    global_step = start

    bds_dict = {
        'nearest_distance': nearest_distance,
        'farthest_distance': farthest_distance,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        return short_circuit(K, args, basedir, expname, hwf, i_test, images, render_kwargs_test, render_poses, start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        rays_rgb = random_ray_batching(H, K, W, i_train, images, poses)
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = args.N_iters + 1

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch_rays, target_s = sample_ray_with_all(N_rand, i_batch, rays_rgb)

        else:
            # Random from one image
            batch_rays, target_s = sample_ray_with_one(H, K, N_rand, W, args, i, i_train, images, poses,
                                                       start)

        rgb, disp, acc, extras = render_image(H, W, K, chunk=args.chunk, rays=batch_rays, include_raw=True,
                                              **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb_0' in extras:
            img_loss0 = img2mse(extras['rgb_0'], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        # update learning rate
        if args.scheduler == 'cosine':
            scheduler.step()

        elif args.scheduler == 'default':
            decay_rate = 0.1
            new_lrate = args.lrate * (decay_rate ** (global_step / N_iters))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            save_weight_ckpt(basedir, expname, global_step, i, optimizer, render_kwargs_train)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            render_video(K, args, basedir, expname, hwf, i, render_kwargs_test, render_poses)

        if i % args.i_testset == 0 and i > 0:
            save_testset(K, args, basedir, expname, hwf, i, i_test, images, poses, render_kwargs_test)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            # Record the result of trainning
            time_end = time.time()

            record_result(basedir, expname, i, loss, psnr, time_end, time_start)

            time_start = time.time()

        global_step += 1


if __name__ == '__main__':
    train()
