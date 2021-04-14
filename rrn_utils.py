import torch.nn as nn
import torch
from absl import flags

import gpu_utils
import rrn_flags
from loss_metrics import *

FLAGS = flags.FLAGS

def upsample(img, is_flow, scale_factor=2.0, align_corners=True):
    img_resized = nn.functional.interpolate(img, scale_factor=scale_factor, mode='trilinear',
                                            align_corners=align_corners)

    if is_flow:
        img_resized *= scale_factor

    return img_resized


def flow_to_warp(flow):

    # Construct a grid of the image coordinates.
    B, _, height, width,depth = flow.shape
    k_grid, j_grid, i_grid = torch.meshgrid(
        torch.linspace(0.0, height - 1.0, int(height)),
        torch.linspace(0.0, width - 1.0, int(width)),
        torch.linspace(0.0, depth - 1.0, int(depth)))
    grid = torch.stack([i_grid, j_grid, k_grid]).to(gpu_utils.device)

    # add batch dimension to match the shape of flow.
    grid = grid[None]
    grid = grid.repeat(B, 1, 1, 1,1)

    # Add the flow field to the image grid.
    if flow.dtype != grid.dtype:
        grid = grid.type(dtype=flow.dtype)
    warp = grid + flow
    return warp


def resample(source, coords):
    _, _, H, W,D = source.shape
    # normalize coordinates to [-1 .. 1] range
    coords = coords.clone()
    coords[:, 0, :, :,:] = 2.0 * coords[:, 0, :, :].clone() / max(D - 1, 1) - 1.0
    coords[:, 1, :, :,:] = 2.0 * coords[:, 1, :, :].clone() / max(W - 1, 1) - 1.0
    coords[:, 2, :, :,:] = 2.0 * coords[:, 2, :, :].clone() / max(H - 1, 1) - 1.0
    coords = coords.permute(0, 2, 3, 4, 1)
    output = torch.nn.functional.grid_sample(source, coords, align_corners=False)
    return output


def compute_cost_volume(features1, features2, max_displacement):
    # Set maximum displacement and compute the number of image shifts.
    _, _, height, width, depth = features1.shape
    if max_displacement <= 0 or max_displacement >= height:
        raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed to compute the
    # cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = torch.nn.functional.pad(
        input=features2,
        pad=[max_disp, max_disp, max_disp, max_disp, max_disp, max_disp],
        mode='constant')
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            for k in range(num_shifts):
                prod = features1 * features2_padded[:, :, i:(height + i), j:(width + j),k:(depth+k)]
                corr = torch.mean(prod, dim=1, keepdim=True)
                cost_list.append(corr)
    cost_volume = torch.cat(cost_list, dim=1)
    return cost_volume


def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
    # Compute feature statistics.

    dim = [1, 2, 3, 4] if moments_across_channels else [2, 3, 4]

    means = []
    stds = []

    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=dim, keepdim=True)
        std = torch.std(feature_image, dim=dim, keepdim=True)
        means.append(mean)
        stds.append(std)

    if moments_across_images:
        means = [torch.mean(torch.stack(means), dim=0, keepdim=False)] * len(means)
        stds = [torch.mean(torch.stack(stds), dim=0, keepdim=False)] * len(stds)

    # Center and normalize features.
    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, means)
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, stds)]

    return feature_list


def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
    image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    return image_batch_gh, image_batch_gw



def compute_loss(i1, warped2, flow, use_mag_loss=False):
    loss = torch.nn.functional.l1_loss(warped2, i1)
    if use_mag_loss:
        flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        mag_loss = flow_mag.mean()
        loss += mag_loss * 1e-4

    return loss


def compute_all_loss(f1, warped_f2, flows,ncc=None):
    all_losses = []
    all_losses = [NCC(f1[0], warped_f2[0])]

    B, _, H, W, D = f1[0].shape

    smoothness_weight = FLAGS.weight_smooth1
    if smoothness_weight is not None and smoothness_weight > 0:
        smooth_loss = (smoothness_weight * regularize_loss2(flows[0]))
        all_losses.append(smooth_loss)


    print('lcc loss=%.5f' % (all_losses[0]) +'smoothness loss=%.5f' % all_losses[1] )

    all_losses = torch.stack(all_losses)
    return all_losses.sum()


def _avg_pool3x3(x):
    return torch.nn.functional.avg_pool3d(x, 3, 1)


