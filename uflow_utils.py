import torch.nn as nn
import torch
from absl import flags

import gpu_utils
import uflow_flags
from loss_metrics import *

FLAGS = flags.FLAGS

def upsample(img, is_flow, scale_factor=2.0, align_corners=True):
    """Double resolution of an image or flow field.
    Args:
      img: [BCHW], image or flow field to be resized
      is_flow: bool, flag for scaling flow accordingly
    Returns:
      Resized and potentially scaled image or flow field.
    """

    img_resized = nn.functional.interpolate(img, scale_factor=scale_factor, mode='trilinear',
                                            align_corners=align_corners)

    if is_flow:
        # Scale flow values to be consistent with the new image size.
        img_resized *= scale_factor

    return img_resized


def flow_to_warp(flow):
    """Compute the warp from the flow field.
    Args:
      [B, 2, H, W] flow: tf.tensor representing optical flow.
    Returns:
      [B, 2, H, W] The warp, i.e. the endpoints of the estimated flow.
    """

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
    """Resample the source image at the passed coordinates.
    Args:
      source: tf.tensor, batch of images to be resampled.
      coords: [B, 2, H, W] tf.tensor, batch of coordinates in the image.
    Returns:
      The resampled image.
    Coordinates should be between 0 and size-1. Coordinates outside of this range
    are handled by interpolating with a background image filled with zeros in the
    same way that SAME size convolution works.
    """

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
    """Compute the cost volume between features1 and features2.
    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.
    Args:
      features1: tf.tensor of shape [b, h, w, c]
      features2: tf.tensor of shape [b, h, w, c]
      max_displacement: int, maximum displacement for cost volume computation.
    Returns:
      tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
      all displacements.
    """

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
    """Normalizes feature tensors (e.g., before computing the cost volume).
    Args:
      feature_list: list of tf.tensors, each with dimensions [b, c, h, w, d]
      normalize: bool flag, divide features by their standard deviation
      center: bool flag, subtract feature mean
      moments_across_channels: bool flag, compute mean and std across channels
      moments_across_images: bool flag, compute mean and std across images
    Returns:
      list, normalized feature_list
    """

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


def robust_l1(x):
    """Robust L1 metric."""
    return (x ** 2 + 0.001 ** 2) ** 0.5


def compute_loss(i1, warped2, flow, use_mag_loss=False):
    loss = torch.nn.functional.l1_loss(warped2, i1)
    if use_mag_loss:
        flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        mag_loss = flow_mag.mean()
        loss += mag_loss * 1e-4

    return loss


def compute_all_loss(f1, warped_f2, flows,ncc=None):
    all_losses = []
    #all_losses = [compute_loss(i1, f2, f) for (i1, f2, f) in zip(f1, warped_f2, flows)]
    all_losses = [NCC(f1[0], warped_f2[0])]

    B, _, H, W, D = f1[0].shape

    smoothness_weight = FLAGS.weight_smooth1
    if smoothness_weight is not None and smoothness_weight > 0:
        '''img_gx, img_gy = image_grads(upsample(f1[0], is_flow=False, scale_factor=0.5, align_corners=False))

        # abs_fn = lambda x: x ** 2
        abs_fn = lambda x: (x ** 2 + 0.0001)**0.5
        edge_constant = FLAGS.smoothness_edge_constant

        weights_x = torch.exp(-abs_fn(edge_constant * img_gx).mean(dim=1, keepdim=True)) + 0.05
        weights_y = torch.exp(-abs_fn(edge_constant * img_gy).mean(dim=1, keepdim=True)) + 0.05

        # Compute second derivatives of the predicted smoothness.
        flow_gx, flow_gy = image_grads(flows[1])

        # Compute weighted smoothness
        smooth_loss = (
                smoothness_weight *
                ((weights_x * robust_l1(flow_gx)).mean() + (weights_y * robust_l1(flow_gy)).mean()) / 2.0
        )'''
        smooth_loss = (smoothness_weight * regularize_loss2(flows[0]))
        all_losses.append(smooth_loss)

    '''ssim_weight = FLAGS.weight_ssim
    if ssim_weight is not None and ssim_weight > 0:
        ssim_error, avg_weight = weighted_ssim(warped_f2[0], f1[0], torch.ones((B, H, W,D)).to(gpu_utils.device))

        ssim_loss = ssim_weight * (
                (ssim_error * avg_weight).sum() / ((avg_weight).sum() + 1e-16))

        all_losses.append(ssim_loss)'''

    print('raw_loss=%.5f' % (all_losses[0]) +'reg=%.5f' % all_losses[1] )

    all_losses = torch.stack(all_losses)
    #all_losses[0] = all_losses[0]*10
    return all_losses.sum()


def _avg_pool3x3(x):
    return torch.nn.functional.avg_pool3d(x, 3, 1)


def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.
    See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
    difference here is that not all pixels are weighted equally when calculating
    the moments - they are weighted by a weight function.
    Args:
      x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
      y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
      weight: A tf.Tensor of shape [B, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations).
      c1: A floating point number, regularizes division by zero of the means.
      c2: A floating point number, regularizes division by zero of the second
        moments.
      weight_epsilon: A floating point number, used to regularize division by the
        weight.
    Returns:
      A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
      similarity loss oer pixel per channel, and the second, of shape
      [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
      know how much to weigh each pixel in the first tensor. For example, if
      `'weight` was very small in some area of the images, the first tensor will
      still assign a loss to these pixels, but we shouldn't take the result too
      seriously.
    """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                         'likely unintended.')
    weight = weight[:, None, :, :] # expand dim
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
    sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x ** 2 + mu_y ** 2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight