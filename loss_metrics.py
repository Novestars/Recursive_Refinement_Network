import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import gpu_utils


def invertible_loss(flow_1, flow_2,reconstruction):
    forward = reconstruction(flow_1, flow_2) + flow_2
    backward = reconstruction(flow_2, flow_1) + flow_1
    ret = ((forward ** 2).sum() / 2 +(backward ** 2).sum() / 2) / torch.prod(torch.FloatTensor(np.array(flow_1.shape[2:])))

    return ret


def NCC(I, J):
    eps = 1e-5
    win_raw = 5
    ndims = 3
    win_size = win_raw
    win = [win_raw] * ndims
    weight_win_size = win_raw
    sgm=2.1
    # weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
    weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)

    conv_fn = F.conv3d
    # compute CC squares
    I2 = I*I
    J2 = J*J
    IJ = I*J
    # compute filters
    # compute local sums via convolution
    I_sum = conv_fn(I, weight, padding=int(win_size/2))
    J_sum = conv_fn(J, weight, padding=int(win_size/2))
    I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
    J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
    IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))
    # compute cross correlation
    win_size = np.prod(win)
    u_I = I_sum/win_size
    u_J = J_sum/win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross * cross / (I_var * J_var + eps)
    # return negative cc.
    mask = torch.logical_and(I_var.detach()>0, J_var.detach()>0)
    cc = cc*mask
    return -1.0 * torch.mean(cc)

class NCC_fft:
    def __init__(self,shape = [256,256,256],sgm=[2.1,2.1,2.1]):
        #shape = I.shape[2:]
        #sgm=[2.1,2.1,2.1]
        shape = torch.tensor(shape)
        sgm = torch.tensor(sgm)
        hsz = 2 * torch.ceil(2 * sgm) + 1
        self.padding_size = (hsz - 1) / 2
        c = torch.ceil(shape/2);
        n1, n2, n3 = torch.meshgrid(
            torch.linspace(0.0, shape[0] - 1.0, int( shape[0])),
            torch.linspace(0.0, shape[1] - 1.0, int( shape[1])),
            torch.linspace(0.0, shape[2] - 1.0, int( shape[2])))

        g = (- (n1 - c[0]).square() / (2 * sgm[0].square()) - (n2 - c[1]).square() / (2 * sgm[1].square()) -
             (n3 - c[2]).square() / (2 * sgm[2].square())).exp()
        g = g/g.sum()
        g = torch.fft.fftshift(g)
        self.fg = torch.fft.fftn(g).to(gpu_utils.device)
    def __call__(self,I,J):
        deps = 1e-4
        mean_def = convop(I,self.fg)
        mean_fix = convop(J,self.fg)
        sgm_def = convop(I.square(),self.fg) - mean_def.square() + deps
        sgm_fix = convop(J.square(),self.fg) - mean_fix.square() + deps

        sprod = convop(I * J,self.fg) - mean_fix * mean_def
        sgm_d_f = sgm_def* sgm_fix

        cc = sprod/sgm_d_f
        mask = torch.logical_and(sgm_def.detach() > 0.02, sgm_fix.detach() > 0.02)
        cc = cc * mask
        return -1.0 * torch.mean(cc)

def convop(fx, fg):
    result = torch.fft.ifftn(torch.fft.fftn(fx)*fg)
    return torch.real(result)


def JacboianDet(y_pred, grid):
    J = y_pred + grid
    J = J.permute(0, 2, 3, 4, 1)
    J = J[..., [2, 1, 0]]
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])
    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet

def neg_Jdet_loss(y_pred,grid ):
    neg_Jdet = -1.0 * JacboianDet(y_pred, grid)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet)
def neg_Jdet(y_pred, grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred,grid)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return selected_neg_Jdet

def regularize_loss2(y_pred):
    # input_size b*3*128*128*128

    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0

def diceLoss(target, predict):
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + 1
    den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + 1

    loss = 1 - num / den
    return loss.mean()

def mask_metrics(seg1, seg2, threshold1 = 0.6, threshold2=0.6):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    b = seg2.shape[0]
    seg1 = torch.reshape(seg1, [b, -1])
    seg2 = torch.reshape(seg2, [b, -1])
    seg1 = (seg1 > threshold1).float()
    seg2 = (seg2 > threshold2).float()
    dice_score = 2.0 * torch.sum(seg1 * seg2, dim=-1) / (
        torch.sum(seg1, dim=-1) + torch.sum(seg2, dim=-1))
    union = torch.sum(torch.max(seg1, seg2), dim=-1)
    return (dice_score, torch.sum(torch.min(seg1, seg2), axis=-1) / torch.clamp(union, min=0.01))

def regularize_loss(flow):
    # input_size b*3*128*128*128

    ret = ((((flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])**2).sum()/2 +
            ((flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])**2).sum()/2 +
            ((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])**2).sum()/2) / torch.prod(torch.FloatTensor(np.array(flow.shape[2:]))))
    return ret
def similarity_loss( img1, warped_img2):
    # input_size b*1*128*128*128
    sizes = torch.prod(torch.IntTensor(np.array(img1.shape[2:])))
    flatten1 = torch.reshape(img1, [-1, sizes])
    flatten2 = torch.reshape(warped_img2, [-1, sizes])

    var1, mean1  = torch.var_mean(flatten1, dim = -1, keepdim=True)
    var2, mean2 = torch.var_mean(flatten2, dim = -1, keepdim=True)

    cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1, keepdim=True)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = torch.sum(raw_loss)
    return raw_loss
def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff
