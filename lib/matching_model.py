import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def NormMap2D_to_unNormMap2D(NormMap2D):
    B, C, H, W = NormMap2D.size()
    mapping = torch.zeros_like(NormMap2D)
    # mesh grid
    mapping[:,0,:,:] = (NormMap2D[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (NormMap2D[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise
    idx = mapping[:, 0, :, :] + mapping[:,1,:,:] * W
    idx = idx.type(torch.cuda.LongTensor)
    return idx

#from normalized mapping to unnormalised flow
def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise
    # print("map(normalized)")
    # print(map[:, 0, 3, 5])
    # print("mapping(unnormalized)")
    # print(mapping[:, 0, 3, 5])
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


def unnormalise_and_convert_mapping_to_flow_and_grid(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise
    # print("map(normalized)")
    # print(map[:, 0, 3, 5])
    # print("mapping(unnormalized)")
    # print(mapping[:, 0, 3, 5])
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow, grid

class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(CorrelationVolume, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)  # shape (b,c,h*w)
        # feature_A = feature_A.view(b, c, h*w).transpose(1,2)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # shape (b,h*w,c)
        feature_mul = torch.bmm(feature_B, feature_A)  # shape (b,h*w,h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        # correlation_numpy = correlation_tensor.detach().cpu().numpy()
        return correlation_tensor  # shape (b,h*w,h,w)


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class OpticalFlowEstimator(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimator, self).__init__()

        dd = np.cumsum([128,128,96,64,32])
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(in_channels + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(in_channels + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(in_channels + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(in_channels + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(in_channels + dd[4])

    def forward(self, x):
        # dense net connection
        x = torch.cat((self.conv_0(x), x),1)
        x = torch.cat((self.conv_1(x), x),1)
        x = torch.cat((self.conv_2(x), x),1)
        x = torch.cat((self.conv_3(x), x),1)
        x = torch.cat((self.conv_4(x), x),1)
        flow = self.predict_flow(x)
        return x, flow


class OpticalFlowEstimatorNoDenseConnection(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorNoDenseConnection, self).__init__()
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x = self.conv_4(self.conv_3(self.conv_2(self.conv_1(self.conv_0(x)))))
        flow = self.predict_flow(x)
        return x, flow


# extracted from DGCNet
def conv_blck(in_channels, out_channels, kernel_size=3,
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMDTop(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False, use_cuda=False):
        super().__init__(in_channels, bn)
        chan = [128, 128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], bn=bn)
        self.conv4 = conv_blck(chan[3], chan[4], bn=bn)
        self.final = conv_head(chan[-1])
        if use_cuda:
            self.conv0.cuda()
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.conv4.cuda()
            self.final.cuda()

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        return self.final(x)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask
    # return output

def unNormMap1D_to_NormMap2D(idx_B_Avec, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False,
                    invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
    batch_size, sz = idx_B_Avec.shape
    w = sz // 25
    h = w
    # fs2: width, fs1: height
    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        # XB, YB = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    elif scale == 'positive':
        XA, YA = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        # XB, YB = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    JA, IA = np.meshgrid(range(w), range(h))
    # JB, IB = np.meshgrid(range(w), range(h))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    # XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).contiguous().view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).contiguous().view(1, -1)))
    # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    iA = IA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    jA = JA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    # iB = IB.expand_as(iA)
    # jB = JB.expand_as(jA)

    xA=XA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    yA=YA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

    xA_WTA = xA.contiguous().view(batch_size, 1, h, w)
    yA_WTA = yA.contiguous().view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

    return Map2D_WTA

def unNormMap1D_to_NormMap2D_inLoc(idx_B_Avec,h,w, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False,
                    invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
    batch_size, sz = idx_B_Avec.shape
    
    # fs2: width, fs1: height
    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        # XB, YB = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    elif scale == 'positive':
        XA, YA = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        # XB, YB = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    JA, IA = np.meshgrid(range(w), range(h))
    # JB, IB = np.meshgrid(range(w), range(h))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    # XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).contiguous().view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).contiguous().view(1, -1)))
    # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    iA = IA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    jA = JA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    # iB = IB.expand_as(iA)
    # jB = JB.expand_as(jA)

    xA=XA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    yA=YA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

    xA_WTA = xA.contiguous().view(batch_size, 1, h, w)
    yA_WTA = yA.contiguous().view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

    return Map2D_WTA


def warp_from_NormMap2D(x, NormMap2D):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid


    vgrid = NormMap2D.permute(0, 2, 3, 1).contiguous()
    output = nn.functional.grid_sample(x, vgrid, align_corners=True) #N,C,H,W
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    #
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output*mask
    # return output
def L1_loss(input_flow, target_flow):
    L1 = torch.abs(input_flow-target_flow)
    L1 = torch.sum(L1, 1)
    return L1
def L1_charbonnier_loss(input_flow, target_flow, sparse=False, mean=True, sum=False):

    batch_size = input_flow.size(0)
    epsilon = 0.01
    alpha = 0.4
    L1 = L1_loss(input_flow, target_flow)
    norm = torch.pow(L1 + epsilon, alpha)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        norm = norm[~mask]
    if mean:
        return norm.mean()
    elif sum:
        return norm.sum()
    else:
        return norm.sum()/batch_size


def EPE(input_flow, target_flow, sparse=False, mean=True, sum=False):

    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    # input_flow_np = input_flow.detach().cpu().numpy()
    batch_size = EPE_map.size(0)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/batch_size


def EPE_mask(input_flow, target_flow, mask_num, sparse=False, mean=False, sum=False):
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    # input_flow_np = input_flow.detach().cpu().numpy()
    batch_size = EPE_map.size(0)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return (EPE_map/ mask_num).sum() /batch_size
def multiscaleEPE(Map2D_WTA, Map2D_NET, mask, sparse=False, robust_L1_loss=False, mean=True, sum=False):
    # b, _, h, w = output.size()
    # if sparse:
    #     target_scaled = sparse_max_pool(target, (h, w))
    #
    #     if mask is not None:
    #         mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
    # else:
    #     target_scaled = F.interpolate(target, (h, w), mode='bilinear')

    if mask is not None:
        mask = mask.cuda().detach().byte()

    if robust_L1_loss:
        if mask is not None:
                return L1_charbonnier_loss(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
        else:
            return L1_charbonnier_loss(output, target_scaled, sparse, mean=mean, sum=False)
    else:
        if mask is not None:
            eps = 1
            src_num_fgnd = mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps

            return EPE_mask(Map2D_WTA * mask.float(), Map2D_NET * mask.float(), src_num_fgnd, sparse, mean=mean, sum=sum)

        else:
            return EPE(Map2D_WTA, Map2D_NET, sparse, mean=mean, sum=False)


def generate_NormMap2D_corr4d_WTA(corr4d):
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()
    nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, 1)
    scores_B, index_B = torch.max(nc_B_Avec, dim=1)
    index1D_B = index_B.view(batch_size, -1)
    Map2D = unNormMap1D_to_NormMap2D(index1D_B)  # (B,2,S,S)
    return Map2D

def generate_mask(flow, flow_bw, occ_thresh):
    output_sum = flow + flow_bw
    output_sum = torch.sum(torch.pow(output_sum.permute(0, 2, 3, 1), 2), 3)
    occ_bw = (output_sum > occ_thresh).float()
    mask_bw = 1. - occ_bw

    return mask_bw

def warp_with_mask(x, flo, masked_flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    mask: [B, C, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)


    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # output_img = output * mask
    output_masked = output * masked_flow

    return output_masked