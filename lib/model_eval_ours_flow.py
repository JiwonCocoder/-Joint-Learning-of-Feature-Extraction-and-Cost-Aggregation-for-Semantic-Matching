from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
import pickle

from lib.torch_util import Softmax1D
from lib.conv4d import Conv4d
from lib.matching_model import CMDTop
from lib.matching_model import unNormMap1D_to_NormMap2D, unnormalise_and_convert_mapping_to_flow
from lib.showPlot import plot_test_map, plot_test_flow
import torch.nn.functional as F


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='resnet101', feature_extraction_model_file='',
                 normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn = feature_extraction_cnn
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
        # for resnet below
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            if last_layer == '':
                last_layer = 'layer3'
            resnet_module_list = [getattr(self.model, l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index(last_layer)
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])

        if feature_extraction_cnn == 'resnet101fpn':
            if feature_extraction_model_file != '':
                resnet = models.resnet101(pretrained=True)
                # swap stride (2,2) and (1,1) in first layers (PyTorch ResNet is slightly different to caffe2 ResNet)
                # this is required for compatibility with caffe2 models
                resnet.layer2[0].conv1.stride = (2, 2)
                resnet.layer2[0].conv2.stride = (1, 1)
                resnet.layer3[0].conv1.stride = (2, 2)
                resnet.layer3[0].conv2.stride = (1, 1)
                resnet.layer4[0].conv1.stride = (2, 2)
                resnet.layer4[0].conv2.stride = (1, 1)
            else:
                resnet = models.resnet101(pretrained=True)
            resnet_module_list = [getattr(resnet, l) for l in resnet_feature_layers]
            conv_body = nn.Sequential(*resnet_module_list)
            self.model = fpn_body(conv_body,
                                  resnet_feature_layers,
                                  fpn_layers=['layer1', 'layer2', 'layer3'],
                                  normalize=normalization,
                                  hypercols=True)
            if feature_extraction_model_file != '':
                self.model.load_pretrained_weights(feature_extraction_model_file)

        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if train_fe == False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization and not self.feature_extraction_cnn == 'resnet101fpn':
            features = featureL2Norm(features)
        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=False):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output

    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(0))
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


class find_correspondence(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence, self).__init__()
        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        print("find_correspondence_gpu:",device)
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,
                                                                                                                  1,
                                                                                                                  3).to(
            device)
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).to(
            device)

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)

        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1, feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        self.y_normal = np.linspace(-1, 1, feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def softmax_with_temperature(self, x, beta, d=1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(beta * x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def kernel_soft_argmax(self, corr):
        b, _, h, w = corr.size()

        # corr = self.apply_gaussian_kernel(corr, sigma=self.kernel_sigma)
        corr = self.softmax_with_temperature(corr, beta=self.beta, d=1)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask  # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def forward(self, corr, GT_mask=None):
        b, _, h, w = corr.size()
        grid_X = self.grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)
        if self.beta is not None:
            grid_x, grid_y = self.kernel_soft_argmax(corr)
        else:  # discrete argmax
            _, idx = torch.max(corr, dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w - 1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h - 1) - 0.5) * 2
            grid_x = grid_x.unsqueeze(1)  # b x 1 x h x w
            grid_y = grid_y.unsqueeze(1)

        grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)),
                         3)
        # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),
                         1)  # 2-channels@1st-dim, first channel for x / second channel for y

        if GT_mask is None:  # test
            return grid.permute(0, 3, 1, 2), flow.permute(0, 3, 1, 2)
        else:  # train
            smoothness = self.get_flow_smoothness(flow, GT_mask)
            return grid, flow, smoothness


class ImMatchNet(nn.Module):
    def __init__(self,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 return_correlation=False,
                 ncons_kernel_sizes=[3, 3, 3],
                 ncons_channels=[10, 10, 1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 relocalization_k_size=0,
                 half_precision=False,
                 checkpoint=None,
                 ):

        super(ImMatchNet, self).__init__()
        # Load checkpoint
        if checkpoint is not None and checkpoint is not '':
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
            # override relevant parameters
            print('Using checkpoint parameters: ')
            ncons_channels = checkpoint['args'].ncons_channels
            print('  ncons_channels: ' + str(ncons_channels))
            ncons_kernel_sizes = checkpoint['args'].ncons_kernel_sizes
            print('  ncons_kernel_sizes: ' + str(ncons_kernel_sizes))
        self.ReLU = nn.ReLU()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)

        self.FeatureCorrelation = FeatureCorrelation(shape='4D', normalization=False)

        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)
        feature_H = 25
        feature_W = 25
        beta = 50
        kernel_sigma = 5
        self.find_correspondence = find_correspondence(feature_H, feature_W, beta, kernel_sigma)
        # nd = 25 * 25  # global correlation
        # od = nd + 2
        # batch_norm = True
        # self.decoder4 = CMDTop(in_channels=od, bn=batch_norm, use_cuda=self.use_cuda)
        # Load weights
        if checkpoint is not None and checkpoint is not '':
            print('Copying weights...')
            for name, param in self.FeatureExtraction.state_dict().items():
                if 'num_batches_tracked' not in name:
                    self.FeatureExtraction.state_dict()[name].copy_(
                        checkpoint['state_dict']['FeatureExtraction.' + name])
            for name, param in self.NeighConsensus.state_dict().items():
                self.NeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['NeighConsensus.' + name])
            print('Done!')

        self.FeatureExtraction.eval()

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data = p.data.half()
            for l in self.NeighConsensus.conv:
                if isinstance(l, Conv4d):
                    l.use_half = True

    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch):
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        if self.half_precision:
            feature_A = feature_A.half()
            feature_B = feature_B.half()
        # feature correlation
        corr4d = self.FeatureCorrelation(feature_A, feature_B)
        # do 4d maxpooling for relocalization
        if self.relocalization_k_size > 1:
            corr4d, max_i, max_j, max_k, max_l = maxpool4d(corr4d, k_size=self.relocalization_k_size)

        # run match processing model
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)
        batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()
        nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        MAP2D_NET, FLOW2D_NET = self.find_correspondence(nc_B_Avec)


        if self.relocalization_k_size > 1:
            delta4d = (max_i, max_j, max_k, max_l)
            return (corr4d, delta4d)
        else:
            return MAP2D_NET

