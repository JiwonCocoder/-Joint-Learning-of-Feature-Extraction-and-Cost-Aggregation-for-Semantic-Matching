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
from lib.matching_model import unNormMap1D_to_NormMap2D, NormMap2D_to_unNormMap2D
from lib.showPlot import plot_test_map, plot_test_flow, warpImg_fromMap, warpImg_fromMap2, matplotlib_imshow, return_plot_test_map, get_img_from_fig
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
        return features
class adap_layer_feat3(nn.Module):
    def __init__(self):
        super(adap_layer_feat3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        print("find_correspondence_gpu:",device)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.conv1.cuda()
            self.conv2.cuda()
    def forward(self, feature):
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        return feature

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
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True):
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
        self.adap_layer_feat3 = adap_layer_feat3()
        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)

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
    def forward(self, tnf_batch, writer, writer_position, mode, label):
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        adap_feature_A = self.adap_layer_feat3(feature_A)
        adap_feature_B = self.adap_layer_feat3(feature_B)

        adap_feature_A = featureL2Norm(adap_feature_A)
        adap_feature_B = featureL2Norm(adap_feature_B)
        if self.half_precision:
            feature_A = feature_A.half()
            feature_B = feature_B.half()
        # feature correlation
        corr4d = self.FeatureCorrelation(adap_feature_A, adap_feature_B)
        batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()
        nc_B_Avec_WTA = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        nc_B_Avec_WTA_norm = torch.nn.functional.softmax(nc_B_Avec_WTA, 1)
        nc_B_Avec_WTA_L2norm = featureL2Norm(nc_B_Avec_WTA)
        scores_WTA_B, index_WTA_B = torch.max(nc_B_Avec_WTA_norm, dim=1)
        scores_WTA_B_noNorm, _ = torch.max(nc_B_Avec_WTA, dim=1)
        scores_WTA_B_L2Norm, _ = torch.max(nc_B_Avec_WTA_L2norm, dim=1)

        index1D_WTA_B = index_WTA_B.view(batch_size, -1)
        Map2D_WTA = unNormMap1D_to_NormMap2D(index1D_WTA_B)

        # do 4d maxpooling for relocalization
        if self.relocalization_k_size > 1:
            corr4d, max_i, max_j, max_k, max_l = maxpool4d(corr4d, k_size=self.relocalization_k_size)
        # run match processing model
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d_NET = MutualMatching(corr4d)

        batch_size, ch, fs1, fs2, fs3, fs4 = corr4d_NET.size()
        nc_B_Avec_NET = corr4d_NET.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        nc_B_Avec_NET_norm = torch.nn.functional.softmax(nc_B_Avec_NET, 1)
        nc_B_Avec_NET_L2norm = featureL2Norm(nc_B_Avec_NET)
        scores_NET_B, index_NET_B = torch.max(nc_B_Avec_NET_norm, dim=1)
        scores_NET_B_noNorm, _ = torch.max(nc_B_Avec_NET, dim=1)
        scores_NET_B_L2Norm, _ = torch.max(nc_B_Avec_NET_L2norm, dim=1)
        index1D_NET_B = index_NET_B.view(batch_size, -1)
        Map2D_NET = unNormMap1D_to_NormMap2D(index1D_NET_B)



        if writer_position % (10-1) == 0 and mode == 'train':
            img_grid = return_plot_test_map(tnf_batch['source_image'][0].unsqueeze(0), tnf_batch['target_image'][0].unsqueeze(0), Map2D_WTA[0].unsqueeze(0),  Map2D_NET[0].unsqueeze(0),scale_factor=16, plot_name='AtoB_MAP')
            writer.add_figure('adap_grid/adap_NET_{}'.format(writer_position), img_grid)
            writer.add_scalar('mean_score/adap_before', torch.mean(scores_WTA_B_noNorm[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('mean_score/adap_softMax', torch.mean(scores_WTA_B[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('mean_score/adap_L2norm', torch.mean(scores_WTA_B_L2Norm[0, :, :]).data.cpu().numpy(),
                              writer_position)                                                            
            writer.add_scalar('mean_score/net_before', torch.mean(scores_NET_B_noNorm[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('mean_score/net_softMax', torch.mean(scores_NET_B[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('mean_score/net_L2norm',torch.mean(scores_NET_B_L2Norm[0, :, :]).data.cpu().numpy(),
                              writer_position)   

            writer.add_scalar('max_score/adap_before', torch.max(scores_WTA_B_noNorm[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('max_score/adap_softMax', torch.max(scores_WTA_B[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('max_score/adap_L2norm', torch.max(scores_WTA_B_L2Norm[0, :, :]).data.cpu().numpy(),
                              writer_position)                                                            
            writer.add_scalar('max_score/net_before', torch.max(scores_NET_B_noNorm[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('max_score/net_softMax', torch.max(scores_NET_B[0, :, :]).data.cpu().numpy(),
                              writer_position)
            writer.add_scalar('max_score/net_L2norm', torch.max(scores_NET_B_L2Norm[0, :, :]).data.cpu().numpy(),
                              writer_position)   

        return corr4d_NET


