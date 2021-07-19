from __future__ import print_function, division
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.model_eval.model_eval_adap_softArgmax import ImMatchNet
from lib.pf_dataset import PFPascalDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import BatchTensorToVars
from lib.point_tnf import corr_to_matches
from lib.eval_util import pck_metric
from lib.dataloader import default_collate
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse
import random
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)
print('NCNet evaluation script - PF Pascal dataset')

use_cuda = torch.cuda.is_available()
print("use_cuda:", use_cuda)
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
torch.cuda.set_device(device)
print('Current cuda device', torch.cuda.current_device())
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--eval_dataset_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')

args = parser.parse_args()
#save_path is correlated with train_checkpoint
save_path = './testLog_pixelCT_both'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(os.path.join(save_path, ''))
# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint)

# Dataset and dataloader
Dataset = PFPascalDataset
collate_fn = default_collate
csv_file = 'image_pairs/test_pairs.csv'

cnn_image_size = (args.image_size, args.image_size)

dataset = Dataset(csv_file=os.path.join(args.eval_dataset_path, csv_file),
                  dataset_path=args.eval_dataset_path,
                  transform=NormalizeImageDict(['source_image', 'target_image']),
                  output_size=cnn_image_size)
dataset.pck_procedure = 'scnet'

# Only batch_size=1 is supported for evaluation
batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        collate_fn=collate_fn)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

model.eval()

# initialize vector for storing results
stats = {}
stats['point_tnf'] = {}
stats['point_tnf']['pck'] = np.zeros((len(dataset), 1))


def unNormMap1D_to_NormMap2D(Map1D):
    batch_size, sz = Map1D.size()
    h = sz // 25
    w = h
    xA_WTA = (Map1D % w).view(batch_size, 1, h, w)
    yA_WTA = (Map1D // w).view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()
    return Map2D_WTA


def unNormMap1D_to_NormMap2D(idx_B_Avec, delta4d=None, k_size=1, do_softmax=False, scale='centered',
                             return_indices=False,
                             invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x
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

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).view(1, -1)))
    # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
    jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
    # iB = IB.expand_as(iA)
    # jB = JB.expand_as(jA)

    xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

    xA_WTA = xA.view(batch_size, 1, h, w)
    yA_WTA = yA.view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

    return Map2D_WTA


# Compute
for i, batch in enumerate(dataloader):
    batch = batch_tnf(batch)
    batch_start_idx = batch_size * i

    # corr4d, Map2D_WTA, Map2D_NET = model(batch)
    corr4d = model(batch)
    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    # nc_B_Avec_norm = torch.nn.functional.softmax(nc_B_Avec, 1)
    nc_B_Avec_L2norm = featureL2Norm(nc_B_Avec)
    scores_B, index_B = torch.max(nc_B_Avec_L2norm, dim=1)
    index1D_B = index_B.view(batch_size, -1)

    Map2D_WTA = unNormMap1D_to_NormMap2D(index1D_B)
    # get matches (normalized_1D_Map)
    xA, yA, xB, yB, sB = corr_to_matches(corr4d, do_softmax=True)
    # 2D flow map
    xA = xA.view(batch_size, 1, feature_size, feature_size)
    yA = yA.view(batch_size, 1, feature_size, feature_size)
    Map_2D_Corr = torch.cat((xA, yA), 1).float()
    # warping
    matches = (xA, yA, xB, yB)
    stats = pck_metric(batch, batch_start_idx, matches, stats, args, use_cuda)

    print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

# Print results
results = stats['point_tnf']['pck']
good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
print('Total: ' + str(results.size))
print('Valid: ' + str(good_idx.size))
filtered_results = results[good_idx]
print('PCK:', '{:.2%}'.format(np.mean(filtered_results)))