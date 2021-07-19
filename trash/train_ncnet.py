from __future__ import print_function, division
import os
from os.path import exists, join, basename
from collections import OrderedDict
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset

from lib.dataloader import DataLoader  # modified dataloader
from lib.model_train.model_ncnet import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool

import argparse
from tensorboardX import SummaryWriter
import random

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/',
                    help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5, 5, 5],
                    help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16, 16, 1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_ncnet_img_lr54_test', help='trained model filename')
parser.add_argument('--result_model_dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params', type=int, default=0, help='number of layers to finetune')
parser.add_argument('--gpu_id', type=int, default=1, help='gpu_id')

args = parser.parse_args()
print(args)

# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed_all(1)  # if use multi-GPU
os.environ['PYTHONHASHSEED'] = str(1)

print("use_cuda:",use_cuda)
GPU_NUM = args.gpu_id
print("GPU_NUM", GPU_NUM)
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
torch.cuda.set_device(device)
print('Current cuda device', torch.cuda.current_device())
print('ImMatchNet training script')

save_path = args.result_model_fn
save_path = os.path.join('/root/dataset2/checkpoint_ivd/', args.result_model_fn)

print("summary_writer_path:", save_path)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(os.path.join(save_path, 'train'))
# Create model
print('Creating CNN model...')

torch.manual_seed(1)
if use_cuda:
    print('torch cuda manual seed used =========================')
    torch.cuda.manual_seed(1)
np.random.seed(1)

random.seed(1)


model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint,
                   ncons_kernel_sizes=args.ncons_kernel_sizes,
                   ncons_channels=args.ncons_channels)

torch.manual_seed(1)
if use_cuda:
    print('torch cuda manual seed used =========================')
    torch.cuda.manual_seed(1)
np.random.seed(1)

random.seed(1)

# Set which parts of the model to train
if args.fe_finetune_params > 0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i + 1)].parameters():
            p.requires_grad = True

print('Trainable parameters:')
for i, p in enumerate(filter(lambda p: p.requires_grad, model.parameters())):
    print(str(i + 1) + ": " + str(p.shape))

# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

cnn_image_size = (args.image_size, args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
test_csv = 'val_pairs.csv'
normalization_tnf = NormalizeImageDict(['source_image', 'target_image'])
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
                  dataset_image_path=args.dataset_image_path,
                  dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file=train_csv,
                  output_size=cnn_image_size)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=0)

dataset_test = Dataset(transform=normalization_tnf,
                       dataset_image_path=args.dataset_image_path,
                       dataset_csv_path=args.dataset_csv_path,
                       dataset_csv_file=test_csv,
                       output_size=cnn_image_size)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)


# Train
best_test_loss = float("inf")
def writer_grad_flow(named_parameters, writer, writer_position):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n, "p.grad is None")
            writer.add_scalar('gradient_flow/{}'.format(n), p.grad.abs().mean().data.cpu().numpy(), writer_position)

def weak_loss(model, batch, writer, writer_position, normalization='softmax', alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization == 'softmax':
        normalize = lambda x: torch.nn.functional.softmax(x, 1)
    elif normalization == 'l1':
        normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)

    b = batch['source_image'].size(0)
    # positive
    # corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr4d = model(batch, writer, writer_position)

    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec = corr4d.view(batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #

    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)

    # compute matching scores
    scores_B, _ = torch.max(nc_B_Avec, dim=1)
    scores_A, _ = torch.max(nc_A_Bvec, dim=1)
    score_pos = torch.mean(scores_A + scores_B) / 2

    # negative
    batch['source_image'] = batch['source_image'][np.roll(np.arange(b), -1), :]  # roll
    corr4d = model(batch, writer, writer_position)
    # corr4d = model({'source_image':batch['source_image'], 'target_image':batch['negative_image']})

    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec = corr4d.view(batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #

    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)

    # compute matching scores
    scores_B, _ = torch.max(nc_B_Avec, dim=1)
    scores_A, _ = torch.max(nc_A_Bvec, dim=1)
    score_neg = torch.mean(scores_A + scores_B) / 2

    # loss


    return score_pos, score_neg


loss_fn = lambda model,batch, writer, writer_position : weak_loss(model,batch,writer, writer_position, normalization='softmax')


# define epoch function
def process_epoch(mode, epoch, model, loss_fn, optimizer, dataloader, batch_preprocessing_fn, writer, use_cuda=True,
                  log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode == 'train':
            optimizer.zero_grad()
        writer_position = (epoch - 1) * len(dataloader) + batch_idx
        tnf_batch = batch_preprocessing_fn(batch)
        score_pos, score_neg = loss_fn(model, tnf_batch, writer, writer_position)
        loss = score_neg - score_pos
        loss_np = loss.data.cpu().numpy()
        if writer_position % 9 == 0:
            writer.add_scalar('Loss_{}/loss_nc'.format(mode), loss, writer_position)
            writer.add_scalar('Loss_{}/score_neg'.format(mode), score_neg, writer_position)
            writer.add_scalar('Loss_{}/score_pos'.format(mode), score_pos, writer_position)        
        print("loss_nc:" + str(loss.data.cpu().numpy()))
        epoch_loss += loss_np
        if mode == 'train':
            loss.backward()
            writer_grad_flow(model.named_parameters(), writer, writer_position)
            optimizer.step()
            if writer_position % 100 == 0:
                for i,p in model.named_parameters():
                    if(p.requires_grad) and ("bias" not in i):
                        writer.add_histogram(i, p.clone().cpu().data.numpy(), writer_position)
        else:
            loss = None
        if batch_idx % log_interval == 0:
            print(mode.capitalize() + ' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize() + ' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss


train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs + 1):
    train_loss[epoch - 1] = process_epoch('train', epoch, model, loss_fn, optimizer, dataloader, batch_preprocessing_fn,
                                          writer, log_interval=1)
    test_loss[epoch - 1] = process_epoch('test', epoch, model, loss_fn, optimizer, dataloader_test,
                                         batch_preprocessing_fn, writer, log_interval=1)

    # remember best loss
    is_best = test_loss[epoch - 1] < best_test_loss
    best_test_loss = min(test_loss[epoch - 1], best_test_loss)
    #to move svr_dir
    args.result_model_dir = os.path.join('/root/dataset2/trained_models_ivd/', args.result_model_dir)
    checkpoint_name = os.path.join(args.result_model_dir,
                                   datetime.datetime.now().strftime(
                                       "%Y-%m-%d_%H:%M") + '_epoch_' + str(
                                       epoch) + '_' + args.result_model_fn + '.pth.tar')

    print('Checkpoint name: ' + checkpoint_name)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best, checkpoint_name)

print('Done!')
