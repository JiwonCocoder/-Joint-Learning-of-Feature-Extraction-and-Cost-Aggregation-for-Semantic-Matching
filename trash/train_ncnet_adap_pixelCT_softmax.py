from __future__ import print_function, division
import os
import numpy as np
import numpy.random
import datetime
import torch
import torch.optim as optim
from torch.nn.functional import relu

from lib.dataloader import DataLoader  # modified dataloader
from lib.model_train.model_pixelCT_ncnet_adap import ImMatchNet
from lib.matching_model import unNormMap1D_to_NormMap2D
from lib.showPlot import plot_test_map, plot_test_flow, warpImg_fromMap, warpImg_fromMap2, matplotlib_imshow, return_plot_test_map, get_img_from_fig

from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint
from lib.torch_util import BatchTensorToVars
import torch.nn.functional as F
import argparse
from tensorboardX import SummaryWriter
import random
use_cuda = torch.cuda.is_available()
print("use_cuda:",use_cuda)
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
torch.cuda.set_device(device)
print('Current cuda device', torch.cuda.current_device())
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

print('ImMatchNet training script')

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/',
                    help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5, 5, 5],
                    help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16, 16, 1], help='channels in neigh. cons')
# parser.add_argument('--result_model_fn', type=str, default='checkpoint_pixelCT_ncnet_adap_lr55_temp007_1', help='trained model filename')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_pixelCT_ncnet_adap_lr55_temp10', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params', type=int, default=0, help='number of layers to finetune')
# parser.add_argument('--temperature', type=float, default=0.07, help='pixelCT_temperature')
parser.add_argument('--temperature', type=float, default=10, help='pixelCT_temperature')

args = parser.parse_args()
print(args)
# save_path = 'trainLog_pixelCT_ncnet_adap_lr54_temp007_1'
save_path = 'trainLog_pixelCT_ncnet_adap_lr55_temp10'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(os.path.join(save_path, ''))
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
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)
def writer_grad_flow(named_parameters, writer, writer_position):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n, "p.grad is None")
            writer.add_scalar('gradient_flow/{}'.format(n), p.grad.abs().mean().data.cpu().numpy(), writer_position)

def calc_pixelCT(nc_A_Bvec, index_NET, temperature):
    batch_size, _, feature_size, feature_size = nc_A_Bvec.size() #(B, B_S * B_S, A_S, A_S)
    nc_BSS = nc_A_Bvec.contiguous().view(batch_size * feature_size * feature_size, feature_size * feature_size)
    nc_BSS_numpy = nc_BSS.detach().cpu().numpy()
    index1D_NET = index_NET.view(batch_size * feature_size * feature_size, 1)
    index1D_NET_numpy = index1D_NET.detach().cpu().numpy()
    # (B * tgt_s * tgt_s, src_s * src_s)
    mask_pixelCT = torch.zeros(batch_size * feature_size * feature_size, feature_size * feature_size).bool()

    mask_pixelCT[torch.arange(batch_size * feature_size * feature_size), index1D_NET.detach().squeeze(1)] = True
    mask_pixelCT_numpy = mask_pixelCT.detach().cpu().numpy()
    # positive = scores_WTA_B.view(batch_size * feature_size * feature_size, -1)
    positive = nc_BSS[mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    print("-------------calc----------------------")
    print("index_NET", index1D_NET[15*25 + 15])
    print("nc_BSS_score", nc_BSS[15*25 + 15, index1D_NET[15*25 + 15]])
    print("positive", positive[15*25 + 15])
    print("-----------------------------------")

    positive_numpy = positive.detach().cpu().numpy()
    negative = nc_BSS[~mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    negative_numpy = negative.detach().cpu().numpy()
    eps_temp = 1e-6
    logits = torch.cat([positive, negative], dim=1)
    logits = (logits / temperature) + eps_temp
    labels = torch.zeros(batch_size * feature_size * feature_size, device=device, dtype=torch.int64)
    loss_pixelCT = F.cross_entropy(logits, labels, reduction='sum')
    loss_pixelCT = loss_pixelCT / (batch_size * feature_size * feature_size)
    return loss_pixelCT


def weak_loss(model, batch, writer_position, mode, normalization='softmax', alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization == 'softmax':
        normalize = lambda x: torch.nn.functional.softmax(x, 1)
    elif normalization == 'l1':
        normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)

    b = batch['source_image'].size(0)
    # positive
    # corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr4d = model(batch, writer, writer_position, mode, label='pos')

    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)

    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec = corr4d.view(batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #
    nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec)
    nc_B_Avec_norm = normalize(nc_B_Avec)
    scores_B, index_B = torch.max(nc_B_Avec_norm, dim=1)

    #check
    # print("before_L2norm(NET)",nc_B_Avec[0, int(index_B[0, 15, 15]), 15, 15])
    # nc_B_Avec = featureL2Norm(nc_B_Avec)
    # print("after_L2norm(NET)",nc_B_Avec[0, int(index_B[0, 15, 15]), 15, 15])
    

    # print("------------------pos-----------------")
    # print("score", scores_B[0, 15, 15])
    # print("index_NET_B", index_B[0, 15, 15])
    loss_pixelCT_NET_B_Avec_by_NET_pos = calc_pixelCT(nc_A_Bvec, index_B, args.temperature)


    # negative
    batch['source_image'] = batch['source_image'][np.roll(np.arange(b), -1), :]  # roll
    corr4d = model(batch, writer, writer_position, mode, label = 'neg')
    # corr4d = model({'source_image':batch['source_image'], 'target_image':batch['negative_image']})

    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec = corr4d.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec = corr4d.view(batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #
    nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec)    
    # nc_A_Bvec = featureL2Norm(nc_A_Bvec)
    nc_B_Avec_norm = normalize(nc_B_Avec)
    scores_B, index_B = torch.max(nc_B_Avec_norm, dim=1)

    # print("score", scores_B[0, 15, 15])
    # print("index_NET_B", index_B[0, 15, 15])
    loss_pixelCT_NET_B_Avec_by_NET_neg = calc_pixelCT(nc_A_Bvec, index_B, args.temperature)

    # loss

    return loss_pixelCT_NET_B_Avec_by_NET_pos, loss_pixelCT_NET_B_Avec_by_NET_neg


loss_fn = lambda model, batch, writer_position, mode : weak_loss(model, batch, writer_position, mode, normalization='softmax')


# define epoch function
def process_epoch(mode, epoch, model, loss_fn, optimizer, dataloader, batch_preprocessing_fn, writer, use_cuda=True,
                  log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode == 'train':
            optimizer.zero_grad()
        writer_position = (epoch -1) * len(dataloader) + batch_idx
        tnf_batch = batch_preprocessing_fn(batch)
        loss_pixelCT_NET_B_Avec_by_NET_pos, loss_pixelCT_NET_B_Avec_by_NET_neg = loss_fn(model, tnf_batch, writer_position, mode)
        loss = loss_pixelCT_NET_B_Avec_by_NET_pos - loss_pixelCT_NET_B_Avec_by_NET_neg
        loss_np = loss.data.cpu().numpy()
        if writer_position % 9 == 0:
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_by_NET_pos'.format(mode), loss_pixelCT_NET_B_Avec_by_NET_pos, writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_by_NET_neg'.format(mode), loss_pixelCT_NET_B_Avec_by_NET_neg,
                            writer_position)
            writer.add_scalar('Loss_{}/loss_nc'.format(mode), loss, writer_position)
        epoch_loss += loss_np
        if mode == 'train':
            loss.backward()
            writer_grad_flow(model.named_parameters(), writer,writer_position)
            optimizer.step()
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
