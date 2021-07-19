from __future__ import print_function, division
import os
import numpy as np
import numpy.random
import datetime
import torch
import torch.optim as optim
from torch.nn.functional import relu
from lib.dataloader import DataLoader # modified dataloader
from lib.model_train.model_pixelCT_both import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint
from lib.torch_util import BatchTensorToVars
import argparse
from lib.matching_model import EPE
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import random
from lib.matching_model import multiscaleEPE
import torch.nn.functional as F


# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed_all(1)  # if use multi-GPU
os.environ['PYTHONHASHSEED'] = str(1)

print("use_cuda:",use_cuda)
GPU_NUM = 1

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
torch.cuda.set_device(device)
print('Current cuda device', torch.cuda.current_device())

print('ImMatchNet training script')

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/', help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0000001, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_pixelCT_both_lr7_new', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')
parser.add_argument('--temperature', type=float, default=0.03, help='pixelCT_temperature')
parser.add_argument('--threshold', type=float, default=0.4, help='pixelCT_threshold')

def calc_pixelCT_mask(nc_vec, index_NET, mask, temperature):
    batch_size, _, feature_size, feature_size = nc_vec.size()
    nc_BSS = nc_vec.contiguous().view(batch_size * feature_size * feature_size, feature_size * feature_size)
    nc_BSS_numpy = nc_BSS.detach().cpu().numpy()
    index1D_NET = index_NET.view(batch_size * feature_size * feature_size, 1)
    index1D_NET_numpy = index1D_NET.detach().cpu().numpy()
    #(B * tgt_s * tgt_s, src_s * src_s)
    mask_pixelCT = torch.zeros(batch_size * feature_size * feature_size, feature_size * feature_size).bool()

    mask_pixelCT[torch.arange(batch_size * feature_size * feature_size), index1D_NET.detach().squeeze(1)] = True
    mask_pixelCT_numpy = mask_pixelCT.detach().cpu().numpy()
    # positive = scores_WTA_B.view(batch_size * feature_size * feature_size, -1)

    positive = nc_BSS[mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    positive_numpy = positive.detach().cpu().numpy()
    negative = nc_BSS[~mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    negative_numpy = negative.detach().cpu().numpy()

    mask1D = torch.zeros(batch_size * feature_size * feature_size, 1).bool()
    mask_label = mask.view(-1, 1).bool()
    mask_label_numpy = mask_label.detach().cpu().numpy()
    mask1D[mask_label] = True
    mask1D_numpy = mask1D.detach().cpu().numpy()
    positive= positive[mask1D.squeeze(1), :]
    positive_numpy2 = positive.detach().cpu().numpy()
    negative = negative[mask1D.squeeze(1), :]
    negative_numpy2 = negative.detach().cpu().numpy()
    masked_logits = torch.cat([positive, negative], dim=1)

    eps_temp = 1e-6
    masked_logits = masked_logits / (temperature + eps_temp)
    src_num_fgnd = mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=0, keepdim=True)
    src_num_fgnd_label = src_num_fgnd.item()
    labels = torch.zeros(int(src_num_fgnd_label), device=device, dtype=torch.int64)

    loss_pixelCT = F.cross_entropy(masked_logits, labels, reduction='sum')
    loss_pixelCT = (loss_pixelCT / src_num_fgnd).sum()
    return loss_pixelCT
def writer_grad_flow(named_parameters, writer, writer_position):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n, "p.grad is None")
            writer.add_scalar('gradient_flow/{}'.format(n), p.grad.abs().mean().data.cpu().numpy(), writer_position)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n)

            print(n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            print(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
args = parser.parse_args()
print(args)
save_path = 'trainLog_pixelCT_both_lr7_new'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(os.path.join(save_path, ''))

# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
				   checkpoint=args.checkpoint,
                   ncons_kernel_sizes=args.ncons_kernel_sizes,
                   ncons_channels=args.ncons_channels,
                   threshold = args.threshold)
torch.manual_seed(1)
if use_cuda:
    print('torch cuda manual seed used =========================')
    torch.cuda.manual_seed(1)
np.random.seed(1)

random.seed(1)
# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters():
            p.requires_grad=True

print('Trainable parameters:')
for i,p in enumerate(filter(lambda p: p.requires_grad, model.parameters())):
    print(str(i+1)+": "+str(p.shape))
for i, p in model.named_parameters():
    # print(str(i + 1) + ": " + str(p.shape))
    writer.add_histogram(i, p.clone().cpu().data.numpy(), 0)
# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

cnn_image_size=(args.image_size,args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
test_csv = 'val_pairs.csv'
normalization_tnf = NormalizeImageDict(['source_image','target_image'])
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
	              dataset_image_path=args.dataset_image_path,
	              dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file = train_csv,
                  output_size=cnn_image_size)
print(args.batch_size)
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



def weak_loss(model,batch,writer, writer_position, normalization='softmax',alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization=='softmax':
        normalize = lambda x: torch.nn.functional.softmax(x,1)
    elif normalization=='l1':
        normalize = lambda x: x/(torch.sum(x,dim=1,keepdim=True)+0.0001)

    b = batch['source_image'].size(0)
    # positive
    #corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr_WTA , corr_NET, mask_B_Avec, masked_index_B_Avec = model(batch, writer, writer_position)

    batch_size = corr_WTA.size(0)
    feature_size = corr_WTA.size(2)
    nc_WTA_B_Avec = corr_WTA.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)  # [batch_idx,k_A,i_B,j_B]
    nc_NET_B_Avec = corr_NET.view(batch_size, feature_size * feature_size, feature_size,
                            feature_size)


    #PixelCT
    loss_pixelCT_WTA_B_Avec = calc_pixelCT_mask(nc_WTA_B_Avec, masked_index_B_Avec, mask_B_Avec, args.temperature)
    loss_pixelCT_NET_B_Avec = calc_pixelCT_mask(nc_NET_B_Avec, masked_index_B_Avec, mask_B_Avec, args.temperature)
    
    writer.add_scalar('Loss_train/loss_masked_WTA_pixelCT', loss_pixelCT_WTA_B_Avec, writer_position)
    writer.add_scalar('Loss_train/loss_masked_NET_pixelCT', loss_pixelCT_NET_B_Avec, writer_position)

    loss = loss_pixelCT_WTA_B_Avec + loss_pixelCT_NET_B_Avec
    
    # loss_pixelCT = (loss_pixelCT_WTA_B_Avec + loss_pixelCT_NET_B_Avec)

    # batch_size, _, feature_size, feature_size = nc_vec[0].size()
    # nc_B_Avec_BSS = nc_vec[0].view(batch_size* feature_size * feature_size, feature_size * feature_size)
    # index1D_NET_B = index_NET[0].view(batch_size*feature_size*feature_size,1)
    # mask_pixelCT_B = torch.zeros(batch_size * feature_size * feature_size, feature_size * feature_size).bool()
    # mask_pixelCT_B[torch.arange(batch_size * feature_size * feature_size), index1D_NET_B.detach().squeeze(1)] = True
    # # positive = scores_WTA_B.view(batch_size * feature_size * feature_size, -1)
    # positive = nc_B_Avec_BSS[mask_pixelCT_B].view(batch_size * feature_size * feature_size, -1)
    # negative = nc_B_Avec_BSS[~mask_pixelCT_B].view(batch_size * feature_size * feature_size, -1)
    #
    # logits = torch.cat([positive, negative], dim=1)
    #
    # mask_B_Avec_BSS = mask[0].view(batch_size*feature_size*feature_size,1)
    # masked_logits = mask_B_Avec_BSS * logits
    # temperature = args.temperature
    # eps_temp = 1e-6
    # masked_logits = masked_logits / (temperature + eps_temp)
    # labels = torch.zeros(batch_size * feature_size * feature_size, device=device, dtype=torch.int64)
    #
    # loss_pixelCT = F.cross_entropy(masked_logits, labels, reduction='sum')
    #
    # eps = 1
    # src_num_fgnd = mask[0].sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps
    #
    # loss_pixelCT = loss_pixelCT / (src_num_fgnd)


    return loss
loss_fn = lambda model,batch, writer, writer_position : weak_loss(model,batch,writer, writer_position, normalization='softmax')

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn, writer, use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':
            optimizer.zero_grad()
        writer_position = (epoch -1) * len(dataloader) + batch_idx
        tnf_batch = batch_preprocessing_fn(batch)
        # loss_nc, loss_flow, loss_contrastive = loss_fn(model,tnf_batch, batch_idx)
        loss = loss_fn(model, tnf_batch, writer, writer_position)
        loss_np = loss.data.cpu().numpy()
        writer.add_scalar('Loss_{}/loss_pixelCT_both'.format(mode), loss, writer_position)
        print("loss_pixelCT_both:" + str(loss.data.cpu().numpy()))
        epoch_loss += loss_np
        # save_checkpoint({
        #     'epoch': epoch,
        #     'args': args,
        #     'state_dict': model.state_dict(),
        #     'best_test_loss': best_test_loss,
        #     'optimizer': optimizer.state_dict(),
        #     'train_loss': epoch_loss,
        #     'test_loss': epoch_loss,
        # }, False, os.path.join('./trained_models/',
        #                      'baseline_feature_extraction' + '.pth.tar'))
        # print("baseline!!")

        if mode=='train':
            loss.backward()
            writer_grad_flow(model.named_parameters(), writer,writer_position)
            # plot_grad_flow(model.named_parameters())
            if writer_position % 100 == 0:
                for i,p in model.named_parameters():
                    if(p.requires_grad) and ("bias" not in i):
                        writer.add_histogram(i, p.clone().cpu().data.numpy(), writer_position)
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(epoch, batch_idx , len(dataloader), 100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs+1):
    train_loss[epoch-1] = process_epoch('train',epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,writer, log_interval=1)
    test_loss[epoch-1] = process_epoch('test',epoch,model,loss_fn,optimizer,dataloader_test,batch_preprocessing_fn,writer, log_interval=1)

    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    # Define checkpoint name
    checkpoint_name = os.path.join(args.result_model_dir,
                                   datetime.datetime.now().strftime(
                                       "%Y-%m-%d_%H:%M") + '_epoch_' + str(epoch) + '_' + args.result_model_fn + '.pth.tar')

    print('Checkpoint name: ' + checkpoint_name)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')