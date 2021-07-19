



from __future__ import print_function, division
import os
import numpy as np
import numpy.random
import datetime
import torch
import torch.optim as optim
from torch.nn.functional import relu
from lib.dataloader import DataLoader # modified dataloader
from lib.model_train.model_pixelCT_mask_both_FB_check_differ import ImMatchNet
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




# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/', help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_ada_agg_both_mask_differ_all_55_temp1_2', help='trained model filename')
parser.add_argument('--trainLog', type=str, default='trainLog_ada_agg_both_mask_differ_all_55_temp1_2', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')
parser.add_argument('--temperature', type=float, default=1, help='pixelCT_temperature')
parser.add_argument('--threshold', type=float, default=0.4, help='pixelCT_threshold')
parser.add_argument('--gpu_id', type=int, default=1, help='gpu_id')
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)
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

def calc_pixelCT_mask(nc_vec, index_NET, mask, temperature):
    batch_size, _, feature_size, feature_size = nc_vec.size()

    nc_BSS = nc_vec.contiguous().view(batch_size * feature_size * feature_size, feature_size * feature_size)
    # nc_BSS_numpy = nc_BSS.detach().cpu().numpy()
    index1D_NET = index_NET.view(batch_size * feature_size * feature_size, 1)
    # index1D_NET_numpy = index1D_NET.detach().cpu().numpy()
    #(B * tgt_s * tgt_s, src_s * src_s)
    mask_pixelCT = torch.zeros(batch_size * feature_size * feature_size, feature_size * feature_size).bool()

    mask_pixelCT[torch.arange(batch_size * feature_size * feature_size), index1D_NET.detach().squeeze(1)] = True
    # mask_pixelCT_numpy = mask_pixelCT.detach().cpu().numpy()
    # positive = scores_WTA_B.view(batch_size * feature_size * feature_size, -1)

    positive = nc_BSS[mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    print("--------------calc_mask---------------------")
    print("index_NET", index1D_NET[15*25 + 15])
    print("nc_BSS_score", nc_BSS[15*25 + 15, index1D_NET[15*25 + 15]])
    positive = nc_BSS[mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    print("positive", positive[15*25 + 15])

    # positive_numpy = positive.detach().cpu().numpy()
    negative = nc_BSS[~mask_pixelCT].view(batch_size * feature_size * feature_size, -1)
    # negative_numpy = negative.detach().cpu().numpy()

    mask1D = torch.zeros(batch_size * feature_size * feature_size, 1).bool()
    mask_label = mask.view(-1, 1).bool()
    # mask_label_numpy = mask_label.detach().cpu().numpy()
    mask1D[mask_label] = True
    print("mask1D", mask1D[15*25 + 15])
    print("-----------------------------------")
    mask1D_numpy = mask1D.detach().cpu().numpy()
    mask1D = mask1D.squeeze(1)
    positive= positive[mask1D, :]

    # psitive2 = positive[mask1D]
    # positive_numpy2 = positive.detach().cpu().numpy()
    negative = negative[mask1D, :]
    # psitive2 = positive[mask1D]
    # negative_numpy2 = negative.detach().cpu().numpy()
    masked_logits = torch.cat([positive, negative], dim=1)


    eps_temp = 1e-6
    masked_logits = (masked_logits / temperature) + eps_temp
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
# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed_all(1)  # if use multi-GPU
os.environ['PYTHONHASHSEED'] = str(1)

print("use_cuda:",use_cuda)
print(args.gpu_id)
GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
torch.cuda.set_device(device)
print('Changed cuda device', torch.cuda.current_device())

print('ImMatchNet training script')
print(args)
# save_path = 'trainLog_ada_agg_both_mask_FB_check_differ'
save_path = args.trainLog
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



def weak_loss(model,batch,writer_position, normalization='softmax',alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization=='softmax':
        normalize = lambda x: torch.nn.functional.softmax(x,1)
    elif normalization=='l1':
        normalize = lambda x: x/(torch.sum(x,dim=1,keepdim=True)+0.0001)

    b = batch['source_image'].size(0)
    # positive
    #corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr_WTA , corr_NET, mask_B, index_B= model(batch, writer, writer_position)

    batch_size = corr_WTA.size(0)
    feature_size = corr_WTA.size(2)
    batch_size, ch, fs1, fs2, fs3, fs4 = corr_NET.size()
    nc_WTA_A_Bvec = corr_WTA.view(batch_size, fs1, fs2, fs3 * fs4).permute(0, 3, 1, 2).contiguous()  # (B,B_S*B_S, A_S,A_S)
    nc_NET_A_Bvec = corr_NET.view(batch_size, fs1, fs2, fs3 * fs4).permute(0, 3, 1, 2).contiguous()  #(B,B_S*B_S, A_S,A_S)
    nc_WTA_A_Bvec = featureL2Norm(nc_WTA_A_Bvec)
    nc_NET_A_Bvec = featureL2Norm(nc_NET_A_Bvec)
    #check
    nc_WTA_B_Avec = corr_WTA.view(batch_size, fs1 * fs2, fs3,fs4) # (B,B_S*B_S, A_S,A_S)
    nc_NET_B_Avec = corr_NET.view(batch_size, fs1 * fs2, fs3,fs4) # (B,B_S*B_S, A_S,A_S)
    print("before_L2norm(WTA)",nc_WTA_B_Avec[0, int(index_B[0][0, 15, 15]), 15, 15])
    print("before_L2norm(NET)",nc_NET_B_Avec[0, int(index_B[1][0, 15, 15]), 15, 15])
    nc_WTA_B_Avec = featureL2Norm(nc_WTA_B_Avec)
    nc_NET_B_Avec = featureL2Norm(nc_NET_B_Avec)
    print("after_L2norm(WTA)",nc_WTA_B_Avec[0, int(index_B[0][0, 15, 15]), 15, 15])
    print("after_L2norm(NET)",nc_NET_B_Avec[0, int(index_B[1][0, 15, 15]), 15, 15])    

    # nc_NET_A_Bvec = featureL2Norm(nc_NET_A_Bvec)
    #Adap
    print("pos")

    print("--------strat------------")
    print("WTA")
    loss_pixelCT_WTA_B_Avec_byWTA_pos = calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[0], mask_B[0], args.temperature)
    loss_pixelCT_WTA_B_Avec_byNET_pos = calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[1], mask_B[1], args.temperature)

    #Aggre
    print("NET")
    loss_pixelCT_NET_B_Avec_byWTA_pos = calc_pixelCT_mask(nc_NET_A_Bvec, index_B[0], mask_B[0], args.temperature)
    loss_pixelCT_NET_B_Avec_byNET_pos = calc_pixelCT_mask(nc_NET_A_Bvec, index_B[1], mask_B[1], args.temperature)
    print("--------end------------")

    #neg
    print("neg")
    
    batch['source_image'] = batch['source_image'][np.roll(np.arange(b), -1), :]  # roll
    corr_WTA , corr_NET, mask_B, index_B= model(batch, writer, writer_position)
    batch_size = corr_WTA.size(0)
    feature_size = corr_WTA.size(2)
    nc_WTA_A_Bvec = corr_WTA.view(batch_size, fs1, fs2, fs3 * fs4).permute(0, 3, 1, 2)  # (B,B_S*B_S, A_S,A_S)
    nc_NET_A_Bvec = corr_NET.view(batch_size, fs1, fs2, fs3 * fs4).permute(0, 3, 1, 2)  #(B,B_S*B_S, A_S,A_S)
    nc_WTA_A_Bvec = featureL2Norm(nc_WTA_A_Bvec)
    nc_NET_A_Bvec = featureL2Norm(nc_NET_A_Bvec)

    #check
    nc_WTA_B_Avec = corr_WTA.view(batch_size, fs1 * fs2, fs3,fs4) # (B,B_S*B_S, A_S,A_S)
    nc_NET_B_Avec = corr_NET.view(batch_size, fs1 * fs2, fs3,fs4) # (B,B_S*B_S, A_S,A_S)
    print("before_L2norm(WTA)",nc_WTA_B_Avec[0, int(index_B[0][0, 15, 15]), 15, 15])
    print("before_L2norm(NET)",nc_NET_B_Avec[0, int(index_B[1][0, 15, 15]), 15, 15])
    nc_WTA_B_Avec = featureL2Norm(nc_WTA_B_Avec)
    nc_NET_B_Avec = featureL2Norm(nc_NET_B_Avec)
    print("after_L2norm(WTA)",nc_WTA_B_Avec[0, int(index_B[0][0, 15, 15]), 15, 15])
    print("before_L2norm(NET)",nc_NET_B_Avec[0, int(index_B[1][0, 15, 15]), 15, 15])    
    if writer_position % 10 == 0:
        writer.add_scalar('corr/corr_adap_after',nc_WTA_B_Avec[0, int(index_B[0][0, 15, 15]), 15, 15].data.cpu().numpy(), writer_position)
        writer.add_scalar('corr/corr_net_after',nc_NET_B_Avec[0, int(index_B[1][0, 15, 15]), 15, 15].data.cpu().numpy(), writer_position)
    # loss_pixelCT_WTA_B_Avec_byWTA_neg= calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[0], mask_B[0], args.temperature)
    # loss_pixelCT_WTA_B_Avec_byNET_neg = calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[1], mask_B[1], args.temperature)
    print("--------strat------------")
    print("WTA")
    loss_pixelCT_WTA_B_Avec_byWTA_neg = calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[0], mask_B[0], args.temperature)
    loss_pixelCT_WTA_B_Avec_byNET_neg = calc_pixelCT_mask(nc_WTA_A_Bvec, index_B[1], mask_B[1], args.temperature)

    #Aggre
    print("NET")
    loss_pixelCT_NET_B_Avec_byWTA_neg = calc_pixelCT_mask(nc_NET_A_Bvec, index_B[0], mask_B[0], args.temperature)
    loss_pixelCT_NET_B_Avec_byNET_neg = calc_pixelCT_mask(nc_NET_A_Bvec, index_B[1], mask_B[1], args.temperature)
    print("--------end------------")

    return [loss_pixelCT_WTA_B_Avec_byWTA_pos, loss_pixelCT_WTA_B_Avec_byNET_pos],\
           [loss_pixelCT_NET_B_Avec_byWTA_pos, loss_pixelCT_NET_B_Avec_byNET_pos],\
           [loss_pixelCT_WTA_B_Avec_byWTA_neg, loss_pixelCT_WTA_B_Avec_byNET_neg],\
           [loss_pixelCT_NET_B_Avec_byWTA_neg, loss_pixelCT_NET_B_Avec_byNET_neg]
loss_fn = lambda model,batch, writer_position : weak_loss(model,batch,writer_position, normalization='softmax')

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn, writer, use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':
            optimizer.zero_grad()
        writer_position = (epoch -1) * len(dataloader) + batch_idx
        tnf_batch = batch_preprocessing_fn(batch)
        # loss_nc, loss_flow, loss_contrastive = loss_fn(model,tnf_batch, batch_idx)
        loss_pixelCT_WTA_pos, loss_pixelCT_NET_pos, loss_pixelCT_WTA_neg, loss_pixelCT_NET_neg = loss_fn(model, tnf_batch, writer_position)
        if writer_position % 10 == 0:
            writer.add_scalar('Loss_{}/loss_pixelCT_WTA_B_Avec_byWTA_pos'.format(mode), loss_pixelCT_WTA_pos[0], writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_WTA_B_Avec_byNET_pos'.format(mode), loss_pixelCT_WTA_pos[1],
                            writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_byWTA_pos'.format(mode), loss_pixelCT_NET_pos[0],
                            writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_byNET_pos'.format(mode), loss_pixelCT_NET_pos[1], writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_WTA_B_Avec_byWTA_neg'.format(mode), loss_pixelCT_WTA_neg[0],
                            writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_WTA_B_Avec_byNET_neg'.format(mode), loss_pixelCT_WTA_neg[1],
                            writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_byWTA_neg'.format(mode), loss_pixelCT_NET_neg[0],
                            writer_position)
            writer.add_scalar('Loss_{}/loss_pixelCT_NET_B_Avec_byNET_neg'.format(mode), loss_pixelCT_NET_neg[1],
                            writer_position)

        score_pos_pixelCT_WTA = (loss_pixelCT_WTA_pos[0] + loss_pixelCT_WTA_pos[1])/2
        score_pos_pixelCT_NET = (loss_pixelCT_NET_pos[0] + loss_pixelCT_NET_pos[1])/2
        score_neg_pixelCT_WTA = (loss_pixelCT_WTA_neg[0] + loss_pixelCT_WTA_neg[1])/2
        score_neg_pixelCT_NET = (loss_pixelCT_NET_neg[0] + loss_pixelCT_NET_neg[1]) / 2
        score_pos_pixelCT = score_pos_pixelCT_WTA + score_pos_pixelCT_NET
        score_neg_pixelCT = score_neg_pixelCT_WTA + score_neg_pixelCT_NET
        print("score_pos_pixelCT:"+ str(score_pos_pixelCT.data.cpu().numpy()))
        print("score_neg_pixelCT:" + str(score_neg_pixelCT.data.cpu().numpy()))
        loss = score_pos_pixelCT - score_neg_pixelCT
        if writer_position % 10 == 0:
            writer.add_scalar('Loss_{}/score_pos_pixelCT_WTA'.format(mode), score_pos_pixelCT_WTA, writer_position)
            writer.add_scalar('Loss_{}/score_pos_pixelCT_NET'.format(mode), score_pos_pixelCT_NET, writer_position)
            writer.add_scalar('Loss_{}/score_neg_pixelCT_WTA'.format(mode), score_neg_pixelCT_WTA, writer_position)
            writer.add_scalar('Loss_{}/score_neg_pixelCT_NET'.format(mode), score_neg_pixelCT_NET, writer_position)
            writer.add_scalar('Loss_{}/score_pos_pixelCT'.format(mode), score_pos_pixelCT, writer_position)
            writer.add_scalar('Loss_{}/score_neg_pixelCT'.format(mode), score_neg_pixelCT, writer_position)
            writer.add_scalar('Loss_{}/score_both_pixelCT'.format(mode), loss, writer_position)
            writer.add_scalar('Loss_{}/score_both_pixelCT'.format(mode), loss, writer_position)
        loss_np = loss.data.cpu().numpy()
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
