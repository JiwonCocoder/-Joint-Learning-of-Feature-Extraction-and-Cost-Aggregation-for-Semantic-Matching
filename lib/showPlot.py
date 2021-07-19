import datetime
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from lib.matching_model import warp_from_NormMap2D, warp, warp_with_mask
import io
import cv2
from torchvision.utils import save_image
def plot_test_map(source_img, target_img, norm_net_map, norm_WTA_map, scale_factor, plot_name): #A_Bvec #B_Avec
    scale_img = 1
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    norm_net_map = F.interpolate(input=norm_net_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)


    warp_net = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)

    warp_WTA = warp_from_NormMap2D(src, norm_WTA_map)
    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_net_np)
    axis[1][0].set_title("net_"+ str(plot_name))
    axis[1][1].imshow(warp_WTA_np)
    axis[1][1].set_title("wta_"+ str(plot_name))
    plt.show()
    # save_path = './WTA_from_Map_Neighbor'
    save_path = './WTA_from_Map_training'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    fig.savefig('{}/res_epoch{}.png'.format(save_path, plot_name),
                bbox_inches='tight')
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# def return_plot_test_map(source_img, target_img, norm_net_map, norm_WTA_map, scale_factor, plot_name): #A_Bvec #B_Avec
#     scale_img = 1
#
#     mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
#     std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
#     if source_img.is_cuda:
#         mean=mean.cuda()
#         std=std.cuda()
#
#     # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
#     # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
#     src=source_img.mul(std).add(mean)
#     tgt=target_img.mul(std).add(mean)
#
#     norm_net_map = F.interpolate(input=norm_net_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
#     norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
#
#
#     warp_net = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)
#
#     warp_WTA = warp_from_NormMap2D(src, norm_WTA_map)
#     src_img = src * 255.0
#     tgt_img = tgt * 255.0
#     src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
#     tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
#
#     warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
#     warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
#     fig, axis = plt.subplots(2, 2, figsize=(20, 20))
#     axis[0][0].imshow(src_np)
#     axis[0][0].set_title("src_"+ str(plot_name))
#     axis[0][1].imshow(tgt_np)
#     axis[0][1].set_title("tgt_" + str(plot_name))
#     axis[1][0].imshow(warp_net_np)
#     axis[1][0].set_title("adaWTA_"+ str(plot_name))
#     axis[1][1].imshow(warp_WTA_np)
#     axis[1][1].set_title("WTA_NET_union_"+ str(plot_name))
#
#     return fig

def return_plot_test_map_mask(source_img, target_img, normed_index, mask_img, scale_factor, plot_name): #A_Bvec #B_Avec
    scale_img = 1

    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    norm_net_map = F.interpolate(input=normed_index, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    # norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)


    wared_img = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)

    masekd_wared_img = wared_img* mask_img

    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    masekd_wared_img_np = masekd_wared_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    wared_img_np = wared_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(20, 20))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(wared_img_np)
    axis[1][0].set_title("adaWTA_masked_"+ str(plot_name))
    axis[1][1].imshow(masekd_wared_img_np)
    axis[1][1].set_title("WTA_NET_union_masked_"+ str(plot_name))

    return fig

def return_plot_test_map(source_img, target_img, norm_WTA_map, norm_net_map, scale_factor, plot_name): #A_Bvec #B_Avec
    scale_img = 1

    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    norm_net_map = F.interpolate(input=norm_net_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)


    warp_net = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)

    warp_WTA = warp_from_NormMap2D(src, norm_WTA_map)

    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(20, 20))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_WTA_np)
    axis[1][0].set_title("base_"+ str(plot_name))
    axis[1][1].imshow(warp_net_np)
    axis[1][1].set_title("ada_"+ str(plot_name))


    return fig

def plot_test_map_mask_img(source_img, target_img, norm_net_map, norm_WTA_map, mask, scale_factor, plot_name): #A_Bvec #B_Avec
    scale_img = 1
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    norm_net_map = F.interpolate(input=norm_net_map, scale_factor=scale_factor, mode='bilinear', align_corners= False)
    norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= False)


    warp_net = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)

    warp_WTA = warp_from_NormMap2D(src, norm_WTA_map)
    warp_WTA = warp_WTA * mask
    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_net_np)
    axis[1][0].set_title("net_"+ str(plot_name))
    axis[1][1].imshow(warp_WTA_np)
    axis[1][1].set_title("wta_"+ str(plot_name))
    plt.show()

def plot_test_map_mask_fm(source_img, target_img, norm_net_map, norm_WTA_map, mask, scale_factor, plot_name): #A_Bvec #B_Avec
    scale_img = 1
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    source_img= F.interpolate(input = source_img, scale_factor = 1/16, mode = 'bilinear')
    target_img= F.interpolate(input = target_img, scale_factor = 1/16, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    norm_net_map = F.interpolate(input=norm_net_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    norm_WTA_map = F.interpolate(input=norm_WTA_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)


    warp_net = warp_from_NormMap2D(src, norm_net_map) #(B, 2, H, W)

    warp_WTA = warp_from_NormMap2D(src, norm_WTA_map)
    warp_WTA = warp_WTA * mask
    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_net_np)
    axis[1][0].set_title("net_"+ str(plot_name))
    axis[1][1].imshow(warp_WTA_np)
    axis[1][1].set_title("wta_"+ str(plot_name))
    plt.show()
    # save_path = './WTA_from_Map_Neighbor'
    # save_path = './WTA_from_Map_training'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # fig.savefig('{}/res_epoch{}.png'.format(save_path, plot_name),
    #             bbox_inches='tight')

def plot_test_flow(source_img, target_img, net_flow, WTA_flow, scale_factor, plot_name): #A_Bvec #B_Avec
    # scale_img = 1/16
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # source_img= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # target_img= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    warp_net = warp(src, net_flow) #(B, 2, H, W)

    warp_WTA = warp(src, WTA_flow)
    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_net_np = warp_net.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_net_np)
    axis[1][0].set_title("net_"+ str(plot_name))
    axis[1][1].imshow(warp_WTA_np)
    axis[1][1].set_title("wta_"+ str(plot_name))
    plt.show()
    # save_path = './WTA_from_Flow_Neighbor'
    # save_path = './WTA_from_Flow_training'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # fig.savefig('{}/res_epoch{}.png'.format(save_path, plot_name),
    #             bbox_inches='tight')

def plot_test_flow_mask(source_img, target_img, WTA_flow, mask, scale_factor, plot_name): #A_Bvec #B_Avec
    # scale_img = 1/16
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # source_img= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear', align_corners=True)
    # target_img= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear', align_corners=True)
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    warp_WTA = warp(src, WTA_flow) #(B, 2, H, W)
    # warp_WTA_mask = warp_with_mask(src, WTA_flow, mask)
    warp_WTA_mask = warp_WTA * mask
    src_img = src * 255.0
    tgt_img = tgt * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    warp_WTA_np = warp_WTA.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    warp_WTA_mask_np = warp_WTA_mask.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(src_np)
    axis[0][0].set_title("src_"+ str(plot_name))
    axis[0][1].imshow(tgt_np)
    axis[0][1].set_title("tgt_" + str(plot_name))
    axis[1][0].imshow(warp_WTA_np)
    axis[1][0].set_title("wta_"+ str(plot_name))
    axis[1][1].imshow(warp_WTA_mask_np)
    axis[1][1].set_title("wta_mask_"+ str(plot_name))
    plt.show()
    # save_path = './WTA_from_Flow_Neighbor'
    # save_path = './WTA_from_Flow_training'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # fig.savefig('{}/res_epoch{}.png'.format(save_path, plot_name),
    #             bbox_inches='tight')
def warpImg_fromMap(source_img,norm_map, scale_factor):
    scale_img = 1
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    src=source_img.mul(std).add(mean)
    norm_map_img = F.interpolate(input=norm_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    warp_img = warp_from_NormMap2D(src, norm_map_img)
    return warp_img

def warpImg_fromMap2(source_img,norm_map, scale_factor):
    scale_img = 1
    source_img = source_img.unsqueeze(0)
    norm_map = norm_map.unsqueeze(0)
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    src=source_img.mul(std).add(mean)
    norm_map_img = F.interpolate(input=norm_map, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    warp_img = warp_from_NormMap2D(src, norm_map_img)
    return warp_img
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize

    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
def save_plot(source_img, target_img, warped_AtoB, plot_name): #A_Bvec #B_Avec
    scale_img = 1
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if source_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    src=source_img.mul(std).add(mean)
    tgt=target_img.mul(std).add(mean)

    src_img = src * 255.0
    tgt_img = tgt * 255.0
    warped_img = warped_AtoB * 255.0
    src_np = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    tgt_np = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    warped_np = warped_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    fig, axis = plt.subplots(1, 3, figsize=(30, 30))
    axis[0].imshow(src_np)
    axis[0].set_title("A")
    axis[1].imshow(tgt_np)
    axis[1].set_title("B")
    axis[2].imshow(warped_np)
    axis[2].set_title("WTA_AtoB")

    # plt.show()
    # save_path = './PLOT_FLOW_KERNEL_Map'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # fig.savefig('{}/{}.png'.format(save_path, plot_name),bbox_inches='tight')
