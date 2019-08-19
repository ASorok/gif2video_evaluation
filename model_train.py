import os, sys, glob, time, progressbar, argparse, numpy as np, cv2, imageio, random
from PIL import Image
from itertools import islice
from functools import partial
from easydict import EasyDict as edict
import scipy.io as sio
from skimage.measure import compare_ssim as comp_ssim

import torch
import torch.nn as nn
import torch.nn.functional as FF
import torch.optim as optim
import torch.utils.data as DD
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch_models import UNet, SlomoNet, Discriminator
from data import gif_faces_utils, utils
from utils import *

import warnings
warnings.filterwarnings("ignore")

opts=edict()

opts.k=0#type of cddnet

#data processing parametres
opts.tCrop=5#sequence length
opts.sCrop=256#spatial patch size
opts.tStride=10#temporal downsampling

#training parametres
opts.batch_size = 2
opts.pool_size=30#image pool size
opts.nEpoch=15

#scheduler/optimizer/loss parametres
opts.solver='adam'
opts.MM=0.9#momentum
opts.WD=1e-4#weight decay
opts.gamma=0.1#decay rate for learning rate
opts.Beta=0.999#beta for adam
opts.LRPolicy='constant'#learning rate policy
opts.LRStart=0.0001#initial learning rate
opts.LRStep=100#steps to change learning rate
opts.L_warp_outlier = 25#initial outlier value for warp loss

#firectory parametres
opts.dither_mode='nodither'#dithering / non-dithering images
opts.inputRoot = 'data/' + 'face_gif_image_PIL/expand1.5_size256_s1_g32_' + opts.dither_mode
opts.saveDir='results/default/'
opts.targetRoot = 'data/' + 'face/expand1.5_size256/'

def create_dataloader():
    trSet = gif_faces_utils.gif_faces_ct_train(inputRoot=opts.inputRoot, targetRoot=opts.targetRoot,
                                               tCrop=opts.tCrop, sCrop=opts.sCrop)
    trLD = DD.DataLoader(trSet, batch_size=opts.batch_size,
        sampler= DD.sampler.RandomSampler(trSet),
        num_workers=0, pin_memory=True, drop_last=True)
    evalSet = gif_faces_utils.gif_faces_ct_eval(inputRoot=opts.inputRoot, targetRoot=opts.targetRoot,
                                                tStride=opts.tStride, tCrop=opts.tCrop)
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=0, pin_memory=True, drop_last=False)
    return trLD, evalLD

def board_vis(epoch, frm1, frm0, frm10, frm01, F01, F10, Vt0s, gif, target, imgs):
    B, L, C, H, W = target.shape
    # I0 and I1
    im0, im1 = frm0[:1].detach(), frm1[:1].detach()
    im0_warp, im1_warp = frm10[:1].detach(), frm01[:1].detach()
    im0_err, im1_err = (im0 - im0_warp).abs(), (im1 - im1_warp).abs()
    im01_diff = (im0 - im1).abs()
    x = torch.cat((im0, im1, im0_warp, im1_warp, im0_err, im1_err, im01_diff), dim=0)
    x = vutils.make_grid(x, nrow=2, normalize=True)
    opts.board.add_image('train_batch/i0_i1', x, epoch)
    # flow
    flow01, flow10 = F01[:1].detach(), F10[:1].detach()
    flow01 = torch.cat([flow01, flow01.new_zeros(1, 1, H, W)], dim=1)
    flow10 = torch.cat([flow10, flow10.new_zeros(1, 1, H, W)], dim=1)
    x = torch.cat([flow01, flow10], dim=0)
    x = vutils.make_grid(x, nrow=2, normalize=True, range=(-1, 1))
    opts.board.add_image('train_batch/f01_f10', x, epoch)
    # vis_map
    vis0s = Vt0s[0].detach().expand(-1, 3, -1, -1)
    vis1s = 1 - vis0s
    x = torch.cat([vis0s, vis1s], dim=0)
    x = vutils.make_grid(x, nrow=L-2, normalize=True)
    opts.board.add_image('train_batch/vis0_vis1', x, epoch)
    # interp
    ims_gif = gif[0].detach()
    ims_gt = target[0].detach()
    ims_est = imgs[0].detach()
    ims_err = (ims_est - ims_gt).abs()
    x = torch.cat((ims_gif, ims_gt, ims_est, ims_err), dim=0)
    x = vutils.make_grid(x, nrow=L, normalize=True)
    opts.board.add_image('train_batch/recover', x, epoch)
    
def create_model():
    model = edict()
    model.netG = SlomoNet.netSlomo(maxFlow=30)
    model.netD = Discriminator.Discriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = nn.DataParallel(model[key].to(DEVICE))
    return model

def train(epoch, trLD, model, color_model1, optimizer, fakeABPool):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = ['D_gan', 'D_real', 'D_fake', 'D_acc'] + ['L_gan', 'L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'L_total']
    epL = utils.AverageMeters(tags)
    N = max(1, round(len(trLD) ))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        gif, target, colors = list(map(lambda x: preprocess(x).to(DEVICE), samples))
        B, L, C, H, W = gif.shape
        gif0, gif1 = gif[:, 0], gif[:, -1]
        color0, color1 = colors[:, 0], colors[:, -1]
        frm0, frm1, frm_ts = target[:, 0], target[:, -1], target[:, 1:L-1]
        ts = np.linspace(0, 1, L)[1:L-1].tolist()
        with torch.no_grad():
            #cddnet part
            I0 = color_model1(gif0).tanh()
            
            I1 = color_model1(gif1).tanh()
            
        
        Its, F01, F10, Ft1s, Ft0s, Vt0s = model.netG(gif0, gif1, I0, I1, ts)
        
        imgs = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
        D_input = lambda A, B: torch.cat((A, B, pad_tl(diff_xy(B))), dim=1)
        realAB = D_input(gif.view(B*L, -1, H, W), target.view(B*L, -1, H, W))
        fakeAB = D_input(gif.view(B*L, -1, H, W), imgs.view(B*L, -1, H, W))

        # (1) Update D network
        optimizer.netD.zero_grad()
        fakeAB_ = fakeABPool.query(fakeAB.detach()).to(DEVICE)
        real_logits = model.netD(realAB)
        fake_logits = model.netD(fakeAB_)
        d_gan, d_real, d_fake = utils.compute_D_loss(real_logits, fake_logits, method='GAN')
        d_acc, _, _ = utils.compute_D_acc(real_logits, fake_logits)

        loss_d = d_gan 
        loss_d.backward()
        if d_acc.item() < 0.75:
            nn.utils.clip_grad_norm_(model.netD.parameters(), 1.0)
            optimizer.netD.step()

        # (2) Update G network
        optimizer.netG.zero_grad()
        fake_logits = model.netD(fakeAB)
        L_gan = utils.compute_G_loss(fake_logits, method='GAN')
        L_idl = 127.5*(f_idl(I0, frm0) + f_idl(I1, frm1) + f_idl(Its, frm_ts))
        L_gdl = 127.5*(f_gdl(I0, frm0) + f_gdl(I1, frm1) + f_gdl(Its, frm_ts))
        L_smooth = 30*(f_smooth(F01) + f_smooth(F10))
        frm10 = SlomoNet.backwarp(frm1, F01*30)
        frm01 = SlomoNet.backwarp(frm0, F10*30)
        frm1ts = torch.cat(list(SlomoNet.backwarp(frm1, Ft1s[:, i]*30).unsqueeze(1) for i in range(Ft1s.shape[1])), dim=1)
        frm0ts = torch.cat(list(SlomoNet.backwarp(frm0, Ft0s[:, i]*30).unsqueeze(1) for i in range(Ft0s.shape[1])), dim=1)
        L_warp = 127.5*(f_idl(frm10, frm0) + f_idl(frm01, frm1) + f_idl(frm1ts, frm_ts) + f_idl(frm0ts, frm_ts))

        Loss_g = L_gan + L_idl * 0.5 + L_gdl * 0.5 + L_warp * 0.5 + L_smooth
        Loss_g.backward()
        if d_acc.item() > 0.25 and L_warp < opts.L_warp_outlier:
            nn.utils.clip_grad_norm_(model.netG.parameters(), 1.0)
            optimizer.netG.step()

        # tags = ['D_gan', 'D_real', 'D_fake', 'D_acc'] + ['L_gan', 'L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'L_total']
        values = list(map(lambda x: x.item(), [d_gan, d_real, d_fake, d_acc, L_gan, L_idl, L_gdl, L_warp, L_smooth, Loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%20==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%20==0:
            board_vis(epoch, frm1, frm0, frm10, frm01, F01, F10, Vt0s, gif, target, imgs)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

    opts.L_warp_outlier = epL['L_warp'].avg * 1.5
    print('outlier threshold for L_warp is set to {}'.format(opts.L_warp_outlier))
    
    
def main_train():
    print('==> create color model')
    color_model1 = UNet.UNet(3, 3, ch=64)
    color_model1.load_state_dict(torch.load("pretrained/ccdnet1_gan_faces_nodither_ep30.pt")["model_netG"])
    color_model1 = nn.DataParallel(color_model1.eval().to(DEVICE))
    
    print('==> create dataset loader')
    trLD, evalLD = create_dataloader()
    fakeABPool = utils.ImagePool(opts.pool_size)

    print('==> create model, optimizer, scheduler')
    model = create_model()
    optimizer = create_optimizer(model, opts)
    scheduler = create_scheduler(optimizer, opts)
    
    initialize(model, '')
    start_epoch = 1

    print('==> start training from epoch %d'%(start_epoch))
    for epoch in range(start_epoch, 1 + opts.nEpoch):
        print('\nEpoch {}:\n'.format(epoch))
        for key in scheduler.keys():
            scheduler[key].step(epoch-1)
            lr = scheduler[key].optimizer.param_groups[0]['lr']
            print('learning rate of {} is set to {}'.format(key, lr))
            if opts.board is not None: opts.board.add_scalar('lr_schedule/'+key, lr, epoch)
                
        train(epoch, trLD, model, color_model1, optimizer, fakeABPool)
        
        if epoch%10==0:
            save_checkpoint(epoch, model, optimizer, opts)
        if epoch%50==0:
            evaluate(epoch, evalLD, model)
            
            
if __name__ == "__main__":
    opts.board = SummaryWriter(os.path.join(opts.saveDir, 'board'))
    main_train()
    opts.board.close()