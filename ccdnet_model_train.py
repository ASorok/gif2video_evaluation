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

opts.k=2#type of cddnet

#data processing parametres
opts.tCrop=5#sequence length
opts.sCrop=256#spatial patch size
opts.tStride=10#temporal downsampling
opts.tDown=8#temporal downsampling

#training parametres
opts.batch_size = 8
opts.pool_size=30#image pool size
opts.nEpoch=40

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
opts.saveDir='ccdnet_results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,gan_d1_g1/k2/'
opts.targetRoot = 'data/face/expand1.5_size256/'


def create_dataloader():
    trSet = gif_faces_utils.gif_faces_train(inputRoot=opts.inputRoot, targetRoot=opts.targetRoot)
    trLD = DD.DataLoader(trSet, batch_size=opts.batch_size,
        sampler= DD.sampler.RandomSampler(trSet),
        num_workers=0, pin_memory=True, drop_last=True)
    evalSet = gif_faces_utils.gif_faces_eval(inputRoot=opts.inputRoot, targetRoot=opts.targetRoot,
                                             tDown=opts.tDown)
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=0, pin_memory=True, drop_last=False)
    return trLD, evalLD

def create_model():
    model = edict()
    iter_ch = 3*4 
    model.netG = UNet.UNet(in_ch=iter_ch, out_ch=3, ch=64)
    model.netD = Discriminator.Discriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = model[key].to(DEVICE)
        if DEVICE != "cpu": model[key] = nn.DataParallel(model[key])
    return model

def board_vis(epoch, gif, pred, target):
    B, C, H, W = target.shape
    error = (pred - target).abs()
    x = torch.cat((gif, target, pred, error), dim=0)
    x = vutils.make_grid(x, nrow=B, normalize=True)
    opts.board.add_image('train_batch/gif_target_pred_error', x, epoch)
    
def iterative_input(fakeB, realA, colors, nColor):
    # fakeB_gif = fakeB
    B, C, H, W = fakeB.shape
    fakeB_gif = []
    for i in range(B):
        _fakeB, _realA = fakeB[i].detach(), realA[i].detach()
        _fakeB = _fakeB.view(C, H*W).transpose(0, 1)
        _colors = colors[i, :nColor[i].item()].detach()
        dist = utils.pairwise_distances(_fakeB, _colors)
        argmin = dist.min(dim=1)[1]
        _fakeB_gif = _colors[argmin].transpose(0, 1).view(1, C, H, W)
        fakeB_gif.append(_fakeB_gif)
    fakeB_gif = torch.cat(fakeB_gif, dim=0)
    new_input = torch.cat([fakeB, realA, fakeB_gif, realA - fakeB_gif], dim=1)
    return new_input


def train(epoch, trLD, model, base_model, optimizer, fakeABPool):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = ['D_gan', 'D_real', 'D_fake', 'D_acc', 'G_gan', 'G_idl', 'G_gdl', 'G_total']
    epL = utils.AverageMeters(tags)
    N = max(1, round(len(trLD) * 0.1))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        realA, realB, colors = list(map(lambda x: preprocess(x).to(DEVICE), samples[:3]))
        nColor = samples[3].to(DEVICE)
        ################################################
        with torch.no_grad():
            initB = base_model(realA).tanh()
        fakeBs = []
        for _ in range(opts.k):
            fakeB = initB if not fakeBs else fakeBs[-1]
            new_input = iterative_input(fakeB, realA, colors, nColor)
            fakeB = (fakeB + model.netG(new_input)).tanh()
            fakeBs.append(fakeB)
        ################################################
        D_input = lambda A, B: torch.cat((A, B, pad_tl(diff_xy(B))), dim=1)
        realAB = D_input(realA, realB)
        fakeAB = D_input(realA, fakeBs[-1])

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
            optimizer.netD.step()

        # (2) Update G network
        optimizer.netG.zero_grad()
        fake_logits = model.netD(fakeAB)
        g_gan = utils.compute_G_loss(fake_logits, method='GAN')
        g_idl = sum(Lp(1)(fakeB, realB) for fakeB in fakeBs) / opts.k
        g_gdl = sum(2 * Lp(1)(diff_xy(fakeB).abs(), diff_xy(realB).abs()) for fakeB in fakeBs) / opts.k

        loss_g = g_gan  + g_idl * 100 + g_gdl * 100
        loss_g.backward()
        if d_acc.item() > 0.25:
            optimizer.netG.step()

        # tags = ['D_gan', 'D_real', 'D_fake', 'D_acc', 'G_gan', 'G_idl', 'G_gdl', 'G_total']
        values = list(map(lambda x: x.item(), [d_gan, d_real, d_fake, d_acc, g_gan, g_idl, g_gdl, loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%20==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%20==0:
            board_vis(epoch, realA, fakeB, realB)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)
            
            
def main_train():
    print('==> create base model')
    base_model = UNet.UNet(3, 3, ch=64)
    #base_model.load_state_dict(torch.load('ccdnet_results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,nogan/k2/ep-0060.pt')['model_netG'])
    base_model = nn.DataParallel(base_model.to(DEVICE))
    
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
        train(epoch, trLD, model, base_model, optimizer, fakeABPool)
        if epoch%10==0:
            save_checkpoint(epoch, model, optimizer, opts)
        if epoch%50==0:
            evaluate(epoch, evalLD, model)
            
            
if __name__ == '__main__':
    opts.board = SummaryWriter(os.path.join(opts.saveDir, 'board'))
    main_train()
    opts.board.close()