import torch
import torch.nn as nn
import torch.nn.functional as FF

import progressbar
from skimage.measure import compare_ssim as comp_ssim

from model_train import create_dataloader, create_model, initialize
from torch_models import UNet, SlomoNet, Discriminator
from data import gif_faces_utils, utils
from data.utils import *
from utils import *


opts=edict()

opts.k=0#type of cddnet

#data processing parametres
opts.tCrop=5#sequence length
opts.sCrop=256#spatial patch size
opts.tStride=10#temporal downsampling

#training parametres
opts.batch_size = 2
opts.pool_size=30#image pool size
opts.nEpoch=30

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

def create_model():
    model = edict()
    model.netG = SlomoNet.netSlomo(maxFlow=30)
    model.netD = Discriminator.Discriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = nn.DataParallel(model[key].to(DEVICE))
    return model

def iterative_input(fakeB, realA, colors, nColor=32):
    # fakeB_gif = fakeB
    B, C, H, W = fakeB.shape
    fakeB_gif = []
    for i in range(B):
        _fakeB, _realA = fakeB[i].detach(), realA[i].detach()
        _fakeB = _fakeB.view(C, H*W).transpose(0, 1)
        _colors = colors[i, :nColor].detach()
        dist = pairwise_distances(_fakeB, _colors)
        argmin = dist.min(dim=1)[1]
        _fakeB_gif = _colors[argmin].transpose(0, 1).view(1, C, H, W)
        fakeB_gif.append(_fakeB_gif)
    fakeB_gif = torch.cat(fakeB_gif, dim=0)
    new_input = torch.cat([fakeB, realA, fakeB_gif, realA - fakeB_gif], dim=1)
    return new_input

comp_psnr = lambda x, y: rmse2psnr((x - y).abs().pow(2).mean().pow(0.5).item(), maxVal=2.0)
tensor2im = lambda x: np.moveaxis(x.cpu().numpy(), 0, 2)


def evaluate(epoch, evalLD, model, color_model1, color_model2=None):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    netG = model.netG
    netG.eval()

    tags = ['PSNR', 'PSNR_gif', 'SSIM', 'SSIM_gif']
    epL = AverageMeters(tags)
    for i, (gif0s, gif1s, targets, color0s, color1s) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (gif0s, gif1s, targets) = 0, next(iter(evalLD))
        # gif0s, gif1s: 1, T, C, H, W
        # targets: 1, T, L, C, H, W
        _, T, L, C, H, W = targets.size()
        for j in range(T):
            gif0, gif1, target, color0, color1 = gif0s[:, j], gif1s[:, j], targets[:, j], color0s[:, j], color1s[:, j]
            gif0, gif1, target, color0, color1 = list(map(lambda x: preprocess(x).to(DEVICE), (gif0, gif1, target, color0, color1)))
            ts = np.linspace(0, 1, L)[1:L-1].tolist()
            with torch.no_grad():
                ################################################
                I0 = color_model1(gif0).tanh()
                for _ in range(opts.k):
                    new_input = iterative_input(I0, gif0, color0, 32)
                    I0 = (I0 + color_model2(new_input)).tanh()
                I1 = color_model1(gif1).tanh()
                for _ in range(opts.k):
                    new_input = iterative_input(I1, gif1, color1, 32)
                    I1 = (I1 + color_model2(new_input)).tanh()
                ################################################
                
                Its, F01, F10, Ft1s, Ft0s, Vt0s = model.netG(gif0, gif1, I0, I1, ts)
                    
                pred = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
                pred_gif = torch.cat(list((gif0 if t<=0.5 else gif1).unsqueeze(1) for t in np.linspace(0, 1, L).tolist()), dim=1)
            
            psnr = comp_psnr(pred, target)
            psnr_gif = comp_psnr(pred_gif, target)

            ssim, ssim_gif = 0.0, 0.0
            for k in range(L):
                ssim += comp_ssim(tensor2im(pred[0, k]), tensor2im(target[0, k]), data_range=2.0, multichannel=True)/L
                ssim_gif += comp_ssim(tensor2im(pred_gif[0, k]), tensor2im(target[0, k]), data_range=2.0, multichannel=True)/L

            values = [psnr, psnr_gif, ssim, ssim_gif]
            assert len(tags) == len(values)
            for tag, value in zip(tags, values):
                epL[tag].update(value, 1.0/T)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, 50, state))

    
