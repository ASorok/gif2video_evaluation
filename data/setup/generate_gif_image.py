import os, sys, glob, progressbar
from PIL import Image
import numpy as np

## settings
s_down = 1  # spatial downsample factor
gif_N = 32  # size of color pallete
dither_opt = 'nodither' # dither or not

# src and dst directory
frameRoot = '../face/expand1.5_size256/'
gifRoot = '../face_gif_image_PIL/expand1.5_size256_s{}_g{}_{}/'.format(s_down, gif_N, dither_opt)

## generate gif images
videos = glob.glob(os.path.join(frameRoot,'*.avi'))
for video in progressbar.progressbar(videos):
    # video = videos[0]
    frameDir = video
    gifDir = os.path.join(gifRoot, video.split('/')[-1])
    os.system('mkdir -p ' + gifDir)

    nFrame = len(glob.glob(os.path.join(frameDir, '*.jpg')))
    for j in range(1, nFrame+1):
        # j = 1
        jpgFile = '{}/frame_{:06d}.jpg'.format(frameDir, j)
        gifFile = '{}/frame_{:06d}.gif'.format(gifDir, j)

        im = Image.open(jpgFile)
        # spatial downsample
        if s_down != 1:
            W, H = im.size
            W, H = W // s_down, H // s_down
            im = im.resize((W, H), Image.ANTIALIAS)
        # gify
        if dither_opt == 'nodither':
            # im.quantize(colors=gif_N, method=0).save(gifFile)
            im.convert(mode='P', dither=Image.NONE, palette=Image.ADAPTIVE, colors=gif_N).save(gifFile)
        elif dither_opt == 'dither':
            raise NotImplementedError
            # I haven't figured out how to use PIL to do quantization & dithering at the same time
            # I tried:
            #     im.convert(mode='P', dither=Image.FLOYDSTEINBER, palette=Image.ADAPTIVE, colors=gif_N)
            # but the result gif image is not dithered.
            # I found a github repo (https://github.com/hbldh/hitherdither) that maybe useful
        else:
            raise NotImplementedError
