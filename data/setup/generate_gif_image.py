import os, sys, glob, progressbar
from PIL import Image
import numpy as np
from math import floor

## settings
s_down = 1  # spatial downsample factor
gif_N = 32  # size of color pallete
dither_opt = 'dither' # dither or not

# src and dst directory
frameRoot = '../face/expand1.5_size256/'
gifRoot = '../face_gif_image_PIL/expand1.5_size256_s{}_g{}_{}/'.format(s_down, gif_N, dither_opt)

##dithering functions
def apply_threshold(value):
    "Returns 0 or 255 depending where value is closer"
    return 255 * floor(value/128)
def floyd_steinberg_dither(new_img):
    """
    https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    Pseudocode:
    for each y from top to bottom
       for each x from left to right
          oldpixel  := pixel[x][y]
          newpixel  := find_closest_palette_color(oldpixel)
          pixel[x][y]  := newpixel
          quant_error  := oldpixel - newpixel
          pixel[x+1][y  ] := pixel[x+1][y  ] + quant_error * 7/16
          pixel[x-1][y+1] := pixel[x-1][y+1] + quant_error * 3/16
          pixel[x  ][y+1] := pixel[x  ][y+1] + quant_error * 5/16
          pixel[x+1][y+1] := pixel[x+1][y+1] + quant_error * 1/16
    find_closest_palette_color(oldpixel) = floor(oldpixel / 256)
    """

    new_img = new_img.convert('RGB')
    pixel = new_img.load()

    x_lim, y_lim = new_img.size

    for y in range(1, y_lim):
        for x in range(1, x_lim):
            red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]

            red_newpixel = apply_threshold(red_oldpixel)
            green_newpixel = apply_threshold(green_oldpixel)
            blue_newpixel = apply_threshold(blue_oldpixel)

            pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel

            red_error = red_oldpixel - red_newpixel
            blue_error = blue_oldpixel - blue_newpixel
            green_error = green_oldpixel - green_newpixel

            if x < x_lim - 1:
                red = pixel[x+1, y][0] + round(red_error * 7/16)
                green = pixel[x+1, y][1] + round(green_error * 7/16)
                blue = pixel[x+1, y][2] + round(blue_error * 7/16)

                pixel[x+1, y] = (red, green, blue)

            if x > 1 and y < y_lim - 1:
                red = pixel[x-1, y+1][0] + round(red_error * 3/16)
                green = pixel[x-1, y+1][1] + round(green_error * 3/16)
                blue = pixel[x-1, y+1][2] + round(blue_error * 3/16)

                pixel[x-1, y+1] = (red, green, blue)

            if y < y_lim - 1:
                red = pixel[x, y+1][0] + round(red_error * 5/16)
                green = pixel[x, y+1][1] + round(green_error * 5/16)
                blue = pixel[x, y+1][2] + round(blue_error * 5/16)

                pixel[x, y+1] = (red, green, blue)

            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x+1, y+1][0] + round(red_error * 1/16)
                green = pixel[x+1, y+1][1] + round(green_error * 1/16)
                blue = pixel[x+1, y+1][2] + round(blue_error * 1/16)

                pixel[x+1, y+1] = (red, green, blue)

    return new_img

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
            floyd_steinberg_dither(im.convert(mode='P', dither=Image.NONE, palette=Image.ADAPTIVE, colors=gif_N)).save(gifFile)
        else:
            raise NotImplementedError
