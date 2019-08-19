import torch
import torch.utils.data as DD

import imageio
from PIL import Image
import numpy as np
import random

trainSplit = 'data/split/face_train.txt'
validSplit = 'data/split/face_valid.txt'

def getPaletteInRgb(gifFile):
    gif = Image.open(gifFile)
    assert gif.mode == 'P', "image should be palette mode"
    nColor = len(gif.getcolors())
    pal = gif.getpalette()
    colors = list(list(pal[i:i+3]) for i in range(0, len(pal), 3))
    return nColor, colors

class gif_faces_ct_train(DD.Dataset):
    def __init__(self, inputRoot, targetRoot, tCrop=5, sCrop=256):
        super(gif_faces_ct_train, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.videoList = self.get_videoList(trainSplit)
        self.sCrop = sCrop
        self.tCrop = tCrop

    def get_videoList(self, splitFile):
        videoList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append([VID, nFrm])
        return videoList

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        inputFileName = lambda t: '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t)
        targetFileName = lambda t: '{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, t)
        inputPalette = lambda t: getPaletteInRgb(inputFileName(t))[1]
        inputFile = lambda t: imageio.imread(inputFileName(t))[:,:,:3]
        targetFile = lambda t: imageio.imread(targetFileName(t))
        # config
        tCrop, sCrop = self.tCrop, self.sCrop
        tFlip, sFlip = random.random() > 0.5, random.random() > 0.5
        # temporal crop and flip
        sFrm, eFrm = 1, nFrm
        t0 = random.randrange(sFrm, eFrm-self.tCrop+2)
        ts = list(range(t0, t0+self.tCrop))
        if tFlip: ts = ts[::-1]
        # spatial crop and flip
        gif = inputFile(t0)
        H, W, _ = gif.shape
        y, x = random.randrange(H - sCrop + 1), random.randrange(W - sCrop + 1)
        def proc(im):
            crop = im[y:y+sCrop, x:x+sCrop]
            return np.flip(crop, axis=1) if sFlip else crop
        gif = np.asarray(list(proc(inputFile(t)) for t in ts))
        target = np.asarray(list(proc(targetFile(t)) for t in ts))
        # get color palette
        colors = np.asarray(list(inputPalette(t) for t in ts))
        # numpy to tensor
        gif = torch.ByteTensor(np.moveaxis(gif, 3, 1).copy()) # T C H W
        target = torch.ByteTensor(np.moveaxis(target, 3, 1).copy()) # T C H W
        colors = torch.ByteTensor(colors)
        return gif, target, colors

    def __len__(self):
        return len(self.videoList)

class gif_faces_ct_eval(DD.Dataset):
    def __init__(self, inputRoot, targetRoot, tStride=10, tCrop=5):
        super(gif_faces_ct_eval, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.videoList = self.get_videoList(validSplit)
        self.tStride = tStride
        self.tCrop = tCrop

    def get_videoList(self, splitFile=validSplit):
        videoList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append([VID, nFrm])
        return videoList

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        inputFileName = lambda t: '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t)
        targetFileName = lambda t: '{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, t)
        inputPalette = lambda t: getPaletteInRgb(inputFileName(t))[1]
        inputFile = lambda t: imageio.imread(inputFileName(t))[:,:,:3]
        targetFile = lambda t: imageio.imread(targetFileName(t))
        #
        ts_finish = np.arange(self.tCrop, nFrm+1, self.tStride)
        ts_start = ts_finish - self.tCrop + 1
        gif0s = np.asarray(list(inputFile(a) for a in ts_start))
        gif1s = np.asarray(list(inputFile(b) for b in ts_finish))
        targets = np.asarray(list(list(targetFile(t) for t in range(a, b+1)) for a, b in zip(ts_start, ts_finish)))
        color0s = np.asarray(list(inputPalette(a)[1] for a in ts_start))
        color1s = np.asarray(list(inputPalette(b)[1] for b in ts_finish))
        # numpy to tensor
        gif0s = torch.ByteTensor(np.moveaxis(gif0s, 3, 1)) # N C H W
        gif1s = torch.ByteTensor(np.moveaxis(gif1s, 3, 1)) # N C H W
        targets = torch.ByteTensor(np.moveaxis(targets, 4, 2)) # N T C H W
        color0s = torch.ByteTensor(color0s) # N P 3
        color1s = torch.ByteTensor(color1s) # N P 3
        return gif0s, gif1s, targets, color0s, color1s

    def __len__(self):
        return len(self.videoList)
    
    
class gif_faces_train(DD.Dataset):
    def __init__(self, inputRoot, targetRoot, patchSize=256):
        super(gif_faces_train, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.imageList = self.get_imageList(trainSplit)
        self.patchSize = patchSize

    def get_imageList(self, splitFile):
        imageList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                for i in range(1, nFrm+1):
                    imageList.append('{}/frame_{:06d}'.format(VID, i))
        return imageList

    def __getitem__(self, index):
        imID = self.imageList[index]
        inputFile = '{}/{}.gif'.format(self.inputRoot, imID)
        targetFile = '{}/{}.jpg'.format(self.targetRoot, imID)
        # get input gif and target image
        input = imageio.imread(inputFile)[:, :, :3]
        target = imageio.imread(targetFile)
        # get random patch
        H, W, _ = input.shape
        y1 = random.randrange(H - self.patchSize + 1)
        x1 = random.randrange(W - self.patchSize + 1)
        input = input[y1:y1+self.patchSize, x1:x1+self.patchSize, :]
        target = target[y1:y1+self.patchSize, x1:x1+self.patchSize, :]
        # get color palette
        nColor, colors = getPaletteInRgb(inputFile)
        # numpy to tensor
        input = torch.ByteTensor(np.moveaxis(input, 2, 0))
        target = torch.ByteTensor(np.moveaxis(target, 2, 0))
        colors = torch.ByteTensor(colors)
        return input, target, colors, nColor

    def __len__(self):
        return len(self.imageList)

class gif_faces_eval(DD.Dataset):
    def __init__(self, inputRoot, targetRoot, tDown=4):
        super(gif_faces_eval, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.tDown = tDown
        self.videoList, self.frameCount = self.get_videoList(validSplit)

    def get_videoList(self, splitFile):
        videoList = []
        frameCount = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append(VID)
                frameCount.append(nFrm)
        return videoList, frameCount

    def __getitem__(self, index):
        VID = self.videoList[index]
        nFrm = self.frameCount[index]
        # get video gif frames and target frames
        input, target = [], []
        for i in range(1, nFrm+1, self.tDown):
            inputFile = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, i)
            targetFile = '{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, i)
            input.append(imageio.imread(inputFile)[:, :, 0:3])
            target.append(imageio.imread(targetFile))

        # get video gif color palettes
        nColor, colors = [], []
        for i in range(1, nFrm+1, self.tDown):
            inputFile = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, i)
            _nColor, _colors = getPaletteInRgb(inputFile)
            nColor.append(_nColor)
            colors.append(_colors)

        # numpy to tensor
        im2tensor = lambda x: torch.ByteTensor(np.moveaxis(np.asarray(x), 3, 1))
        input, target = im2tensor(input), im2tensor(target) # T [RGB] H W
        nColor = torch.LongTensor(nColor)
        colors = torch.ByteTensor(colors)
        return input, target, colors, nColor

    def __len__(self):
        return len(self.videoList)