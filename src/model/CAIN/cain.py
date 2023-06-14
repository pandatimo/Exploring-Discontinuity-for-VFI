import math
import numpy as np

import torch
import torch.nn as nn

from model.CAIN.common import *


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(5, 12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2, x3, x4):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)
        feats3 = self.shuffler(x3)
        feats4 = self.shuffler(x4)
        feats = self.interpolate(feats1, feats2, feats3, feats4)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, frames):
        x1 = frames[0]
        x2 = frames[1]
        x3 = frames[2]
        x4 = frames[3]

        ref = x2
        x1, _ = sub_mean(x1)
        x2, m2 = sub_mean(x2)
        x3, m3 = sub_mean(x3)
        x4, _ = sub_mean(x4)

        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)
            x3 = paddingInput(x3)
            x4 = paddingInput(x4)

        feats, Dmap = self.encoder(x1, x2, x3, x4)
        out = self.decoder(feats)

        if not self.training:
            out = paddingOutput(out)
            Dmap = paddingOutput(Dmap)

        mi = (m2 + m3) / 2
        out += mi
        out = ref * Dmap + out * (1. - Dmap)

        return out
