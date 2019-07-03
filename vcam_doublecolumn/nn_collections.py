import torch.nn as nn
from collections import OrderedDict

from nn_fundamentals import *
from binlinear_lut import BinLinear_LUT
from binconv2d_lut import BinConv2d_LUT

def BNN_784_512_512_512_10():
    return nn.Sequential( OrderedDict([
        ('flatten', Flatten()),

        ('fc1',  BinLinear_LUT(28*28, 512)),
        ('bn1',  nn.BatchNorm1d(512)),
        ('act1', ZeroOneAct()),

        ('fc2',  BinLinear_LUT(512, 512, sWL=32, sW=16)),
        ('bn2',  nn.BatchNorm1d(512)),
        ('act2', ZeroOneAct()),      

        ('fc3',  BinLinear_LUT(512, 512, sWL=32, sW=16)),
        ('bn3',  nn.BatchNorm1d(512)),
        ('act3', ZeroOneAct()),

        ('fc4',  BinLinear_LUT(512, 10, sWL=32, sW=16)),
        ('bn4',  nn.BatchNorm1d(10)),
    ]))


def BNN_conv6_fc3_std():
    return nn.Sequential( OrderedDict([
        ('conv1', BinConv2d_LUT(3, 126, 3, 1, 1)), # in/out channels, kernel size, stride, padding
        ('bn1',   nn.BatchNorm2d(126)),
        ('act1',  ZeroOneAct()),

        ('conv2', BinConv2d_LUT(126, 126, 3, 1, 1, sWL=32, sW=27)),
        ('bn2',   nn.BatchNorm2d(126)),
        ('act2',  ZeroOneAct()),

        ('pool1', nn.MaxPool2d(2, 2, 0)), #kernel size, stride, padding

        ('conv3', BinConv2d_LUT(126, 252, 3, 1, 1, sWL=32, sW=27)), # in/out channels, kernel size, stride, padding
        ('bn3',   nn.BatchNorm2d(252)),
        ('act3',  ZeroOneAct()),

        ('conv4', BinConv2d_LUT(252, 252, 3, 1, 1, sWL=32, sW=27)),
        ('bn4',   nn.BatchNorm2d(252)),
        ('act4',  ZeroOneAct()),

        ('pool2', nn.MaxPool2d(2, 2, 0)), #kernel size, stride, padding
       
        ('conv5', BinConv2d_LUT(252, 504, 3, 1, 1, sWL=32, sW=27)), # in/out channels, kernel size, stride, padding
        ('bn5',   nn.BatchNorm2d(504)),
        ('act5',  ZeroOneAct()),

        ('conv6', BinConv2d_LUT(504, 504, 3, 1, 1, sWL=32, sW=27)),
        ('bn6',   nn.BatchNorm2d(504)),
        ('act6',  ZeroOneAct()),

        ('pool3', nn.MaxPool2d(2, 2, 0)), #kernel size, stride, padding

        ('flatten', Flatten()),

        ('fc7',   BinLinear_LUT(504*4*4, 1008, sWL=32, sW=28)),
        ('bn7',   nn.BatchNorm1d(1008)),
        ('act7',  ZeroOneAct()),

        ('fc8',   BinLinear_LUT(1008, 1008, sWL=32, sW=28)),
        ('bn8',   nn.BatchNorm1d(1008)),
        ('act8',  ZeroOneAct()),

        ('fc9',   BinLinear_LUT(1008, 10, sWL=32, sW=28)),
        ('bn9',   nn.BatchNorm1d(10))
    ]))
