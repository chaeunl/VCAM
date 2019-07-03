import torch
import torch.nn as nn
from torch.nn import functional

class Flatten(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Flatten, self).__init__(*args, **kwargs)
        self.__module__ = 'flatten'

    def forward(self, input):
        return input.view( input.size(0), -1 ) #Assuming input comes as a packed 'batch'!

#####################################################

class sgn_func(torch.autograd.Function):

    @staticmethod #correct usage?
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return input.sign() #just sign

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        #Straight-Through Estimator.
        grad_input = grad_output.clone()
        grad_input[input >=  1] = 0
        grad_input[input <= -1] = 0
        return grad_input

class Sgn_act(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sgn_act, self).__init__(*args, **kwargs)
        self.__module__ = 'sgn_act'

    def forward(self, input):
        return sgn_func.apply(input)

class zeroone_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.sign().clamp_(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[input >=  1] = 0
        grad_input[input <= -1] = 0
        return grad_input

class ZeroOneAct(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ZeroOneAct, self).__init__(*args, **kwargs)
        self.__module__ = 'zerooneact'

    def forward(self, input):
        return zeroone_func.apply(input)


#####################################################

class BinLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinLinear, self).__init__(*args, **kwargs)
        self.__module__ = 'binlinear'

    def forward(self, input):
        if not hasattr(self.weight, 'original_data'): #save original data once.
            self.weight.original_data = self.weight.data.clone()

        self.weight.data.sign_() #apply sign function to the weight data
        return functional.linear(input, self.weight, self.bias)


class BinConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__module__ = 'binconv2d'

    def forward(self, input):
        if not hasattr(self.weight, 'original_data'):
            self.weight.original_data = self.weight.data.clone()

        self.weight.data.sign_()
        return functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


