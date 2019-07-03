import torch
from torch.nn import functional
from torch.autograd import Variable

import FindFromLUT_cuda as FFL
#from nn_tools import *
from nn_fundamentals import BinConv2d

from nn_tools import dp
import time

class BinConv2d_LUT(BinConv2d):
    def __init__(self, *args, sWL=32, sW=27, **kwargs):
        #self.LUT = None
        #self.alpha = None
        #self.has_neuron = False
        #self.rand_seq = None

        self.vref_base = 0.86
        self.sWL = sWL
        self.sW = sW

        self.is_lut_mode = False
        self.reset_memory = True

        super().__init__(*args, **kwargs)
        self.__module__ = 'binconv2d_lut'


    def forward(self, input):
        #get and register in/out shape automatically.
        #note that batch size might be different if batch division is not even.

        if self.is_lut_mode:
            self.is_lut_mode = False
            if hasattr(self, 'weight_n'):
                self.weight.add_( -self.weight_n ) #restore combined weight mode

        self.in_shape = input.size() 
        output = super().forward(input.to(self.weight.device)) #handles dev-to-dev co-op
        self.out_shape = output.size() 

        return output

    def forward_bwsnn(self, input):

        if self.is_lut_mode:
            self.is_lut_mode = False
            if hasattr(self, 'weight_n'):
                self.weight.add_( -self.weight_n )

        return functional.conv2d(input.to(self.weight.device), self.weight, self.alpha_internal.to(self.weight.device), self.stride, self.padding, self.dilation, self.groups)


    def forward_lut(self, input, tune=False, bias_noise_std=0.0, same_seq_through_group=False):

        if not self.is_lut_mode:
            self.is_lut_mode = True
            self.register_parameter("weight_n", torch.nn.Parameter( -self.weight.clamp(max=0.0).to(self.weight.device), requires_grad=False )) #now negative weight is also save-able parameter.
            self.weight.clamp_(min=0.0)

        x = input.to(self.weight.device)

        sB = x.size(0)
        sO,sI,kw,kh = self.weight.size()
        nCBA = int(sI*kw*kh // self.sW)
        sSplit = int(self.sW // (kw*kh))

        #if self.reset_memory:
        #    self.reset_memory = False

        wxp = torch.Tensor(sB, sO, self.out_shape[2], self.out_shape[3], nCBA).type(torch.cuda.FloatTensor).to(self.weight.device)
        wxn = wxp.clone().to(self.weight.device)


        for i in range(nCBA):
            wxp[:,:,:,:,i] = functional.conv2d(x[:,i*sSplit:(i+1)*sSplit,:,:], self.weight[:,i*sSplit:(i+1)*sSplit,:,:], None, self.stride, self.padding, self.dilation, self.groups)
        
            wxn[:,:,:,:,i] = functional.conv2d(x[:,i*sSplit:(i+1)*sSplit,:,:], self.weight_n[:,i*sSplit:(i+1)*sSplit,:,:], None, self.stride, self.padding, self.dilation, self.groups)

        if tune:
            #add bias and clamp
            sOW = wxp.size(2)
            sOH = wxp.size(3)

            sbias = self.sbias.to(self.weight.device).expand(sB, sO, sOW, sOH, nCBA)

            #self.wxp = torch.where(sbias > 0, self.wxp + sbias, self.wxp)
            #self.wxn = torch.where(sbias > 0, self.wxn - sbias, self.wxn)
            wxp.add_( sbias.clamp(min=0.0)        ).clamp_(min=0.0, max=self.sWL)
            wxn.add_( sbias.clamp(max=0.0).neg_() ).clamp_(min=0.0, max=self.sWL)

        #making indices of monte-trials: 0~99 if 100 monte trial is done.
        if same_seq_through_group:
            rand_seq = self.rand_seq[0:sO*sOW*sOH].view([sO,sOW,sOH]).to(self.weight.device)
            lut_no   = rand_seq.expand(nCBA,sO,sOW,sOH).permute(1,2,3,0).expand([sB,sO,sOW,sOH,nCBA]).contiguous()
        else:
            #different random sequence is applied to each group. (Default)
            rand_seq = self.rand_seq[0:sO*sOW*sOH*nCBA].view([sO,sOW,sOH,nCBA]).to(self.weight.device)
            lut_no   = rand_seq.expand([sB,sO,sOW,sOH,nCBA]).to(self.weight.device).contiguous()

        vp = torch.zeros_like(wxp)
        vn = torch.zeros_like(wxn)

        vrefround = round(self.vref_base, 3) #note that built-in round works as 'nearest even'.
        vref = torch.full_like(wxp, self.LUTidx1[vrefround]).type(torch.cuda.FloatTensor)
        #vref = torch.tensor([vrefround])

        with torch.cuda.device(self.weight.device): #ensure operation is done in designated device. what's the difference?
            FFL.get_vout_from_lut(vp, wxp, vref, lut_no, self.LUT)
            FFL.get_vout_from_lut(vn, wxn, vref, lut_no, self.LUT)
            #print(x[0:5,0:5,0,0])

        #err = (vp.cpu() <= -50)
        #if err.any():
        #    raise Exception("cuda: Error @ {}, {}".format(vp.min(), vp.argmin()))

        #err = (vn.cpu() <= -50)
        #if err.any():
        #    raise Exception("cuda: Error @ {}, {}".format(vn.min(), vn.argmin()))

        #check error and post-process

        vp = vp.sum(4).div_(nCBA)
        vn = vn.sum(4).div_(nCBA)
        #vout = vout.sum(4) / nCBA

        if bias_noise_std > 0.0:
            vp.add_( vp.clone().normal_(0.0, bias_noise_std/sB) )
            vn.add_( vn.clone().normal_(0.0, bias_noist_std/sB) )
        
        #make op_amp vout.
        vp.add_( vn.neg_() ) #vp = vp - vn. now works as vout.

        #numerical stability control?

        return vp #as vout.


    def calc_and_set_alpha(self, touching_bn_layer):

        cv_b = self.bias
        bn_m = touching_bn_layer.running_mean
        bn_v = touching_bn_layer.running_var
        bn_g = touching_bn_layer.weight
        bn_b = touching_bn_layer.bias

        t0 = bn_v.add(touching_bn_layer.eps).sqrt_().div_(bn_g)    #sqrt(σ^2+ε)/γ
        alpha = cv_b.add(-bn_m).addcmul_(bn_b, t0).ceil() # b + (-μ) + (sqrt(σ^2+ε)/γ .* β)

        self.alpha_internal = alpha #internal conv use

        #alpha expansion is required for bias distribution
        sOW = self.out_shape[2]
        sOH = self.out_shape[3]

        alpha_ex = alpha.repeat(sOH, sOW, 1)
        alpha_ex.transpose_(0,2)
        self.alpha = alpha_ex
        self.alpha_original = alpha_ex


    def set_distributed_bias(self):
        I = self.in_channels
        kw, kh = self.kernel_size

        nCBA = int(I*kw*kh // self.sW)

        sbias = self.alpha // nCBA
        remainder = self.alpha % nCBA

        sbias = sbias.repeat(nCBA, 1, 1, 1)
        for k in range(nCBA):
            incr = (remainder > 0).type(torch.cuda.FloatTensor)
            sbias[k,:,:,:] += incr
            remainder -= incr

        bias_lim = self.sWL - self.sW
        sbias.clamp_(min=-bias_lim, max=bias_lim)
        sbias = sbias.permute(1,2,3,0)

        self.register_parameter('sbias', torch.nn.Parameter(sbias, requires_grad=False))
