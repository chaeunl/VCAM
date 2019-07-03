import torch
from torch.nn import functional, Parameter
from torch.autograd import Variable

import FindFromLUT_cuda as FFL
#from nn_tools import *
from nn_fundamentals import BinLinear

class BinLinear_LUT(BinLinear): #inherits BinLinear with additional functions!
    def __init__(self, *args, sWL=32, sW=16, **kwargs):
        #self.LUT = None
        #now LUT comes with LUTidx0 and LUTidx1 dict.
        #self.alpha = None
        #self.has_neuron = False
        #self.rand_seq = None

        #dict-able parameters
        self.vref_base = 0.86
        self.sWL = sWL
        self.sW = sW

        super(BinLinear_LUT, self).__init__(*args, **kwargs)
        self.__module__ = 'binlinear_lut'

    def forward(self, input):
        return super().forward(input.to(self.weight.device))

    def forward_bwsnn(self, input):
        #if not hasattr(self.weight, 'original_data'):
        #    self.weight.original_data = self.weight.data.clone()

        #self.weight.data.sign_()
        return functional.linear(input.to(self.weight.device), self.weight, self.alpha.to(self.weight.device))

    #advanced method for intuitive arguments... 
    def forward_lut(self, input, tune=False, bias_noise_std=0.0, same_seq_through_group=False):

        x = input.to(self.weight.device)  # [BATCH_DIM * INPUT_DIM]

        sB,sI = x.size()
        sO, _ = self.weight.size()

        nCBA = sI // self.sW  #should be perfectly divided! else, exception occurs at torch.stack.

        #we = w.expand(sB, sO, sI)                  #DIM: [B,O,I]
        #xe = x.expand(sO, sB, sI).transpose(0,1)   #DIM: [B,O,I]

        rz = self.weight.expand(sB, sO, sI) * x.expand(sO, sB, sI).transpose_(0,1) #analogy: z[o,b] = sum(w[o,i]*x'[i,b]) = w[o,1]*x'[1,b] + w[o,2]*x'[2,b] + ... =: z[o,1,b] + z[o,2,b] + ...

        rz = list(rz.split(self.sW, 2)) #split through input dim.
        rz = torch.stack(rz, 3) #stack through new dimension.
        #rz = torch.stack(rz.split(self.sW, 2), 3)

        #wp_i = rz > 0 #returns True/False as 1/0. 
        #wn_i = rz < 0
        wps = (rz > 0).type(torch.cuda.FloatTensor).sum(2) #now sum through the input dimension.
        wns = (rz < 0).type(torch.cuda.FloatTensor).sum(2)
        #wps, wns contains integer-like weight values?

        if tune:
            #add bias and clamp
            sbias = self.sbias.to(self.weight.device).expand(sB, sO, nCBA)

            wps.add_( sbias.clamp(min=0.0)        ).clamp_(min=0.0, max=self.sWL).to(self.weight.device)
            wns.add_( sbias.clamp(max=0.0).neg_() ).clamp_(min=0.0, max=self.sWL).to(self.weight.device)


        #making indices of monte-trials: 0~99 if 100 monte trial is done.
        #vout_raw = wps.clone() #DIM: [B,O,G]
        vout = torch.zeros_like(wps)
        
        if same_seq_through_group:
            rand_seq = self.rand_seq[0:sO].to(self.weight.device) #DIM:[O]
            lut_no   = rand_seq.expand(nCBA,sO).transpose(0,1).expand([sB,sO,nCBA]).contiguous()
        else:
            #different random sequence is applied to each group. (Default)
            rand_seq = self.rand_seq[0:sO*nCBA].view(sO,nCBA).to(self.weight.device) #DIM:[O,G]
            lut_no   = rand_seq.expand(sB,sO,nCBA).contiguous()

        #find and assign corresponding vout_raw in pre-existing LUT.
        #args: vout, idx_post, idx_neg, monte_no, lut_table

        vp = torch.zeros_like(wps)
        vn = torch.zeros_like(wns)

        vrefround = round(self.vref_base, 3)
        vref = torch.full_like(wps, self.LUTidx1[vrefround]).type(torch.cuda.FloatTensor)

        with torch.cuda.device(self.weight.device):
            FFL.get_vout_from_lut(vp, wps, vref, lut_no, self.LUT)
            FFL.get_vout_from_lut(vn, wns, vref, lut_no, self.LUT)

        #check error and post-process
        #err = (vp.cpu() <= -50.0)
        #if err.any():
        #    raise Exception("cuda: Error @ {}, {}".format(vp.min(), vp.argmin()))

        #err = (vn.cpu() <= -50.0)
        #if err.any():
        #    raise Exception("cuda: Error @ {}, {}".format(vn.min(), vn.argmin()))

        vp = vp.sum(2).div_(nCBA)
        vn = vn.sum(2).div_(nCBA)

        if bias_noise_std > 0.0:
            vp.add_( vp.clone().normal_(0.0, bias_noise_std/sB) )
            vn.add_( vn.clone().normal_(0.0, bias_noise_std/sB) )
        
        #make op_amp vout.
        vp.add_( vn.neg_() )

        #numerical stability control?

        return vp #as vout.

    def calc_and_set_alpha(self, touching_bn_layer):
        cv_b = self.bias.data
        bn_m = touching_bn_layer.running_mean
        bn_v = touching_bn_layer.running_var
        bn_g = touching_bn_layer.weight.data
        bn_b = touching_bn_layer.bias.data

        t0 = bn_v.add(touching_bn_layer.eps).sqrt_().div_(bn_g)    #sqrt(σ^2+ε)/γ
        alpha = cv_b.add(-bn_m).addcmul_(bn_b, t0).ceil() # b + (-μ) + (sqrt(σ^2+ε)/γ .* β)

        self.alpha = alpha #used in vref & bias shifting!
        self.alpha_original = alpha

    def set_distributed_bias(self):
        I = self.in_features

        nCBA = int(I // self.sW)

        sbias = self.alpha // nCBA
        remainder = self.alpha % nCBA

        sbias = sbias.repeat(nCBA, 1)
        for k in range(nCBA):
            incr = (remainder > 0).type(torch.cuda.FloatTensor)
            sbias[k,:] += incr
            remainder -= incr

        bias_lim = self.sWL - self.sW
        sbias.clamp_(min=-bias_lim, max=bias_lim).transpose_(0,1)
        self.register_parameter('sbias', torch.nn.Parameter(sbias, requires_grad=False)) #now sbias can be called and saved using state_dict.

