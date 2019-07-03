import torch, nn_tools
import FindFromLUT_cuda as FFL

p = torch.rand(50,50,50,50,50).mul_(16).floor_().cuda(1)
q = torch.rand(50,50,50,50,50).mul_(16).floor_().cuda(1)
r = torch.rand(50,50,50,50,50).mul_(100).floor_().cuda(1)
LUT0, idx0, idx1 = nn_tools.load_LUT_cuda("path_circuitsim/gv0lv0.1/LUT_results/sample0")
LUT0 = LUT0.cuda(1)
vout = torch.zeros_like(p).cuda(1)

import time


FFL.get_vout_from_lut(vout,p,q,r,LUT0)


print("dbg0: sum of zeros: {}\n".format( (vout==0).sum() ))

print(vout[0:10,0:10,0,0,0])
