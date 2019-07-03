import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import math, os

from nn_tools import *
#from nn_base import * #future: nn_parameters

class NNVcamDriver():

    def __init__(self, networkcontainer, *args, **kwargs):
        self.NC = networkcontainer #get neural network and parameters.


    def run(self): #run whole vcam process.

        #step 1: verify trained model with standard forwarding.
        self.NC.network.eval() #drive network with evaluation mode.
        acc, loss = self.test()
        dp("NNVD: 1st test result: acc. {}%, loss {}.\n".format(acc*100., loss))

        #step 2: setup neuron environments: alpha, LUT, properties, ...
        neuron_lists = self.setup_neuron()

        #step 3: run global/local vref/bias tuning.
        self.run_sbias_tuning(neuron_lists)

        #write models
        if not os.path.exists("path_output/" + self.NC.env.envname_output):
            os.mkdir("path_output/" + self.NC.env.envname_output)

        save_network(self.NC.network, None, "path_output/" + self.NC.env.envname_output + "/model_cal.pth")

        #step 4: test for shifted vref with modified model.
        voltage_lists = np.arange(0.76, 0.96, 0.04)
        self.test_vref_shift(neuron_lists, voltage_lists)


    def test(self, mode='default', usr_batch=None, use_all_batch_per_iter=True, iter_count=0):

        starttime = time.time()

        #self.NC.network.eval() #sets the module in evalutation mode.

        loss = 0
        corrects = 0
        total = 0

        if usr_batch is not None:
            batch_in = usr_batch
        else:
            batch_in = self.NC.dataset[1]

        if not use_all_batch_per_iter:
            blen = len(batch_in)
            bidx = int(iter_count % blen)
            batch_in = [batch_in[bidx]]

        reset_firing_rate(self.NC.network, namespace = 'fr0' if mode == 'bwsnn' else 'fr')

        ### memory management code ###

        for m in list(self.NC.network.modules()):
            if hasattr(m, 'reset_memory'):
                m.reset_memory = True

        ### end memory management  ###

        with torch.no_grad():
        
            for I, L in batch_in:

                if mode == 'bwsnn': #for base firing rate verification
                    O = self.forward_op(I, 'bwsnn')
                elif mode == 'lut_sbias': #for LUT based calculation
                    O = self.forward_op(I, 'lut_sbias')
                elif mode == 'default': #for base test accuracy verification
                    O = self.NC.network(I)
                else:
                    raise Exception("Invalid mode input: {}!!".format(mode))

                #tie breaker lines?

                loss += self.NC.criterion(O.cpu(), L.cpu()).item()
                _, pred = torch.max(O.cpu(), 1, keepdim=True)
                corrects += pred.eq(L.cpu().view_as(pred)).sum().item()
                total += len(L)

        set_firing_rate(self.NC.network, namespace = 'fr0' if mode == 'bwsnn' else 'fr')

        acc = corrects/total
        loss = loss / len(batch_in)

        runtime = time.time() - starttime
        if runtime > 300:
            dp("\nNNVD: Test time = {}s\n".format(runtime))

        return acc, loss


    def forward_op(self, inputs, mode):
        x = inputs
        mdl = list(self.NC.network.modules())
        nml = len(mdl)


        for i, m in enumerate(mdl):
            if m.__module__ == "torch.nn.modules.container":
                continue #always comes first. Do nothing.

            is_BN_next = (i+1 < nml) and (mdl[i+1].__module__ == "torch.nn.modules.batchnorm") #batchnorm at next layer?
            #has_alpha  = hasattr(m, 'alpha') #has alpha?
            is_neuron  = hasattr(m, 'has_neuron') and m.has_neuron #has neuron?

            if is_BN_next and is_neuron:
                if mode == 'bwsnn':
                    x = m.forward_bwsnn(x)
                    add_unit_fires(m, x, namespace='fr0')
                    mdl.remove(mdl[i+1])
                    nml -= 1
                elif mode == 'lut_sbias':
                    x = m.forward_lut(x, tune=True)
                    add_unit_fires(m, x)
                    mdl.remove(mdl[i+1])
                    nml -= 1
                else:
                    raise Exception("Something went wrong...")
            else:
                x = m(x)

        return x


    def setup_neuron(self):
        mdl = list(self.NC.network.modules())
        nml = len(mdl)
        neuron_lists = []

        first_layer = True
        rseq = random_seq_cuda(100) #Number of montes. [INTERNAL_PARAMETER]
        LUT0, idx0, idx1 = load_LUT_cuda("path_circuitsim/" + self.NC.env.envname_circuitsim + "/LUT_results/" + self.NC.env.LUTSampleDir)

        self.idx1 = idx1 #used in voltage -> index mapping

        for i, m in enumerate(mdl):
            
            is_Valid_BN = (i < nml-1) and (mdl[i+1].__module__ == "torch.nn.modules.batchnorm") #batchnorm at next layer?
            has_weight  = hasattr(m, 'weight') #has weight?

            if is_Valid_BN and has_weight:
                m.has_neuron = False
                if first_layer:
                    #First layer gets 'analog' input. Cannot work as a BNN CBA.
                    first_layer = False
                else:
                    m.has_neuron = True

                    #setup neuron module
                    m.calc_and_set_alpha(mdl[i+1]) #alpha for BNN is calculated and set.
                    m.rand_seq = rseq
                    m.LUT = LUT0.to(m.weight.device) #designates correct device. what makes the change?
                    m.LUTidx0 = idx0
                    m.LUTidx1 = idx1

                    neuron_lists.append(m)

        #tie breaker

        return neuron_lists

    def run_sbias_tuning(self, neuron_lists):

        #step 3-1: setup CBA bias from alpha.
        self.set_distributed_bias(neuron_lists)

        #step 3-2: Test with LUT
        dp("NNVD: Testing with LUT...")
        acc, loss = self.test(mode='lut_sbias')
        dp("Ok.\n")
        dp("NNVD: Test result with LUT: acc: {}%, loss: {}\n".format(acc*100., loss))

        #step 3-3: Global calibration with Vref and bias.
        dp("NNVD: Global bias calibration...")
        voltage_lists = np.arange(0.76, 0.96, 0.04) #hard coded array.
        self.perform_vref_bias_shift(neuron_lists, voltage_lists, n_batch_tuning=self.NC.env.n_batch_tuning) #need to read value from global file.
        acc, loss = self.test(mode='lut_sbias')
        dp("Ok.\n")
        dp("NNVD: Intermediate calibration result: acc: {}%, loss: {}\n".format(acc*100., loss))

        #step 3-4: Local calibration with Vref and bias.
        dp("NNVD: Local bias calibration...")
        self.adjust_bias_by_binarysearch(neuron_lists, n_batch_tuning=self.NC.env.n_batch_tuning)
        acc, loss = self.test(mode='lut_sbias')
        dp("Ok.\n")
        dp("NNVD: Final calibration result: acc: {}%, loss: {}\n".format(acc*100., loss))


    def perform_vref_bias_shift(self, neuron_lists, voltage_lists, n_batch_tuning=1):

        dp("dbg0: n_batch_tuning = {}\n".format(n_batch_tuning))
        if n_batch_tuning > 0:
            batch_in = self.NC.dataset[0][0:n_batch_tuning] #changed dataset to train set.
        else:
            batch_in = self.NC.dataset[0]

        #bwsnn test mode for reference firing rate calculation!
        acc, loss = self.test(mode='bwsnn', usr_batch=batch_in)
        dp("\nPVBS: BWSNN mode test result: acc: {}%, loss: {}\n".format(acc*100, loss))

        best_vref = 0.76
        best_shift = -1
        best_acc = -1.0

        #coarse tune
        #for sh in range(-1, -2, -1): #original: 5, 4, 3, ... -3.
        #    for vref in voltage_lists:
        #        dp(".")
        #        for m in neuron_lists:
        #            nCBA = m.sbias.data.size(-1)
        #            m.alpha = m.alpha_original + nCBA*sh
        #            m.vref_base = vref #both alpha and vref_base affects test phase.

        #        self.set_distributed_bias(neuron_lists)

        #        acc, loss = self.test(mode='lut_sbias', usr_batch=batch_in)

        #        print(acc, loss)

        #        if acc > best_acc:
        #            dp("\nPVBS: Best achived during coarse tuning: Vref/Shift/Acc: {}, {}, {}%\n".format(vref, sh, acc*100))
        #            best_vref = vref
        #            best_shift = sh
        #            best_acc = acc

        #for vref in [best_vref-0.001, best_vref+0.001]: #fine voltage tuning.
        #    for m in neuron_lists:
        #        nCBA = m.sbias.data.size(-1)
        #        m.alpha = m.alpha_original + nCBA*best_shift
        #        m.vref_base = vref

        #    self.set_distributed_bias(neuron_lists)

        #    acc, loss = self.test(mode='lut_sbias', usr_batch=batch_in)

        #    if acc > best_acc:
        #        dp("\nPVBS: Best achived during fine tuning: Vref/Shift/Acc: {}, {}, {}%\n".format(vref, best_shift, acc*100))
        #        best_vref = vref
        #        best_acc = acc

        #finalize
        dp("\nPVBS: Final Best: Vref/Shift/Acc: {}, {}, {}%\n".format(best_vref, best_shift, best_acc*100))
        for m in neuron_lists:
            nCBA = m.sbias.data.size(-1)
            m.alpha = m.alpha_original + nCBA*best_shift
            m.vref_base = best_vref #both alpha and vref_base affects test phase.

        self.set_distributed_bias(neuron_lists)


    def test_vref_shift(self, neuron_lists, voltage_lists):

        #step 4: Test accuracy with shifted vref.
        dp("TVSH: Vref-noise Test...")

        #gather centric voltage value
        V0 = neuron_lists[0].vref_base

        #voltage_lists = np.arange(V0-0.1, V0+0.1, 0.004).tolist()
         
        for vref in voltage_lists:
            for i, m in enumerate(neuron_lists):
                m.vref_base = vref

            acc, loss = self.test(mode='lut_sbias')
            dp("\nVref = {}: acc = {}%: loss = {}".format(vref, acc*100, loss))

        dp("\nTVSH: Test finished.")


    def set_distributed_bias(self, neuron_lists):

        for m in neuron_lists:
            m.set_distributed_bias()


    def adjust_bias_by_binarysearch(self, neuron_lists, n_batch_tuning=1, iter_max=100):
        
        if n_batch_tuning > 0:
            batch_in = self.NC.dataset[0][0:n_batch_tuning]  #train data set.
            #img_count = self.NC.env.n_batch_test * n_batch_tuning
        else:
            batch_in = self.NC.dataset[0]
            #img_count = self.NC.env.n_batch_test


        for i, m in enumerate(neuron_lists):
            #skip inference & firing rate calculation
            dp(".")
            da_0 = self.comp_activation_diff(m, batch_in)
            nCBA = m.sbias.data.size(-1)

            max_bias_delta = (m.sWL-m.sW)*nCBA
            exponent = math.ceil(math.log(max_bias_delta, 2))
            init_step = max( 2**(exponent-1), 2 )
            step = da_0.clone().fill_(init_step) #designate different step through each column?

            #sbias backup
            sbias_original = m.sbias.data.clone()

            for iter in range(iter_max):
                m.alpha += -da_0.sign() * step #update bias value: B <-- B - sign(ΔA)*step
                self.set_distributed_bias(neuron_lists) #re-program RRAM.
                da = self.comp_activation_diff(m, batch_in) #compute average activation and calculate Δa.

                da_0 = da.clone()
                step = step // 2 #equivalent?: reduce step.
                init_step = init_step // 2

                if init_step == 0:
                    break

            #move once more!
            m.alpha += -da_0.sign() #step == 1
            self.set_distributed_bias(neuron_lists)
            da = self.comp_activation_diff(m, batch_in) #calculate difference for last step.

            worse_idx = da_0.abs() < da.abs() #take best between one of last 2 trials.
            m.alpha[worse_idx] += da_0.sign()[worse_idx]
            self.set_distributed_bias(neuron_lists)

    def comp_activation_diff(self, module, batch_in):

        fr0 = load_firing_rate(module, namespace='fr0') #base fr should be evaluated before tuning!

        #inference_mode and get new firing rate
        self.test(mode='lut_sbias', usr_batch=batch_in)
        fr  = load_firing_rate(module)

        return fr - fr0


