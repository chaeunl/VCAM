import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from nn_tools import *

import os

class NNContainerSetup:
    def __init__(self):
        self.dat_type = 'MNIST' #MNIST or CIFAR10
        self.net_type = 'BNN_784_512_512_512_10'

        self.n_batch_train = 128
        self.n_batch_test  = 128
        self.n_batch_tuning = 1

        self.epochs = 500
        self.lr = 0.02
        self.lr_decay = 0.979

        self.bias_noise_std = 0.0
        self.same_seq_through_group = False

        self.envname_circuitsim = "gv0lv0.1"
        self.LUTSampleDir = "sample0.0"
        self.envname_model = "BNN0"
        self.envname_output = "gv0lv0.1_MNISt"
        #self.LUT_filepath = "LUT_path/sample0.0"
        #self.Model_loadpath = "model_path/test.pth"
        #self.Model_savepath = "model_path/test.pth"

    def get_Vt_d2d(self):
        return np.loadtxt("path_circuitsim/" + self.envname_circuitsim + "/Vardata_results/Vt_d2d.txt")

    def get_R_d2d(self):
        return np.loadtxt("path_circuitsim/" + self.envname_circuitsim + "/Vardata_results/R_d2d.txt")
        

class NNContainer:
    def __init__(self, nnContSetup):
        dp("CONT: init...")
        self.env = nnContSetup

        nbtrain = self.env.n_batch_train
        nbtest  = self.env.n_batch_test

        #self.dataset: index 0: trainset, index 1: testset.
        self.dataset = load_mnist_dataset(nbtrain, nbtest) if nnContSetup.dat_type == 'MNIST' else load_cifar10_dataset(nbtrain, nbtest)
        
        self.network = build_network(self.env.net_type).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.env.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.env.lr_decay)

        if os.path.exists("path_model/" + self.env.envname_model):
            inflate_network(self.network, self.optimizer, "path_model/" + self.env.envname_model + "/model.pth")
        else:
            os.mkdir("path_model/" + self.env.envname_model)

        dp("CONT: Ok.\n")


