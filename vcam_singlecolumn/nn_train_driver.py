
from torch.autograd import Variable

from nn_tools import *
from nn_base import *

class NNTrainDriver():
    def __init__(self, networkcontainer, *args, **kwargs):
        super(NNTrainDriver, self).__init__(*args, **kwargs)
        self.NC = networkcontainer #as NNContainer
        self.Model_savepath = "path_model/" + self.NC.env.envname_model + "/"

    def run(self):
        self.train()

    def train(self):

        #dataset structure: dataset[train/test][batch_idx][image/label]

        best_acc = 0.0
        best_loss = 99.0

        for e in range(self.NC.env.epochs):
            dp(".")
            self.train_once()
            acc, loss = self.test()
            self.NC.scheduler.step()

            dp("NNTD: Test @ epoch {}: acc: {:.2f}%, loss: {:.4f}\n".format(e, acc*100., loss), logfilepath=self.Model_savepath + "train_epoch.txt")
            if acc > best_acc:
                dp("NNTD: Best @ epoch {}: acc: {:.2f}%, loss: {:.4f}\n".format(e, acc*100., loss))
                best_acc = acc
                best_loss = loss
                dp("NNTD: Saving best model...")
                save_network(self.NC.network, self.NC.optimizer, self.Model_savepath + "model.pth")

                self.restore_weight_binary_layers()
                save_network(self.NC.network, self.NC.optimizer, self.Model_savepath + "model_f.pth")

                self.test() #purpose of binarizing weights
                dp("Ok.\n")

    def train_once(self):
        self.NC.network.train() #set network mode to train.

        #dataset structure: dataset[train/test][batch_idx][image/label]
        train_set = self.NC.dataset[0]

        for bidx, (inputs, labels) in enumerate(train_set):
           
            #forward
            O = self.NC.network(Variable(inputs))

            loss = self.NC.criterion(O, Variable(labels))
            #_, argmax = torch.max(O[0], 0)
            #plot(inputs[0], argmax, labels[0])

            #reset the parameter gradients
            self.NC.optimizer.zero_grad()
            #backward
            loss.backward()

            #rollback to real weight for binary layers
            self.restore_weight_binary_layers()

            #weight update
            self.NC.optimizer.step()

            #clamp for binary layers
            self.clamp_weight_binary_layers()
            self.clamp_weight_bn_layers()

    #defined at BinLinear and its inherits. 
    def restore_weight_binary_layers(self):
        for p in list(self.NC.network.parameters()):
            if hasattr(p, 'original_data'):
                p.data.copy_(p.original_data)

    def clamp_weight_binary_layers(self):
        for p in list(self.NC.network.parameters()):
            if hasattr(p, 'original_data'):
                p.original_data.copy_(p.data.clamp_(-1,1))

    def clamp_weight_bn_layers(self):
        for m in list(self.NC.network.modules()):
            if m.__module__ == 'torch.nn.modules.batchnorm':
                m.weight.data.clamp_(min=0.5) #set min_gamma = 0.5 for bn layers.

    def test(self, mode='default', batch_list=None, use_all_batch_per_iter=True, iter_count=0):
        if batch_list is None:
            batch_list = self.NC.dataset[1] #test dataset

        self.NC.network.eval() #set network to test mode.

        loss = 0
        correct = 0
        count = 0

        if use_all_batch_per_iter:
            batch_in = batch_list
        else:
            blen = len(batch_list)
            bidx = int(iter_count % blen)
            batch_in = [batch_list[bidx]]

        for inputs, labels in batch_in:

            inputs, labels = Variable(inputs), Variable(labels)

            O = self.NC.network(inputs)
            #other types of forward are not implemented.

            #O = O * 1.0 #output scaledown
            _, pred = torch.max(O, 1, keepdim=True)
            loss += self.NC.criterion(O, labels).data
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            count += len(labels)

        loss /= len(batch_in)
        acc = correct.item()/count

        return acc, loss
