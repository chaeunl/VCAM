import torch, os, sys, shutil, time
import numpy as np

from torchvision import transforms as tfm
from torchvision import datasets

import matplotlib.pyplot as plt

# # # # # # # # DATA_LOADERS # # # # # # # #

MASTER_DATA_LOCATION = "./data_path"

def load_mnist_dataset(batch_size_train, batch_size_test, isCudaActive=True):
    kwargs = {'num_workers':1, 'pin_memory':True} if isCudaActive else {}

    mean_val = (0.1307,)
    std_val  = (0.3081,)
    dp("TLLM: Preparing loader...")
    tf_train = tfm.Compose( [tfm.ToTensor(), tfm.Normalize(mean=mean_val, std=std_val)] )
    tf_test  = tfm.Compose( [tfm.ToTensor(), tfm.Normalize(mean=mean_val, std=std_val)] )

    set_train = datasets.MNIST(root=MASTER_DATA_LOCATION, train=True, download=True, transform=tf_train)
    ldr_train = torch.utils.data.DataLoader(set_train, batch_size=batch_size_train, shuffle=True, **kwargs)

    set_test  = datasets.MNIST(root=MASTER_DATA_LOCATION, train=False, download=True, transform=tf_test )
    ldr_test  = torch.utils.data.DataLoader(set_test,  batch_size=batch_size_test,  shuffle=False, **kwargs)
    dp("Ok.\n")
    classes = {'0','1','2','3','4','5','6','7','8','9'}

    return __get_list_and_cuda_alloc__(ldr_train, ldr_test, classes)

def load_cifar10_dataset(batch_size_train, batch_size_test, isAugmented=False, isCudaActive=True):
    kwargs = {'num_workers':1, 'pin_memory':True} if isCudaActive else {}

    mean_val = (0.4914,0.4822,0.4465)
    std_val  = (0.2023,0.1994,0.2010)

    if isAugmented:
        tf_train = tfm.Compose( [tfm.RandomHorizontalFlip(), tfm.RandomCrop(32,3), tfm.ToTensor(), tfm.Normalize(mean=mean_val, std=std_val)] )
    else:
        tf_train = tfm.Compose( [tfm.ToTensor(), tfm.Normalize(mean=mean_val, std=std_val)] )

    tf_test  = tfm.Compose( [tfm.ToTensor(), tfm.Normalize(mean=mean_val, std=std_val)] )

    set_train = datasets.CIFAR10(root=MASTER_DATA_LOCATION, train=True, download=True, transform=tf_train)
    ldr_train = torch.utils.data.DataLoader(set_train, batch_size=batch_size_train, shuffle=True, **kwargs)

    set_test  = datasets.CIFAR10(root=MASTER_DATA_LOCATION, train=False, download=True, transform=tf_test )
    ldr_test  = torch.utils.data.DataLoader(set_test,  batch_size=batch_size_test,  shuffle=False, **kwargs)

    classes = {'plane','car','bird','cat','deer','dog','frog','horse','ship','truck'}

    return __get_list_and_cuda_alloc__(ldr_train, ldr_test, classes)

def __get_list_and_cuda_alloc__(ldr_train, ldr_test, classes):
    dp("GLCA: Loading data and allocating...")
    blist_train = list(ldr_train)
    blist_test  = list(ldr_test )

    for i, (img,label) in enumerate(blist_train):
        blist_train[i][0] = img.squeeze().cuda(0)
        blist_train[i][1] = label.cuda(0)

    for i, (img,label) in enumerate(blist_test):
        blist_test[i][0] = img.squeeze().cuda(0)
        blist_test[i][1] = label.cuda(0)

    dp("Complete!\n")
    return blist_train, blist_test

# # # # # # # # BUILDERS # # # # # # # #

def build_network(net_type):
    try:
        loader = getattr(__import__("nn_collections"), net_type)
    except AttributeError:
        raise ValueError("Invalid network type string!")

    return loader()

def inflate_network(network, optimizer, filepath):
    dp("TLLN: Loading model...")
    try:
        state = torch.load(filepath)
        if network is not None:
            network.load_state_dict(state['net_state'])
        if optimizer is not None:
            optimizer.load_state_dict(state['optim_state'])
        dp("Loaded model from {}\n".format(filepath))
    except:
        dp("No such file exists. Load cancelled.\n")
    #state load complete. net and optim are reference.

def save_network(network, optimizer, filepath):
    state = {}
    if network is not None:
        state['net_state'] = network.state_dict()
    if optimizer is not None:
        state['optim_state'] = optimizer.state_dict()

    torch.save(state, filepath)
    dp("TLSN: Saved model @ {}\n".format(filepath))

# # # # # # # # CUDA TOOLS # # # # # # # #

def random_seq_cuda(limit_below, size=100_000_000):
    return torch.cuda.FloatTensor(size).random_(0, limit_below)

#def load_LUT_cuda(LUT_folder_path, N_weights=33, fill_default=1.091):
#    dp("TLLL: Loading LUT...")
#    fileList = os.listdir(LUT_folder_path)
#    N_monte = len(fileList)
#    LUT = torch.cuda.FloatTensor(N_monte, N_weights, N_weights).fill_(fill_default)
#    for i, f in enumerate(fileList):
#        t1 = np.loadtxt(LUT_folder_path + "/" + f)
#        #desired format: w_p, w_n, vout
#        idx = t1[:,0:2].T.astype(int).tolist()
#        val = torch.from_numpy(t1[:,2].T).float().cuda()
#        LUT[i][idx] = val
#
#    dp("complete!: LUT[0][1][2] = {}\n".format(LUT[0][1][2]))
#    return LUT

def load_LUT_cuda(LUT_folder_path, fill_default=0.0):
    dp("TLLL: Loading LUT...")
    fileList = os.listdir(LUT_folder_path)
    N_monte = len(fileList)

    sampleLUT = np.loadtxt(LUT_folder_path + "/" + fileList[0])
    key0list = np.unique(sampleLUT[:,0])
    key1list = np.unique(sampleLUT[:,1])
    N_key0 = len(key0list)
    N_key1 = len(key1list)

    idx0 = dict( zip(key0list, range(N_key0)) )
    idx1 = dict( zip(key1list, range(N_key1)) )

    LUT = torch.cuda.FloatTensor(N_monte, N_key0, N_key1).fill_(fill_default)
    for i, f in enumerate(fileList):
        t1 = np.loadtxt(LUT_folder_path + "/" + f)
        
        p0 = list(map( idx0.get, t1[:,0] ))
        p1 = list(map( idx1.get, t1[:,1] ))
        pcmbn = [p0,p1]
        val = torch.from_numpy(t1[:,2].T).float().cuda()
        LUT[i][pcmbn] = val

    dp("complete!: Use 2 key-mappings to access LUT.\n")
    return LUT, idx0, idx1


# # # # # # # # NEW SPIKE TOOLS # # # # # # # #

def load_firing_rate(module, namespace='fr'):
    return getattr(module, namespace, None).to(module.weight.device)

def reset_firing_rate(network, namespace='fr'):
    mdl = list(network.modules())

    for m in mdl:
        if hasattr(m, 'has_neuron') and m.has_neuron:
            setattr(m, namespace, None)
            m.fr_n_samples = None


def add_unit_fires(module, output, namespace='fr'):
    sz_batch = output.size(0)

    if hasattr(module, namespace):
        val = getattr(module, namespace, None)
        if val is None:
            setattr(module, namespace, output.data.sign().clamp_(min=0.0).sum(0))
            module.fr_n_samples = sz_batch
        else:
            setattr(module, namespace, val + output.data.sign().clamp_(min=0.0).sum(0))
            module.fr_n_samples += sz_batch


def set_firing_rate(network, namespace='fr'): #namespace should be 'fr' or 'fr0'!
    mdl = list(network.modules())
    for m in mdl:
        if hasattr(m, 'has_neuron') and m.has_neuron:
            val = getattr(m, namespace, None)
            setattr(m, namespace, val / m.fr_n_samples)

        
# # # # # # # # DEBUG TOOLS  # # # # # # # #

def dp(string, debugMode=True, logfilepath=None):
    if debugMode:
        print(string, end='')
        sys.stdout.flush()

    if logfilepath is not None:
        with open(logfilepath, "a") as f:
            f.write(string)

def plot(image, prediction, label):
    plt.imshow(image.cpu(), cmap='gray')
    plt.title("Pred:{}, True:{}".format(prediction, label))
    plt.show()


def cleanup(model=True, data=False, LUT=False, Vardata=False, Log=True):

    list1 = []
    if model:
        list1.append("model_path")
    if data:
        list1.append("data_path")
    if LUT:
        list1.append("LUT_path")
    if Vardata:
        list1.append("Vardata_path")
    if Log:
        list1.append("log_path")

    for p in list1:
        shutil.rmtree(p)
        time.sleep(0.1)
        os.mkdir(p)
