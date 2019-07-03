from nn_vcam_driver import *
#from nn_train_driver import *
#from nn_base_MNIST import *
from nn_base_CIFAR10 import *
from nn_tools import cleanup

#cleanup()

setup1 = NNContainerSetup()
netcontainer = NNContainer(setup1)

#inst1 = NNTrainDriver(netcontainer)
#inst1.run()
#del inst1

inst2 = NNVcamDriver(netcontainer)
inst2.run()
del inst2
