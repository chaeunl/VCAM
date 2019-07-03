import os, sys

args = sys.argv[1:]

#arg_0: input environment name
#arg_1: data zip file

envname = args[0]
datname = args[1]

envdir = "path_circuitsim/" + envname
os.mkdir(envdir)

os.system("unzip " + datname + " -d " + envdir)
print("Imported data. Give '{}' as input argument to NNContainerSetup.".format(envname))

