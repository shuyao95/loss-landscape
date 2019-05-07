import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False, auxiliary=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel, auxiliary)
    return net
