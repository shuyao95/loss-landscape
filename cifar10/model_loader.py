import os
import torch
import torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet

import cifar10.models.darts as darts
import cifar10.models.dd_add as dd_add
import cifar10.models.dd_concat as dd_concat
import cifar10.models.dd_nogroup as dd_nogroup
import cifar10.models.dd_seq as dd_seq

# map between model name and function
models = {
    'vgg9': vgg.VGG9,
    'densenet121': densenet.DenseNet121,
    'resnet18': resnet.ResNet18,
    'resnet18_noshort': resnet.ResNet18_noshort,
    'resnet34': resnet.ResNet34,
    'resnet34_noshort': resnet.ResNet34_noshort,
    'resnet50': resnet.ResNet50,
    'resnet50_noshort': resnet.ResNet50_noshort,
    'resnet101': resnet.ResNet101,
    'resnet101_noshort': resnet.ResNet101_noshort,
    'resnet152': resnet.ResNet152,
    'resnet152_noshort': resnet.ResNet152_noshort,
    'resnet20': resnet.ResNet20,
    'resnet20_noshort': resnet.ResNet20_noshort,
    'resnet32_noshort': resnet.ResNet32_noshort,
    'resnet44_noshort': resnet.ResNet44_noshort,
    'resnet50_16_noshort': resnet.ResNet50_16_noshort,
    'resnet56': resnet.ResNet56,
    'resnet56_noshort': resnet.ResNet56_noshort,
    'resnet110': resnet.ResNet110,
    'resnet110_noshort': resnet.ResNet110_noshort,
    'wrn56_2': resnet.WRN56_2,
    'wrn56_2_noshort': resnet.WRN56_2_noshort,
    'wrn56_4': resnet.WRN56_4,
    'wrn56_4_noshort': resnet.WRN56_4_noshort,
    'wrn56_8': resnet.WRN56_8,
    'wrn56_8_noshort': resnet.WRN56_8_noshort,
    'wrn110_2_noshort': resnet.WRN110_2_noshort,
    'wrn110_4_noshort': resnet.WRN110_4_noshort,
    'darts_v1': darts.darts_v1,
    'darts_v2': darts.darts_v2,
    'darts_nasnet': darts.darts_nasnet,
    'darts_nasnet_conn1': darts.darts_nasnet_conn1,
    'darts_nasnet_conn2': darts.darts_nasnet_conn2,
    'darts_nasnet_conn3': darts.darts_nasnet_conn3,
    'darts_nasnet_conn4': darts.darts_nasnet_conn4,
    'darts_amoebanet': darts.darts_amoebanet,
    'darts_amoebanet_conn1': darts.darts_amoebanet_conn1,
    'darts_amoebanet_conn2': darts.darts_amoebanet_conn2,
    'darts_amoebanet_conn3': darts.darts_amoebanet_conn3,
    'darts_amoebanet_conn4': darts.darts_amoebanet_conn4,
    'darts_enas': darts.darts_enas,
    'darts_enas_conn1': darts.darts_enas_conn1,
    'darts_enas_conn2': darts.darts_enas_conn2,
    'darts_enas_conn3': darts.darts_enas_conn3,
    'darts_enas_conn4': darts.darts_enas_conn4,
    'darts_snas': darts.darts_snas,
    'darts_snas_conn1': darts.darts_snas_conn1,
    'darts_snas_conn2': darts.darts_snas_conn2,
    'darts_snas_conn3': darts.darts_snas_conn3,
    'darts_snas_conn4': darts.darts_snas_conn4,
    'darts_ops1': darts.darts_ops1,
    'darts_ops2': darts.darts_ops2,
    'darts_ops3': darts.darts_ops3,
    'darts_ops4': darts.darts_ops4,
    'darts_ops5': darts.darts_ops5,
    'darts_conn1': darts.darts_conn1,
    'darts_conn2': darts.darts_conn2,
    'darts_conn3': darts.darts_conn3,
    'darts_conn4': darts.darts_conn4,
    'dd_nogroup': dd_nogroup.dd_nogroup,
    'dd_seq': dd_seq.dd_seq,
    'dd_add': dd_add.dd_add,
    'dd_node1': dd_concat.dd_node1,
    'dd_node1_c46': dd_concat.dd_node1_c46,
    'dd_node1_c36': dd_concat.dd_node1_c36,
    'dd_node2': dd_concat.dd_node2,
    'dd_node3': dd_concat.dd_node3,
    'dd_node4': dd_concat.dd_node4,
    'dd_node5': dd_concat.dd_node5,
    'dd_prev1': dd_concat.dd_prev1,
    'dd_prev2': dd_concat.dd_prev2,
    'dd_prev3': dd_concat.dd_prev3,
    'dd_prev4': dd_concat.dd_prev4,
}


def load(model_name, model_file=None, data_parallel=False, auxiliary=False):
    if 'darts' in model_name:
        net = models[model_name](auxiliary)
    else:
        net = models[model_name]()
    if data_parallel:  # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel:  # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
