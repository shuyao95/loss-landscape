import torch
from torch import nn
from utils import drop_path


class SepConv(nn.Module):

	def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
		super(SepConv, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
			          groups=C_in, bias=False),
			nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(C_in, affine=affine),
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in,
			          bias=False),
			nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(C_out, affine=affine),
		)

	def forward(self, x):
		return self.op(x)


# the training speed should be able to boost for grouped sepconv

class GroupSepConv(nn.Module):

	def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, affine=True):
		super(GroupSepConv, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
			          groups=C_in, bias=False),
			nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, groups=groups, bias=False),
			nn.BatchNorm2d(C_in, affine=affine),
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in,
			          bias=False),
			nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, groups=groups, bias=False),
			nn.BatchNorm2d(C_out, affine=affine),
		)

	def forward(self, x):
		return self.op(x)


class ReLUConvBN(nn.Module):

	def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
		super(ReLUConvBN, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
			nn.BatchNorm2d(C_out, affine=affine)
		)

	def forward(self, x):
		return self.op(x)


class FactorizedReduce(nn.Module):

	def __init__(self, C_in, C_out, stride, affine=True):
		super(FactorizedReduce, self).__init__()
		assert C_out % 2 == 0
		self.relu = nn.ReLU(inplace=False)
		self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
		self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
		self.bn = nn.BatchNorm2d(C_out, affine=affine)

	def forward(self, x):
		x = self.relu(x)
		out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
		out = self.bn(out)
		return out


def sepconv3x3(C, stride, affine=True):
	return SepConv(C, C, 3, stride, 1, affine=affine)


def group_sepconv3x3(C_in, C_out, stride, group, affine=True):
	return GroupSepConv(C_in, C_out, 3, stride, 1, groups=group, affine=affine)


class AuxiliaryHeadCIFAR(nn.Module):

	def __init__(self, C, num_classes):
		"""assuming input size 8x8"""
		super(AuxiliaryHeadCIFAR, self).__init__()
		self.features = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
			nn.Conv2d(C, 128, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 768, 2, bias=False),
			nn.BatchNorm2d(768),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Linear(768, num_classes)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x.view(x.size(0), -1))
		return x


class Cell(nn.Module):
	def __init__(self, prev_channels, prev_connects, prev_reduction, nodes, channels, reduction):
		super(Cell, self).__init__()
		self.prev_channels = prev_channels[-prev_connects:]
		self.prev_reduction = prev_reduction[-prev_connects:]
		self.prev_connects = prev_connects
		self.nodes = nodes
		self.channels = channels

		# preprocessing layers
		self.pre_ops = nn.ModuleList()
		for i in range(self.prev_connects):
			if i > 0:
				stride = 2 * len(list(filter(lambda x: x is True, prev_reduction[:i])))
				if stride > 0:
					op = FactorizedReduce(prev_channels[i], channels, stride=stride)
				else:
					op = ReLUConvBN(prev_channels[i], channels, 1, 1, 0)
			else:
				# print(prev_channels[i], channels)
				op = ReLUConvBN(prev_channels[i], channels, 1, 1, 0)
			self.pre_ops.append(op)

		# build based on group sep conv
		if reduction:
			stride = 2
		else:
			stride = 1
		self.op = group_sepconv3x3(channels * self.prev_connects * nodes,
		                           channels * self.prev_connects * nodes,
		                           stride,
		                           group=self.prev_connects * nodes)

	def forward(self, inputs, drop_prob=0):
		# forward based on group sepconv
		states = []
		for i, input in enumerate(inputs):
			states.append(self.pre_ops[i](input))
		inp = []
		for i in range(self.nodes):
			inp += states
		inp = torch.cat(inp, dim=1)
		outs = self.op(inp)

		N, C, H, W = outs.size()
		outs = outs.view(N, self.channels, self.prev_connects, self.nodes, H, W)
		outs = drop_path(outs, drop_prob, dims=(0, 2))
		outs = outs.sum(dim=2).view(N, self.channels * self.nodes, H, W)
		return outs


class NetworkCIFAR(nn.Module):

	def __init__(self, C, nodes, num_classes, layers, auxiliary, prev_connects=2):
		super(NetworkCIFAR, self).__init__()
		self._layers = layers
		self._auxiliary = auxiliary
		self._prev_connects = prev_connects

		stem_multiplier = 3
		C_curr = stem_multiplier * C
		self.stem = nn.Sequential(
			nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
			nn.BatchNorm2d(C_curr)
		)

		prev_channels = []
		prev_reduction = []

		for i in range(prev_connects):
			prev_channels.insert(0, C_curr)
			prev_reduction.append(False)
		C_curr = C

		self.cells = nn.ModuleList()
		for i in range(layers):
			if i in [layers // 3, 2 * layers // 3]:
				C_curr *= 2
				reduction = True
			else:
				reduction = False
			cell = Cell(prev_channels, prev_connects, prev_reduction, nodes, C_curr, reduction)
			self.cells += [cell]
			if i == 2 * layers // 3:
				C_to_auxiliary = prev_channels[0]
			prev_channels.insert(0, C_curr * nodes)
			prev_reduction.insert(0, reduction)

		if auxiliary:
			self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
		self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(prev_channels[0], num_classes)

	def forward(self, input):
		logits_aux = None
		s = self.stem(input)
		inputs = []

		for i in range(self._prev_connects):
			inputs.append(s)

		for i, cell in enumerate(self.cells):
			out = cell(inputs[:self._prev_connects], self.drop_path_prob)
			inputs.insert(0, out)
			if i == 2 * self._layers // 3:
				if self._auxiliary and self.training:
					logits_aux = self.auxiliary_head(out)
		out = self.global_pooling(out)
		logits = self.classifier(out.view(out.size(0), -1))
		return logits, logits_aux

def dd_node1():
	return NetworkCIFAR(C=36, nodes=1, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_node2():
	return NetworkCIFAR(C=36, nodes=2, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_node3():
	return NetworkCIFAR(C=36, nodes=3, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_node4():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_node5():
	return NetworkCIFAR(C=36, nodes=5, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_prev1():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=1)

def dd_prev2():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)

def dd_prev3():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=3)

def dd_prev4():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=4)