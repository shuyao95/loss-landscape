import torch
from torch import nn
from torch.autograd import Variable

def drop_path(x, drop_prob, dims=(0,)):
    var_size = [1 for _ in range(x.dim())]
    for i in dims:
        var_size[i] = x.size(i)
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(*var_size).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

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

		# processing layers
		self.ops = nn.ModuleList()
		for _ in range(nodes * prev_connects):
			if reduction:
				stride = 2
			else:
				stride = 1
			op = sepconv3x3(channels, stride)
			self.ops.append(op)

	def forward(self, inputs, drop_prob=0):
		# only inputs needed by this layers are passed, last in is in the first place
		states = []
		for i, input in enumerate(inputs):
			states.append(self.pre_ops[i](input))
		outs = []
		indices = 0
		for _ in range(self.nodes):
			out = sum([drop_path(op(states[i]), drop_prob) for i, op in
			           enumerate(self.ops[indices:indices + self.prev_connects])])
			outs.append(out)
			indices += self.prev_connects
		return sum(outs)


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
			prev_channels.insert(0, C_curr)
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
			out = cell(inputs[:self._prev_connects], drop_prob=0)
			inputs.insert(0, out)
			if i == 2 * self._layers // 3:
				if self._auxiliary and self.training:
					logits_aux = self.auxiliary_head(out)
		out = self.global_pooling(out)
		logits = self.classifier(out.view(out.size(0), -1))
		return logits, logits_aux

def dd_add():
	return NetworkCIFAR(C=36, nodes=4, num_classes=10, layers=20, auxiliary=False, \
	                                                                      prev_connects=2)