import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple

OPS = {
	'none': lambda C, stride, affine: Zero(stride),
	'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
	                                                       count_include_pad=False),
	'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
	'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C,
	                                                                                          affine=affine),
	'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
	'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
	'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
	'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
	'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
	'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
		nn.ReLU(inplace=False),
		nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
		nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
		nn.BatchNorm2d(C, affine=affine)
	),
}


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


class DilConv(nn.Module):

	def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
		super(DilConv, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
			          dilation=dilation, groups=C_in, bias=False),
			nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(C_out, affine=affine),
		)

	def forward(self, x):
		return self.op(x)


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


class Identity(nn.Module):

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Zero(nn.Module):

	def __init__(self, stride):
		super(Zero, self).__init__()
		self.stride = stride

	def forward(self, x):
		if self.stride == 1:
			return x.mul(0.)
		return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

	def __init__(self, C_in, C_out, affine=True):
		super(FactorizedReduce, self).__init__()
		assert C_out % 2 == 0
		self.relu = nn.ReLU(inplace=False)
		self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
		self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
		self.bn = nn.BatchNorm2d(C_out, affine=affine)

	def forward(self, x):
		x = self.relu(x)
		out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
		out = self.bn(out)
		return out


def drop_path(x, drop_prob):
	if drop_prob > 0.:
		keep_prob = 1. - drop_prob
		mask = Variable(torch.cuda.FloatTensor(
			x.size(0), 1, 1, 1).bernoulli_(keep_prob))
		x.div_(keep_prob)
		x.mul_(mask)
	return x


class Cell(nn.Module):

	def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
		super(Cell, self).__init__()
		# print(C_prev_prev, C_prev, C)

		if reduction_prev:
			self.preprocess0 = FactorizedReduce(C_prev_prev, C)
		else:
			self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
		self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

		if reduction:
			op_names, indices = zip(*genotype.reduce)
			concat = genotype.reduce_concat
		else:
			op_names, indices = zip(*genotype.normal)
			concat = genotype.normal_concat
		self._compile(C, op_names, indices, concat, reduction)

	def _compile(self, C, op_names, indices, concat, reduction):
		assert len(op_names) == len(indices)
		self._steps = len(op_names) // 2
		self._concat = concat
		self.multiplier = len(concat)

		self._ops = nn.ModuleList()
		for name, index in zip(op_names, indices):
			stride = 2 if reduction and index < 2 else 1
			op = OPS[name](C, stride, True)
			self._ops += [op]
		self._indices = indices

	def forward(self, s0, s1, drop_prob):
		s0 = self.preprocess0(s0)
		s1 = self.preprocess1(s1)

		states = [s0, s1]
		for i in range(self._steps):
			h1 = states[self._indices[2 * i]]
			h2 = states[self._indices[2 * i + 1]]
			op1 = self._ops[2 * i]
			op2 = self._ops[2 * i + 1]
			h1 = op1(h1)
			h2 = op2(h2)
			if self.training and drop_prob > 0.:
				if not isinstance(op1, Identity):
					h1 = drop_path(h1, drop_prob)
				if not isinstance(op2, Identity):
					h2 = drop_path(h2, drop_prob)
			s = h1 + h2
			states += [s]
		return torch.cat([states[i] for i in self._concat], dim=1)


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


class NetworkCIFAR(nn.Module):

	def __init__(self, C, num_classes, layers, auxiliary, genotype):
		super(NetworkCIFAR, self).__init__()
		self._layers = layers
		self._auxiliary = auxiliary

		stem_multiplier = 3
		C_curr = stem_multiplier * C
		self.stem = nn.Sequential(
			nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
			nn.BatchNorm2d(C_curr)
		)

		C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
		self.cells = nn.ModuleList()
		reduction_prev = False
		for i in range(layers):
			if i in [layers // 3, 2 * layers // 3]:
				C_curr *= 2
				reduction = True
			else:
				reduction = False
			cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
			reduction_prev = reduction
			self.cells += [cell]
			C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
			if i == 2 * layers // 3:
				C_to_auxiliary = C_prev

		if auxiliary:
			self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
		self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(C_prev, num_classes)

	def forward(self, input):
		logits_aux = None
		s0 = s1 = self.stem(input)
		for i, cell in enumerate(self.cells):
			s0, s1 = s1, cell(s0, s1, drop_prob=0)
			if i == 2 * self._layers // 3:
				if self._auxiliary and self.training:
					logits_aux = self.auxiliary_head(s1)
		out = self.global_pooling(s1)
		logits = self.classifier(out.view(out.size(0), -1))
		return logits, logits_aux

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
DARTS_V1 = Genotype(
	normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
	        ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0),
	        ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
	reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
	normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
	        ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
	        ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

DARTS_OPS1 = Genotype(
	normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
	        ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1),
	        ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])
DARTS_OPS2 = Genotype(
	normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1),
	        ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1),
	        ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])
DARTS_OPS3 = Genotype(
	normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1),
	        ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_5x5', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1),
	        ('dil_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])
DARTS_OPS4 = Genotype(
	normal=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1),
	        ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1),
	        ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])
DARTS_OPS5 = Genotype(
	normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1),
	        ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1),
	        ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])

DARTS_CONN1 = Genotype(
	normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
	        ('sep_conv_3x3', 3), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_3x3', 0)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
	        ('max_pool_3x3', 2), ('skip_connect', 1), ('skip_connect', 1), ('max_pool_3x3', 0)],
	reduce_concat=[2, 3, 4, 5])
DARTS_CONN2 = Genotype(
	normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
	        ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_3x3', 2)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1),
	        ('max_pool_3x3', 3), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 1)],
	reduce_concat=[2, 3, 4, 5])
DARTS_CONN3 = Genotype(
	normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
	        ('sep_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4), ('dil_conv_3x3', 4)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2),
	        ('max_pool_3x3', 2), ('skip_connect', 3), ('skip_connect', 4), ('max_pool_3x3', 0)],
	reduce_concat=[2, 3, 4, 5])
DARTS_CONN4 = Genotype(
	normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
	        ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_3x3', 4)],
	normal_concat=[2, 3, 4, 5],
	reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
	        ('max_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 3), ('max_pool_3x3', 2)],
	reduce_concat=[2, 3, 4, 5])


def darts_v2(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_V2)

def darts_v1(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_V1)

def darts_ops1(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_OPS1)

def darts_ops2(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_OPS2)

def darts_ops3(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_OPS3)

def darts_ops4(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_OPS4)

def darts_ops5(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_OPS5)

def darts_conn1(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_CONN1)

def darts_conn2(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_CONN2)

def darts_conn3(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_CONN3)

def darts_conn4(auxiliary):
	return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=auxiliary, genotype=DARTS_CONN4)