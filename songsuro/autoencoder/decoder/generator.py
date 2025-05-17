# -----------------------------------------------------------
# This code is adapted from HiFi-GAN: https://github.com/jik876/hifi-gan
# Original repository: https://github.com/jik876/hifi-gan
# License: MIT License
# -----------------------------------------------------------


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from songsuro.utils.util import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
	def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
		super().__init__()
		self.convs1 = nn.ModuleList(
			[
				weight_norm(
					Conv1d(
						channels,
						channels,
						kernel_size,
						1,
						dilation=d,
						padding=get_padding(kernel_size, d),
					)
				)
				for d in dilation[:3]
			]
		)
		self.convs1.apply(init_weights)

		self.convs2 = nn.ModuleList(
			[
				weight_norm(
					Conv1d(
						channels,
						channels,
						kernel_size,
						1,
						dilation=1,
						padding=get_padding(kernel_size, 1),
					)
				)
				for _ in range(3)
			]
		)
		self.convs2.apply(init_weights)

	def forward(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c1(xt)
			xt = F.leaky_relu(xt, LRELU_SLOPE)
			xt = c2(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for layer in self.convs1:
			remove_weight_norm(layer)
		for layer in self.convs2:
			remove_weight_norm(layer)

	def add_weight_norm(self):
		for layer in self.convs1:
			weight_norm(layer)
		for layer in self.convs2:
			weight_norm(layer)


class ResBlock2(torch.nn.Module):
	def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
		super().__init__()
		self.convs = nn.ModuleList(
			[
				weight_norm(
					Conv1d(
						channels,
						channels,
						kernel_size,
						1,
						dilation=d,
						padding=get_padding(kernel_size, d),
					)
				)
				for d in dilation[:2]
			]
		)
		self.convs.apply(init_weights)

	def forward(self, x):
		for c in self.convs:
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for layer in self.convs:
			remove_weight_norm(layer)


class Generator(torch.nn.Module):
	def __init__(
		self,
		generator_input_channels=80,
		resblock="1",
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[8, 8, 2, 2],
		upsample_initial_channel=512,
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
	):
		super().__init__()
		self.num_kernels = len(resblock_kernel_sizes)
		self.num_upsamples = len(upsample_rates)
		self.upsample_rates = upsample_rates
		self.conv_pre = weight_norm(
			Conv1d(generator_input_channels, upsample_initial_channel, 7, 1, padding=3)
		)
		resblock = ResBlock1 if resblock == "1" else ResBlock2

		self.ups = nn.ModuleList()
		for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
			self.ups.append(
				weight_norm(
					ConvTranspose1d(
						upsample_initial_channel // (2**i),
						upsample_initial_channel // (2 ** (i + 1)),
						k,
						u,
						padding=(k - u) // 2,
					)
				)
			)

		self.resblocks = nn.ModuleList()
		for i in range(len(self.ups)):
			ch = upsample_initial_channel // (2 ** (i + 1))
			for j, (k, d) in enumerate(
				zip(resblock_kernel_sizes, resblock_dilation_sizes)
			):
				self.resblocks.append(resblock(ch, k, d))

		self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
		self.ups.apply(init_weights)
		self.conv_post.apply(init_weights)

	def forward(self, x):
		x = self.conv_pre(x)
		for i in range(self.num_upsamples):
			x = F.leaky_relu(x, LRELU_SLOPE)
			x = self.ups[i](x)
			xs = None
			for j in range(self.num_kernels):
				if xs is None:
					xs = self.resblocks[i * self.num_kernels + j](x)
				else:
					xs += self.resblocks[i * self.num_kernels + j](x)
			x = xs / self.num_kernels
		x = F.leaky_relu(x)
		x = self.conv_post(x)
		x = torch.tanh(x)

		return x

	def remove_weight_norm(self):
		print("Removing weight norm...")
		for layer in self.ups:
			remove_weight_norm(layer)
		for layer in self.resblocks:
			layer.remove_weight_norm()
		remove_weight_norm(self.conv_pre)
		remove_weight_norm(self.conv_post)

	def add_weight_norm(self):
		print("Adding weight norm...")
		for layer in self.ups:
			weight_norm(layer)
		for layer in self.resblocks:
			layer.add_weight_norm()
		weight_norm(self.conv_pre)
		weight_norm(self.conv_post)
