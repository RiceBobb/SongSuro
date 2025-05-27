# -----------------------------------------------------------
# This code is adapted from HiFi-GAN: https://github.com/jik876/hifi-gan
# Original repository: https://github.com/jik876/hifi-gan
# License: MIT License
# -----------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d

from songsuro.utils.util import get_padding

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
	def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
		super().__init__()
		self.period = period
		# norm_f = weight_norm if not use_spectral_norm else spectral_norm
		self.convs = nn.ModuleList(
			[
				# norm_f(
				Conv2d(
					1,
					32,
					(kernel_size, 1),
					(stride, 1),
					padding=(get_padding(5, 1), 0),
					# )
				),
				# norm_f(
				Conv2d(
					32,
					128,
					(kernel_size, 1),
					(stride, 1),
					padding=(get_padding(5, 1), 0),
					# )
				),
				# norm_f(
				Conv2d(
					128,
					512,
					(kernel_size, 1),
					(stride, 1),
					padding=(get_padding(5, 1), 0),
					# )
				),
				# norm_f(
				Conv2d(
					512,
					1024,
					(kernel_size, 1),
					(stride, 1),
					padding=(get_padding(5, 1), 0),
					# )
				),
				Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),  # had norm_f
			]
		)
		self.conv_post = Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))  # had norm_f

	def forward(self, x):
		fmap = []
		# 입력 텐서에 채널 차원 추가
		if len(x.shape) == 2:
			x = x.unsqueeze(
				1
			)  # [batch_size, time_steps] -> [batch_size, 1, time_steps]

		# 1d to 2d
		b, c, t = x.shape
		if t % self.period != 0:  # pad first
			n_pad = self.period - (t % self.period)
			x = F.pad(x, (0, n_pad), "reflect")
			t = t + n_pad
		x = x.view(b, c, t // self.period, self.period)

		for i, layer in enumerate(self.convs):
			x = layer(x)
			x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
			fmap.append(x.detach().half())

		x = self.conv_post(x)
		fmap.append(x.detach().half())
		x = torch.flatten(x, 1, -1)

		return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.discriminators = nn.ModuleList(
			[
				DiscriminatorP(2),
				DiscriminatorP(3),
				DiscriminatorP(5),
				# DiscriminatorP(7),
				# DiscriminatorP(11),
			]
		)

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
	def __init__(self, use_spectral_norm=False):
		super().__init__()
		# norm_f = weight_norm if not use_spectral_norm else spectral_norm
		self.convs = nn.ModuleList(  # all convs had norm_f
			[
				Conv1d(1, 128, 15, 1, padding=7),
				Conv1d(128, 128, 41, 2, groups=4, padding=20),
				Conv1d(128, 256, 41, 2, groups=16, padding=20),
				Conv1d(256, 512, 41, 4, groups=16, padding=20),
				Conv1d(512, 1024, 41, 4, groups=16, padding=20),
				Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
				Conv1d(1024, 1024, 5, 1, padding=2),
			]
		)
		self.conv_post = Conv1d(1024, 1, 3, 1, padding=1)

	def forward(self, x):
		fmap = []
		for i, layer in enumerate(self.convs):
			x = layer(x)
			x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
			fmap.append(x)

		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)

		return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.discriminators = nn.ModuleList(
			[
				DiscriminatorS(use_spectral_norm=True),
				# DiscriminatorS(),
				# DiscriminatorS(),
			]
		)
		self.meanpools = nn.ModuleList(
			[AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
		)

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs
