import torch
from torch import nn


class LayerNorm(nn.Module):
	def __init__(self, channels, eps=1e-4):
		super().__init__()
		self.channels = channels
		self.eps = eps

		self.gamma = nn.Parameter(torch.ones(channels))
		self.beta = nn.Parameter(torch.zeros(channels))

	def forward(self, x):
		n_dims = len(x.shape)
		mean = torch.mean(x, 1, keepdim=True)
		variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

		x = (x - mean) * torch.rsqrt(variance + self.eps)

		shape = [1, -1] + [1] * (n_dims - 2)
		x = x * self.gamma.view(*shape) + self.beta.view(*shape)
		return x
