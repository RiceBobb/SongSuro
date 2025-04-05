# This code is originated by Diffwave, copyright 2020 LMNT, Inc.
# I modified the code, and original code is under the Apache 2.0 License
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

Linear = nn.Linear


def Conv1d(*args, **kwargs):
	layer = nn.Conv1d(*args, **kwargs)
	nn.init.kaiming_normal_(layer.weight)
	return layer


class DiffusionEmbedding(nn.Module):
	def __init__(self, max_steps):
		super().__init__()
		self.register_buffer(
			"embedding", self._build_embedding(max_steps), persistent=False
		)  # Here set the 'self.embedding'
		self.projection1 = Linear(128, 512)  # First FC from Diffusion-step embedding
		self.projection2 = Linear(512, 512)  # Second FC from Diffusion-step embedding

	def forward(self, diffusion_step):
		# Input will be just an integer or float represents diffusion step number
		if diffusion_step.dtype in [torch.int32, torch.int64]:
			x = self.embedding[diffusion_step]
		else:
			x = self._lerp_embedding(diffusion_step)
		x = self.projection1(x)
		x = F.mish(x)
		x = self.projection2(x)
		# I do not put the activation here followed by the HiddenSinger paper
		return x

	def _lerp_embedding(self, t):
		# Ensure t is 2D: (batch_size, step_num)

		low_idx = torch.floor(t).long()
		high_idx = torch.ceil(t).long()

		# Clamp indices to valid range
		low_idx = torch.clamp(low_idx, 0, self.embedding.shape[0] - 1)
		high_idx = torch.clamp(high_idx, 0, self.embedding.shape[0] - 1)

		# Gather embeddings for low and high indices
		low = self.embedding[low_idx]
		high = self.embedding[high_idx]

		# Compute interpolation weights
		weights = (t - low_idx.float()).unsqueeze(-1)

		# Perform linear interpolation
		interpolated = low + (high - low) * weights

		return interpolated

	def _build_embedding(self, max_steps):
		steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
		dims = torch.arange(64).unsqueeze(0)  # [1,64]
		table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
		table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
		return table


class ResidualBlock(nn.Module):
	def __init__(
		self,
		input_condition_dim: int,
		channel_size: int = 256,
		kernel_size: int = 3,
		dilation: int = 1,
	):
		super().__init__()

		self.channel_size = channel_size
		self.conditioner_projection = Linear(
			in_features=input_condition_dim, out_features=channel_size * 2
		)  # (B, 2C, L)
		self.dilated_conv = Conv1d(
			channel_size,
			channel_size * 2,
			kernel_size,
			dilation=dilation,
			padding=dilation,
		)  # (B, 2C, L)
		self.output_projection = Conv1d(channel_size, channel_size * 2, kernel_size=1)

	def forward(self, x, diffusion_step_embedding, condition_embedding):
		"""
		Forward step out of the single ResidualBlock of the denoiser.

		:param x: Input to the Residual Block which is concatenation of the previous step result and prior estimator result.
		:param diffusion_step_embedding: t-th step embedding from the DiffusionEmbedding class.
		:param condition_embedding: condition embedding from the conditioner Encoder.
			Shape is (B, input_condition_dim, L).
		:return:
		"""
		diffusion_step_embedding = diffusion_step_embedding.unsqueeze(
			-1
		)  # Regulation of diffusion_step_embedding
		y = x + diffusion_step_embedding  # (B, C, L)
		after_dil_conv = self.dilated_conv(y)  # (B, 2C, L)

		# condition path
		condition_embedding = condition_embedding.transpose(1, 2)
		condition = self.conditioner_projection(condition_embedding)
		condition = condition.transpose(1, 2)
		condition_residual = after_dil_conv + condition

		gate, filter = torch.chunk(condition_residual, 2, dim=1)
		after_gate = torch.sigmoid(gate) * torch.tanh(filter)
		after_gate = self.output_projection(after_gate)
		residual, skip = torch.chunk(after_gate, 2, dim=1)
		return (residual + y) / sqrt(2.0), skip  # (B, C, L), (B, C, L)
