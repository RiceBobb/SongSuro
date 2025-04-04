# This code is originated by Diffwave, copyright 2020 LMNT, Inc.
# I modified the code, and original code is under the Apache 2.0 License
import torch
import torch.nn as nn
import torch.nn.functional as F


Linear = nn.Linear


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
		low_idx = torch.floor(t).long()
		high_idx = torch.ceil(t).long()
		low = self.embedding[low_idx]
		high = self.embedding[high_idx]
		return low + (high - low) * (t - low_idx)

	def _build_embedding(self, max_steps):
		steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
		dims = torch.arange(64).unsqueeze(0)  # [1,64]
		table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
		table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
		return table
