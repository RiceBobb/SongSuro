import torch
from torch.nn import functional as F


class LengthRegulator(torch.nn.Module):
	def __init__(self, pad_value=0.0):
		super(LengthRegulator, self).__init__()
		self.pad_value = pad_value

	def forward(self, dur, dur_padding=None, alpha=1.0):
		"""
		Example (no batch dim version):
		    1. dur = [2,2,3]
		    2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
		    3. token_mask = [[1,1,0,0,0,0,0],
		                     [0,0,1,1,0,0,0],
		                     [0,0,0,0,1,1,1]]
		    4. token_idx * token_mask = [[1,1,0,0,0,0,0],
		                                 [0,0,2,2,0,0,0],
		                                 [0,0,0,0,3,3,3]]
		    5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

		:param dur: Batch of durations of each frame (B, T_txt)
		:param dur_padding: Batch of padding of each frame (B, T_txt)
		:param alpha: duration rescale coefficient
		:return:
		    mel2ph (B, T_speech)
		"""
		assert alpha > 0
		dur = torch.round(dur.float() * alpha).long()
		if dur_padding is not None:
			dur = dur * (1 - dur_padding.long())
		token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
		dur_cumsum = torch.cumsum(dur, 1)
		dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

		pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
		token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
			pos_idx < dur_cumsum[:, :, None]
		)
		mel2ph = (token_idx * token_mask.long()).sum(1)
		return mel2ph


class LayerNorm(torch.nn.LayerNorm):
	"""Layer normalization module.
	:param int nout: output dim size
	:param int dim: dimension to be normalized
	"""

	def __init__(self, nout, dim=-1):
		"""Construct an LayerNorm object."""
		super(LayerNorm, self).__init__(nout, eps=1e-12)
		self.dim = dim

	def forward(self, x):
		"""Apply layer normalization.
		:param torch.Tensor x: input tensor
		:return: layer normalized tensor
		:rtype torch.Tensor
		"""
		if self.dim == -1:
			return super(LayerNorm, self).forward(x)
		return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
