"""
Code from  Relative positional encoding and FFT block
Lyrics encoder and enhanced condition encoders.

Relative position encoder github: https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
FastSpeech2 transformer github: https://github.com/ming024/FastSpeech2/blob/master/transformer/
"""

import torch.nn as nn
from songsuro.modules.fft.relative_pos_encoding import RelativePosition
from songsuro.modules.fft.layers import FFTBlock


# TODO: Use config yaml file later
class FFTransformerBlock(nn.Module):
	def __init__(self, d_model=192, d_k=64, d_v=64, n_head=2, d_inner=768):
		super().__init__()
		max_relative_position = 4

		self.ff_block = FFTBlock(
			hidden_size_d_model=d_model,
			d_k=d_k,
			d_v=d_v,
			d_inner=d_inner,
			n_head=n_head,
			kernel_size=9,
		)
		self.relative_position = RelativePosition(d_model, max_relative_position)

	def forward(self, x, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1).unsqueeze(2)
		x, _ = self.ff_block(x, mask=mask)
		return x


class LyricsEncoder(nn.Module):
	"""
	Block is 4 in HiddenSinger. - n_layers = 4
	"""

	def __init__(self, vocab_size, d_model=192, n_layers=4):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, d_model)
		self.blocks = nn.ModuleList(
			[FFTransformerBlock(d_model=d_model) for _ in range(n_layers)]
		)
		self.prior = nn.Linear(d_model, d_model)

	def forward(self, x, mask=None):
		x = self.embed(x)
		for block in self.blocks:
			x = block(x, mask)
		return self.prior(x)


class EnhancedConditionEncoder(nn.Module):
	def __init__(self, d_model=192, n_layers=4):
		super().__init__()
		self.blocks = nn.ModuleList(
			[FFTransformerBlock(d_model=d_model) for _ in range(n_layers)]
		)
		self.prior = nn.Linear(d_model, d_model)

	def forward(self, x, mask=None):
		for block in self.blocks:
			x = block(x, mask)
		return self.prior(x)
