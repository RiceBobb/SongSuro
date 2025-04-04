"""
Code from  Relative positional encoding and FFT block
Lyrics encoder and enhanced condition encoders.

Relative position encoder github: https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
FastSpeech2 transformer github: https://github.com/ming024/FastSpeech2/blob/master/transformer/Models.py
"""

import torch.nn as nn


class LyricsEncoder(nn.Module):
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
