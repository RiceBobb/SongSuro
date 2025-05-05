"""
Includes code for relative positional encoding and the FFT block,
as well as the lyrics encoder and enhanced condition encoders.

FastSpeech2 github: https://github.com/ming024/FastSpeech2/blob/master/transformer/
Glow-tts github: https://github.com/jaywalnut310/glow-tts
"""

import math
import torch
from torch import nn

from songsuro.modules.fft.attentions import Encoder


class FFTEncoder(nn.Module):
	def __init__(
		self,
		input_channel,
		hidden_channels=192,
		filter_channels=768,
		n_heads=2,
		n_layers=4,
		kernel_size=9,
		p_dropout=0.0,
		window_size=None,
		block_length=None,
		mean_only=False,
	):
		super().__init__()

		self.input_channel = input_channel
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.window_size = window_size
		self.block_length = block_length
		self.mean_only = mean_only

		self.emb = nn.Embedding(input_channel, hidden_channels)
		nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

		self.encoder = Encoder(
			hidden_channels,
			filter_channels,
			n_heads,
			n_layers,
			kernel_size,
			p_dropout,
			window_size=window_size,
			block_length=block_length,
		)

	def forward(self, x, x_lengths):
		x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
		x = torch.transpose(x, 1, -1)  # [batch, hidden, time(seq_len)]
		# mask for padding
		x_mask = torch.unsqueeze(self.sequence_mask(x_lengths, x.size(2)), 1).to(
			x.dtype
		)

		x = self.encoder(x, x_mask)
		return x

	def sequence_mask(self, length, max_length=None):
		if max_length is None:
			max_length = length.max()
		x = torch.arange(max_length, dtype=length.dtype, device=length.device)
		return x.unsqueeze(0) < length.unsqueeze(1)
