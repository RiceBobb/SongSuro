from torch import nn

from songsuro.modules.commons.conv import ConvBlocks
from songsuro.utils.util import temporal_avg_pool


class TimbreEncoder(nn.Module):
	"""
	Code from TCSinger (EMNLP 2024)
	Timbre Encoder for singer identity and timbre
	"""

	def __init__(
		self, hidden_size: int, vq_input_dim: int
	):  # The recommended vq_input_dim is 80. (Of course can change it)
		super().__init__()

		self.hidden_size = hidden_size
		self.global_conv_in = nn.Conv1d(vq_input_dim, self.hidden_size, 1)
		self.global_encoder = ConvBlocks(
			self.hidden_size,
			self.hidden_size,
			None,
			kernel_size=31,
			layers_in_block=2,
			is_BTC=False,
			num_layers=5,
		)  # Timbre Encoder

	def encode_spk_embed(self, x):  # Forward Timbre Encoder
		in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
		# forward encoder
		x_global = self.global_conv_in(x) * in_nonpadding
		global_z_e_x = (
			self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
		)
		# group by hidden to phoneme-level
		global_z_e_x = temporal_avg_pool(
			x=global_z_e_x, mask=(in_nonpadding == 0)
		)  # (B, C, T) -> (B, C, 1)
		# (Batch, Channel Dimension, Sequence Length (Time)) -> (Batch, Channel Dimension, 1)
		spk_embed = global_z_e_x
		return spk_embed
