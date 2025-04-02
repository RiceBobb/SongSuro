"""
Code from TCSinger (EMNLP 2024)
https://github.com/AaronZ345/TCSinger
"""

from torch import nn

from songsuro.modules.commons.leftpad_conv import ConvBlocks as LeftPadConvBlocks
from songsuro.utils.util import group_hidden_by_segs
from songsuro.modules.TCSinger.vq import VQEmbeddingEMA, VectorQuantiser


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		try:
			nn.init.xavier_uniform_(m.weight.data)
			m.bias.data.fill_(0)
		except AttributeError:
			print("Skipping initialization of ", classname)


# clustering style encoder
class StyleEncoder(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		self.hidden_size = hparams["hidden_size"]
		self.vq_ph_channel = hparams["vq_ph_channel"]

		self.ph_conv_in = nn.Conv1d(80, self.hidden_size, 1)
		self.ph_encoder = LeftPadConvBlocks(
			self.hidden_size,
			self.hidden_size,
			None,
			kernel_size=3,
			layers_in_block=2,
			is_BTC=False,
			num_layers=5,
		)  # Phoneme Encoder (음소)
		self.ph_postnet = LeftPadConvBlocks(
			self.hidden_size,
			self.hidden_size,
			None,
			kernel_size=3,
			layers_in_block=2,
			is_BTC=False,
			num_layers=5,
		)
		self.ph_latents_proj_in = nn.Conv1d(
			self.hidden_size, hparams["vq_ph_channel"], 1
		)  # Linear Projection
		if self.hparams["vq"] == "ema":
			self.vq = VQEmbeddingEMA(
				hparams["vq_ph_codebook_dim"], hparams["vq_ph_channel"]
			)
		elif self.hparams["vq"] == "cvq":
			self.vq = VectorQuantiser(
				hparams["vq_ph_codebook_dim"],
				hparams["vq_ph_channel"],
				hparams["vq_ph_beta"],
			)
		self.ph_latents_proj_out = nn.Conv1d(
			hparams["vq_ph_channel"], self.hidden_size, 1
		)

		self.apply(weights_init)

	def encode_ph_vqcode(
		self, x, in_nonpadding, in_mel2ph, max_ph_length, ph_nonpadding
	):
		# forward encoder
		x_ph = self.ph_conv_in(x) * in_nonpadding
		ph_z_e_x = (
			self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding
		)  # (B, C, T)

		# Forward ph postnet
		ph_z_e_x = group_hidden_by_segs(
			ph_z_e_x, in_mel2ph, max_ph_length, is_BHT=True
		)[0]  # Grouping Mel-spectrogram

		ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
		ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
		ph_vqcode = self.vq.encode_indice(ph_z_e_x)
		return ph_vqcode

	def vqcode_to_latent(self, ph_vqcode):
		# VQ process
		z_q_x_bar_flatten = self.vq.decode(ph_vqcode)
		ph_z_q_x_bar_ = z_q_x_bar_flatten.view(
			ph_vqcode.size(0), ph_vqcode.size(1), self.vq_ph_channel
		)
		ph_z_q_x_bar = ph_z_q_x_bar_.permute(0, 2, 1).contiguous()
		ph_z_q_x_bar = self.ph_latents_proj_out(ph_z_q_x_bar)
		return ph_z_q_x_bar

	def encode_style(self, x, in_nonpadding, in_mel2ph, ph_nonpadding, ph_lengths):
		# forward encoder
		x_ph = self.ph_conv_in(x) * in_nonpadding
		ph_z_e_x = (
			self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding
		)  # (B, C, T)

		# Forward ph postnet
		ph_z_e_x = group_hidden_by_segs(
			ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True
		)[0]

		ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
		ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)

		ph_z_q_x_s, vq_loss, indices, _ = self.vq(ph_z_e_x)
		ph_z_q_x_st = self.ph_latents_proj_out(ph_z_q_x_s)

		return ph_z_q_x_st, vq_loss, indices
