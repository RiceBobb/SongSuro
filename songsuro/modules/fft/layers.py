import torch

from .sublayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(torch.nn.Module):
	"""
	FFT Block in HiddenSinger(), FastSpeech2(), and GlowTTS().
	- block: 4 (FFT blocks - HiddenSinger)
	- heads: 2 (attention heads - HiddenSinger)
	- hidden size: 192 (latent representation size - HiddenSinger)
	- kernel size: 9 (Convolution kernel size - HiddenSinger)

	    - d_inner: 768 (Convolution filter size - GlowTTS referenced in HiddenSinger)

	-> d_model is hidden size. In the paper, d_model is latent representation size.
	"""

	# TODO: 4/7 - Merge RelativePosition and FFTBlock
	def __init__(
		self,
		d_k,
		d_v,
		d_inner,
		n_head=2,
		hidden_size_d_model=192,
		kernel_size=9,
		dropout=0.1,
	):
		super().__init__()
		self.slf_attn = MultiHeadAttention(
			n_head=n_head,
			d_model=hidden_size_d_model,
			d_k=d_k,
			d_v=d_v,
			dropout=dropout,
		)
		self.pos_ffn = PositionwiseFeedForward(
			d_inner, hidden_size_d_model, kernel_size, dropout=dropout
		)

	def forward(self, enc_input, mask=None, slf_attn_mask=None):
		enc_output, enc_slf_attn = self.slf_attn(
			enc_input, enc_input, enc_input, mask=slf_attn_mask
		)
		enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

		enc_output = self.pos_ffn(enc_output)
		enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

		return enc_output, enc_slf_attn
