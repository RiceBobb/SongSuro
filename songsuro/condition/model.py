import torch
from torch import nn

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator


class ConditionalEncoder(nn.Module):
	def __init__(
		self,
		lyrics_input_channel: int,
		melody_input_channel: int,
		prior_output_dim: int,
		hidden_size: int = 192,
	):
		super().__init__()

		self.lyrics_encoder = FFTEncoder(lyrics_input_channel)
		# self.melody_encoder = FFTEncoder(melody_input_channel)
		# self.timbre_encoder = TimbreEncoder(hidden_size=hidden_size, vq_input_dim=80)
		# self.style_encoder = StyleEncoder(hidden_size=hidden_size)

		self.enhanced_condition_encoder = FFTEncoder()
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

	def forward(
		self,
		lyrics,
		lyrics_lengths: torch.Tensor,
		quantized_f0,
		quantized_f0_lengths: torch.Tensor,
	):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Tokenized lyrics sequence. Should be a tensor or can be converted to one.
		:param lyrics_lengths: Lengths of the lyrics sequences. The shape is [batch_size]
		:param quantized_f0: Pre-processed quantized f0 data.
		:param quantized_f0_lengths: Lengths of the quantized f0 sequences. The shape is [batch_size].
		:return: conditional embedding vector and prior
		"""
		if not isinstance(lyrics, torch.Tensor):
			raise TypeError("Input lyrics must be a tensor")

		# TODO: expand
		lyrics_embedding = self.lyrics_encoder(lyrics, lyrics_lengths)

		# if not isinstance(quantized_ f0, torch.Tensor):
		#     raise TypeError("Input lyrics must be a tensor")
		# quantized_f0 = torch.tensor(quantized_f0).unsqueeze(0)

		# TODO: We need to apply pitch embedding and then encode it with FFTEncoder
		# melodyU_embedding = self.melody_encoder(quantized_f0, quantized_f0_lengths)
		# timbre_embedding = self.timbre_encoder(lyrics)
		# style_embedding = self.style_encoder(lyrics)

		summation_embedding = (
			lyrics_embedding
			# + melodyU_embedding
			# + timbre_embedding + style_embedding
		)

		summation_lengths = torch.cat(
			(
				lyrics_lengths,
				# quantized_f0_lengths,
				# timbre_lengths,
				# style_lengths
			)
		)

		enhanced_condition_embedding = self.enhanced_condition_encoder(
			summation_embedding, summation_lengths
		)

		prior = self.prior_estimator(torch.mean(enhanced_condition_embedding, -1))

		return enhanced_condition_embedding, prior
