import torch
from torch import nn
from typing import Optional

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator


class ConditionalEncoder(nn.Module):
	def __init__(
		self,
		lyrics_input_channel: int,
		melody_input_channel: int,
		prior_output_dim: int,
		hidden_size: Optional[int] = 192,
	):
		super().__init__()

		self.lyrics_encoder = FFTEncoder(lyrics_input_channel)
		# self.melody_encoder = FFTEncoder(melody_input_channel)
		# self.timbre_encoder = TimbreEncoder(hidden_size=hidden_size, vq_input_dim=80)
		# self.style_encoder = StyleEncoder(hidden_size=hidden_size)

		self.enhanced_condition_encoder = FFTEncoder()
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

	def forward(self, lyrics, quantized_f0):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Tokenized lyrics sequence. Should be a tensor or can be converted to one.
		:param quantized_f0: Pre-processed quantized f0 data.
		:return: conditional embedding vector and prior
		"""
		if not isinstance(lyrics, torch.Tensor):
			lyrics = torch.tensor(lyrics)

		lyrics_lengths = torch.tensor([lyrics.shape[1]])
		lyrics_embedding = self.lyrics_encoder(lyrics, lyrics_lengths)
		# TODO: expand
		# if not isinstance(quantized_f0, torch.Tensor):
		# 	quantized_f0 = torch.tensor(quantized_f0).unsqueeze(0)
		# quantized_f0_lengths = torch.tensor([quantized_f0.shape[1]])

		# TODO: We need to apply pitch embedding and then encode it with FFTEncoder
		# melodyU_embedding = self.melody_encoder(quantized_f0, quantized_f0_lengths)
		# timbre_embedding = self.timbre_encoder(lyrics)
		# style_embedding = self.style_encoder(lyrics)

		summation_embedding = (
			lyrics_embedding
			# + melodyU_embedding
			# + timbre_embedding + style_embedding
		)

		summation_lengths = torch.tensor([summation_embedding.shape[2]])

		enhanced_condition_embedding = self.enhanced_condition_encoder(
			summation_embedding, summation_lengths
		)

		prior = self.prior_estimator(torch.mean(enhanced_condition_embedding, -1))

		return enhanced_condition_embedding, prior
