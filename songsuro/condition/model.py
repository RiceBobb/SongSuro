from torch import nn

from songsuro.condition.encoder.style import StyleEncoder
from songsuro.condition.encoder.timbre import TimbreEncoder
from songsuro.condition.prior_estimator import PriorEstimator


class ConditionalEncoder(nn.Module):
	def __init__(self, hidden_size: int, prior_output_dim: int):
		super().__init__()
		self.hidden_size = hidden_size
		self.prior_output_dim = prior_output_dim
		self.timbre_encoder = TimbreEncoder(hidden_size, vq_input_dim=80)
		self.style_encoder = StyleEncoder(hidden_size)
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

	def forward(self, lyrics, original_audio):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Lyrics text sequence.
		:param original_audio: Original audio mel-spectrogram.
		:return: conditional embedding vector
		"""
		timbre_embedding = self.timbre_encoder(lyrics)
		style_embedding = self.style_encoder(lyrics)

		prior = self.prior_estimator(style_embedding + timbre_embedding)

		return style_embedding + timbre_embedding, prior
