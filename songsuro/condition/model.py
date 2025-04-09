from torch import nn

from songsuro.condition.encoder.style import StyleEncoder
from songsuro.condition.encoder.timbre import TimbreEncoder


class ConditionalEncoder(nn.Module):
	def __init__(self, hidden_size: int):
		super().__init__()
		self.timbre_encoder = TimbreEncoder(hidden_size, vq_input_dim=80)
		self.style_encoder = StyleEncoder(hidden_size)

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

		return style_embedding + timbre_embedding
