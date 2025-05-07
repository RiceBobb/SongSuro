from torch import nn

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator


class ConditionalEncoder(nn.Module):
	def __init__(
		self,
		lyrics_input_channel: int,
		melody_input_channel: int,
		enhanced_channel: int,
		hidden_size: int,
		prior_output_dim: int,
	):
		super().__init__()
		self.lyrics_encoder = FFTEncoder(lyrics_input_channel)
		self.melody_encoder = FFTEncoder(melody_input_channel)

		# TODO: style embedding, timbre embedding will be used after training lyrics encoder  and melody encoder
		# self.timbre_encoder = TimbreEncoder(hidden_size, vq_input_dim=80)
		# self.style_encoder = StyleEncoder(hidden_size)

		self.enhanced_condition_encoder = FFTEncoder(enhanced_channel)
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

	def forward(self, lyrics, audio_filepath):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Lyrics text sequence.
		:param audio_filepath: audio file path for melodyU encoding.
		:return: conditional embedding vector
		"""
		lyrics_embedding = self.lyrics_encoder(lyrics)
		melodyU_embedding = self.melody_encoder(audio_filepath)

		# timbre_embedding = self.timbre_encoder(lyrics)
		# style_embedding = self.style_encoder(lyrics)

		summation_embedding = (
			lyrics_embedding + melodyU_embedding
			# + timbre_embedding + style_embedding
		)

		enhanced_condition_embedding = self.enhanced_condition_encoder(
			summation_embedding, summation_embedding.size
		)
		prior = self.prior_estimator(enhanced_condition_embedding)

		return enhanced_condition_embedding, prior
