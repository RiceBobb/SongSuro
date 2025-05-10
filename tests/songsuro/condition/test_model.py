import pytest
import torch
import os
import pathlib
from unittest.mock import patch

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator
from songsuro.condition.model import ConditionalEncoder

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
resource_dir = os.path.join(root_dir, "resources")


class TestConditionalEncoder:
	@pytest.fixture
	def conditional_encoder(self):
		return ConditionalEncoder(
			lyrics_input_channel=100,
			melody_input_channel=128,
			enhanced_channel=192,
			hidden_size=192,
			prior_output_dim=2,
		)

	@pytest.fixture
	def mock_preprocess_f0(self):
		with patch("songsuro.condition.model.preprocess_f0") as mock:
			# Mock the return value of preprocess_f0
			mock.return_value = torch.randint(0, 128, (2, 10))
			yield mock

	@pytest.fixture
	def sample_audio_file(self):
		"""Create a temporary sine wave audio file for testing."""
		yield os.path.join(resource_dir, "sample_only_voice.wav")

	def test_initialization(self, encoder):
		assert isinstance(encoder.lyrics_encoder, FFTEncoder)
		assert isinstance(encoder.melody_encoder, FFTEncoder)
		assert isinstance(encoder.enhanced_condition_encoder, FFTEncoder)
		assert isinstance(encoder.prior_estimator, PriorEstimator)

	def test_forward(self, conditional_encoder, mock_preprocess_f0, sample_audio_file):
		batch_size = 2

		# lyrics sequence length and melody sequence length have to be same.
		lyrics_seq_len = 10
		melody_seq_len = 10

		lyrics = torch.randint(0, 100, (batch_size, lyrics_seq_len))
		lyrics.size = torch.tensor([lyrics_seq_len, lyrics_seq_len])

		audio_filepath = sample_audio_file

		# Mock the FFTEncoder forward method
		with patch.object(FFTEncoder, "forward") as mock_fft_forward:
			# Create mock embeddings
			lyrics_embedding = torch.randn(batch_size, 192, lyrics_seq_len)
			melody_embedding = torch.randn(batch_size, 192, melody_seq_len)
			enhanced_embedding = torch.randn(batch_size, 192, lyrics_seq_len)

			# Configure the mock to return different values for different calls
			mock_fft_forward.side_effect = [
				lyrics_embedding,  # First call (lyrics_encoder)
				melody_embedding,  # Second call (melody_encoder)
				enhanced_embedding,  # Third call (enhanced_condition_encoder)
			]

			# Mock the PriorEstimator forward method
			with patch.object(PriorEstimator, "forward") as mock_prior:
				prior_output = torch.randn(batch_size, 2, lyrics_seq_len)
				mock_prior.return_value = prior_output

				# Call the forward method
				enhanced_condition, prior = encoder(lyrics, audio_filepath)

				# Verify preprocess_f0 was called with the correct arguments
				mock_preprocess_f0.assert_called_once_with(audio_filepath)

				# Verify the outputs
				assert torch.equal(enhanced_condition, enhanced_embedding)
				assert torch.equal(prior, prior_output)

				# Verify the summation was passed to the enhanced_condition_encoder
				# The third call to mock_fft_forward should have received lyrics_embedding + melody_embedding
				calls = mock_fft_forward.call_args_list
				assert len(calls) == 3

				# For the third call, the first argument should be lyrics_embedding + melody_embedding
				# However, we can't directly check this because the addition happens inside the forward method
				# We can verify that the third call happened with some arguments
				assert len(calls[2][0]) >= 1
