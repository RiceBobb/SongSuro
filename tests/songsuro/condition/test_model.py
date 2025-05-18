import pytest
import torch
import os
import pathlib

from songsuro.condition.model import ConditionalEncoder

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
resource_dir = os.path.join(root_dir, "resources")


class TestConditionalEncoder:
	@pytest.fixture
	def encoder(self):
		return ConditionalEncoder(
			lyrics_input_channel=512,
			melody_input_channel=128,
		)

	@pytest.fixture
	def sample_lyrics(self):
		# Create a sample lyrics tensor (batch_size, seq_len)
		return torch.randint(0, 100, (1, 50), dtype=torch.long)

	@pytest.fixture
	def sample_audio_file(self):
		"""Create a temporary sine wave audio file for testing."""
		yield os.path.join(resource_dir, "sample_only_voice.wav")

	@pytest.fixture
	def sample_f0(self):
		# Create a sample quantized f0 tensor (batch_size, seq_len)
		return torch.randint(0, 100, (1, 200), dtype=torch.long)

	def test_initialization(self, encoder):
		"""Test if the encoder initializes correctly with all components."""
		assert hasattr(encoder, "lyrics_encoder")
		assert hasattr(encoder, "enhanced_condition_encoder")
		assert hasattr(encoder, "prior_estimator")

	def test_forward_with_tensors(self, encoder, sample_lyrics, sample_f0):
		"""Test forward pass with tensor inputs."""
		condition_embedding, prior = encoder(sample_lyrics, quantized_f0=sample_f0)

		# Check output shapes and types
		assert isinstance(condition_embedding, torch.Tensor)
		assert isinstance(prior, torch.Tensor)
		assert condition_embedding.dim() == 3  # (batch, channels, seq_len)
		assert prior.dim() == 2  # Assuming prior has shape (batch, dim, seq_len)

	@pytest.mark.parametrize("batch_size,seq_len", [(1, 50), (2, 30), (4, 20)])
	def test_different_batch_sizes(self, encoder, batch_size, seq_len):
		"""Test with different batch sizes and sequence lengths."""
		lyrics = torch.randint(0, 100, (batch_size, seq_len))
		f0 = torch.randint(0, 100, (batch_size, seq_len * 2))  # Typically f0 is longer

		condition_embedding, prior = encoder(lyrics, quantized_f0=f0)

		assert condition_embedding.size(0) == batch_size
		assert prior.size(0) == batch_size

	def test_input_output_consistency(self, encoder, sample_lyrics, sample_f0):
		"""Test that the same input produces the same output."""
		embedding1, prior1 = encoder(sample_lyrics, quantized_f0=sample_f0)
		embedding2, prior2 = encoder(sample_lyrics, quantized_f0=sample_f0)

		torch.testing.assert_close(embedding1, embedding2)
		torch.testing.assert_close(prior1, prior2)
