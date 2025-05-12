import pytest
import torch
import os
import pathlib
from unittest.mock import patch

from songsuro.condition.model import ConditionalEncoder

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
resource_dir = os.path.join(root_dir, "resources")


class TestConditionalEncoder:
	@pytest.fixture
	def encoder(self):
		return ConditionalEncoder(
			lyrics_input_channel=512,
			melody_input_channel=128,
			device=torch.device("cpu"),
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
		assert hasattr(encoder, "melody_encoder")
		assert hasattr(encoder, "enhanced_condition_encoder")
		assert hasattr(encoder, "prior_estimator")
		assert encoder.device == torch.device("cpu")

	def test_forward_with_tensors(self, encoder, sample_lyrics, sample_f0):
		"""Test forward pass with tensor inputs."""
		condition_embedding, prior = encoder(sample_lyrics, quantized_f0=sample_f0)

		# Check output shapes and types
		assert isinstance(condition_embedding, torch.Tensor)
		assert isinstance(prior, torch.Tensor)
		assert condition_embedding.dim() == 3  # (batch, channels, seq_len)
		assert prior.dim() == 3  # Assuming prior has shape (batch, dim, seq_len)

	# preprocess_f0 is called in conditional encoder model located in preprocess.py
	@patch("songsuro.condition.model.preprocess_f0")
	def test_forward_with_audio_file(
		self, mock_preprocess, encoder, sample_lyrics, sample_f0, sample_audio_file
	):
		"""Test forward pass with audio filepath."""
		# Mock the preprocess_f0 function to return our sample f0
		mock_preprocess.return_value = sample_f0

		condition_embedding, prior = encoder(
			sample_lyrics, audio_filepath=sample_audio_file
		)

		# Verify preprocess_f0 was called with the correct path
		mock_preprocess.assert_called_once_with(sample_audio_file)

		# Check output shapes and types
		assert isinstance(condition_embedding, torch.Tensor)
		assert isinstance(prior, torch.Tensor)

	def test_device_change(self, encoder):
		"""Test if the model can be moved to a different device."""
		# Skip if CUDA is not available
		if not torch.cuda.is_available():
			pytest.skip("CUDA not available, skipping device test")

		encoder = encoder.to("cuda")
		assert encoder.device == torch.device("cuda")

		# Move back to CPU for cleanup
		encoder = encoder.to("cpu")
		assert encoder.device == torch.device("cpu")

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

	def test_gradients_flow(self, encoder, sample_lyrics, sample_f0):
		"""
		Test that gradients flow through the model.
		This function verifies that gradients are properly computed and propagated
		through the encoder model when processing lyrics and f0 (fundamental frequency) inputs.
		"""
		# Set the model to training mode to enable gradient computation
		encoder.train()

		# Convert input tensors to long type (typically used for embedding indices)
		lyrics = sample_lyrics
		f0 = sample_f0

		# Forward pass through the encoder
		# This generates embeddings and prior distributions from the inputs
		embedding, prior = encoder(lyrics, quantized_f0=f0)

		# Create a dummy loss by summing all outputs
		# In a real scenario, this would be replaced by an actual loss function
		loss = embedding.sum() + prior.sum()

		# Backpropagate the gradients through the network
		loss.backward()

		# Check if any parameters in the encoder have received gradients
		has_grad = any(param.grad is not None for param in encoder.parameters())

		# Assert that at least some parameters have gradients
		# If this fails, it means gradients aren't flowing properly through the model
		assert has_grad, "No gradients were computed for encoder parameters"

		# Alternative: Check specific parameters if needed
		for name, param in encoder.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"

	@pytest.mark.parametrize("device", ["cpu", "cuda"])
	def test_device_consistency(self, encoder, sample_lyrics, sample_f0, device):
		"""Test that the model works consistently across devices."""
		if device == "cuda" and not torch.cuda.is_available():
			pytest.skip("CUDA not available")

		encoder = encoder.to(device)
		lyrics = sample_lyrics.to(device)
		f0 = sample_f0.to(device)

		embedding, prior = encoder(lyrics, quantized_f0=f0)

		assert embedding.device.type == device
		assert prior.device.type == device
