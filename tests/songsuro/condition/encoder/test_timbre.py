import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from songsuro.modules.commons.conv import ConvBlocks
from songsuro.condition.encoder.timbre import TimbreEncoder


class TestTimbreEncoder:
	@pytest.fixture
	def encoder(self):
		return TimbreEncoder(hidden_size=320, vq_input_dim=80)

	def test_initialization(self, encoder):
		"""Test that the encoder initializes correctly"""
		assert isinstance(encoder, nn.Module)
		assert isinstance(encoder.global_conv_in, nn.Conv1d)
		assert isinstance(encoder.global_encoder, ConvBlocks)

	def test_encode_spk_embed_shape(self, encoder):
		"""Test that encode_spk_embed returns the expected shape"""
		batch_size = 4
		seq_len = 100
		input_dim = 80

		x = torch.randn(batch_size, input_dim, seq_len)
		spk_embed = encoder.encode_spk_embed(x)

		# The output should be (batch_size, hidden_size, 1)
		assert spk_embed.shape == (batch_size, encoder.hidden_size, 1)

	def test_encode_spk_embed_zero_padding(self, encoder):
		"""Test that zero padding is handled correctly"""
		batch_size = 2
		seq_len = 50
		input_dim = 80  # 2-d mel-spectrogram

		# Create input with some zero padding
		x = torch.randn(batch_size, input_dim, seq_len)
		# Set half of the sequence to zeros for the first batch
		x[0, :, seq_len // 2 :] = 0

		spk_embed = encoder.encode_spk_embed(x)

		# Check that output shape is correct
		assert spk_embed.shape == (batch_size, encoder.hidden_size, 1)

	def test_encode_spk_embed_all_zeros(self, encoder):
		"""Test behavior with all-zero input"""
		batch_size = 2
		seq_len = 30
		input_dim = 80

		x = torch.zeros(batch_size, input_dim, seq_len)
		spk_embed = encoder.encode_spk_embed(x)

		# Output should be all zeros
		assert torch.all(spk_embed == 0)

	def test_encode_spk_embed_forward_consistency(self, encoder):
		"""Test that the encoder produces consistent outputs for the same input"""
		batch_size = 3
		seq_len = 40
		input_dim = 80

		x = torch.randn(batch_size, input_dim, seq_len)

		# Run the encoder twice with the same input
		spk_embed1 = encoder.encode_spk_embed(x)
		spk_embed2 = encoder.encode_spk_embed(x)

		# Outputs should be identical
		assert_close(spk_embed1, spk_embed2)

	@pytest.mark.parametrize("batch_size", [1, 4, 8])
	@pytest.mark.parametrize("seq_len", [20, 100, 200])
	def test_encode_spk_embed_various_shapes(self, encoder, batch_size, seq_len):
		"""Test encoder with various input shapes"""
		input_dim = 80

		x = torch.randn(batch_size, input_dim, seq_len)
		spk_embed = encoder.encode_spk_embed(x)

		assert spk_embed.shape == (batch_size, encoder.hidden_size, 1)
