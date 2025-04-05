import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from songsuro.autoencoder.decoder.generator import Generator, ResBlock1, ResBlock2


class TestGenerator:
	@pytest.fixture
	def config(self):
		# Create a mock object for Generator configuration
		h = MagicMock()
		h.resblock_kernel_sizes = [3, 7, 11]
		h.upsample_rates = [8, 8, 2, 2]
		h.upsample_kernel_sizes = [16, 16, 4, 4]
		h.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
		h.upsample_initial_channel = 512
		h.resblock = "1"
		h.input_channels = 128  # Important: Set decoder input channels to 128
		return h

	@pytest.fixture
	def generator(self, config):
		return Generator(config)

	def test_initialization(self, generator, config):
		# Test initialization
		assert generator.h == config
		assert generator.num_kernels == len(config.resblock_kernel_sizes)
		assert generator.num_upsamples == len(config.upsample_rates)
		assert len(generator.ups) == generator.num_upsamples
		assert (
			len(generator.resblocks) == generator.num_upsamples * generator.num_kernels
		)

		# Check layer structure
		assert isinstance(generator.conv_pre, nn.Conv1d)
		assert isinstance(generator.conv_post, nn.Conv1d)

		# Check upsampling layers
		for up in generator.ups:
			assert isinstance(up, nn.ConvTranspose1d)

		# Check ResBlock layers
		for resblock in generator.resblocks:
			assert isinstance(resblock, (ResBlock1, ResBlock2))

	def test_forward(self, generator):
		# Create input tensor (batch size 2, channels 80, time steps 100)
		x = torch.randn(2, 128, 100)

		# Run forward pass
		output = generator(x)

		# Check output shape
		assert isinstance(output, torch.Tensor)
		assert output.shape[0] == 2  # batch size
		assert output.shape[1] == 1  # output channel

		# Check output length based on upsampling rates
		expected_length = 100
		for rate in generator.h.upsample_rates:
			expected_length *= rate
		assert output.shape[2] == expected_length

		# Check if output values are within tanh range (-1 to 1)
		assert torch.all(output >= -1.0)
		assert torch.all(output <= 1.0)

	def test_remove_weight_norm(self, generator):
		# Check if weight_norm exists before removal
		assert hasattr(generator.conv_pre, "weight_v")
		assert hasattr(generator.conv_post, "weight_v")

		# Remove weight_norm
		generator.remove_weight_norm()

		# Check if weight_norm was removed
		assert not hasattr(generator.conv_pre, "weight_v")
		assert not hasattr(generator.conv_post, "weight_v")

		for up in generator.ups:
			assert not hasattr(up, "weight_v")

	def test_generator_with_different_resblock(self, config):
		# Test using ResBlock2
		config.resblock = "2"
		generator = Generator(config)

		# Create input tensor
		x = torch.randn(2, 128, 100)

		# Run forward pass
		output = generator(x)

		# Check output shape
		assert isinstance(output, torch.Tensor)
		assert output.shape[0] == 2
		assert output.shape[1] == 1

		# Check that ResBlock2 is used
		assert isinstance(generator.resblocks[0], ResBlock2)
