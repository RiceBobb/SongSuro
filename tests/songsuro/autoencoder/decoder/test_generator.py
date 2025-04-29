import pytest
import torch
import torch.nn as nn

from songsuro.autoencoder.decoder.generator import Generator, ResBlock1, ResBlock2


@pytest.fixture
def generator_args():
	# Arguments for Generator instantiation
	return dict(
		generator_input_channels=128,
		resblock="1",
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[8, 8, 2, 2],
		upsample_initial_channel=512,
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
	)


@pytest.fixture
def generator(generator_args):
	return Generator(**generator_args)


def test_initialization(generator, generator_args):
	# Test if the generator is initialized with correct structure
	assert generator.num_kernels == len(generator_args["resblock_kernel_sizes"])
	assert generator.num_upsamples == len(generator_args["upsample_rates"])
	assert len(generator.ups) == generator.num_upsamples
	assert len(generator.resblocks) == generator.num_upsamples * generator.num_kernels

	# Check pre and post conv layers
	assert isinstance(generator.conv_pre, nn.Conv1d)
	assert isinstance(generator.conv_post, nn.Conv1d)

	# Check upsampling layers
	for up in generator.ups:
		assert isinstance(up, nn.ConvTranspose1d)

	# Check ResBlock layers
	for resblock in generator.resblocks:
		assert isinstance(resblock, (ResBlock1, ResBlock2))


def test_forward(generator, generator_args):
	# Create input tensor (batch size 2, input channels 128, time steps 100)
	x = torch.randn(2, generator_args["generator_input_channels"], 100)

	# Run forward pass
	output = generator(x)

	# Check output shape
	assert isinstance(output, torch.Tensor)
	assert output.shape[0] == 2  # batch size
	assert output.shape[1] == 1  # output channel

	# Check output length based on upsampling rates
	expected_length = 100
	for rate in generator.upsample_rates:
		expected_length *= rate
	assert output.shape[2] == expected_length

	# Check if output values are within tanh range (-1 to 1)
	assert torch.all(output >= -1.0)
	assert torch.all(output <= 1.0)


def test_remove_weight_norm(generator):
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


def test_generator_with_different_resblock(generator_args):
	# Test Generator with ResBlock2
	generator_args["resblock"] = "2"
	generator = Generator(**generator_args)

	# Create input tensor
	x = torch.randn(2, generator_args["generator_input_channels"], 100)

	# Run forward pass
	output = generator(x)

	# Check output shape
	assert isinstance(output, torch.Tensor)
	assert output.shape[0] == 2
	assert output.shape[1] == 1

	# Check that ResBlock2 is used in resblocks
	assert isinstance(generator.resblocks[0], ResBlock2)
