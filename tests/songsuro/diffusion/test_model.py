import pytest
import torch
import torch.nn as nn
from torch.nn import Linear

from songsuro.diffusion.model import DiffusionEmbedding, ResidualBlock


class TestDiffusionEmbedding:
	@pytest.fixture
	def embedding_model(self):
		max_steps = 100
		return DiffusionEmbedding(max_steps)

	def test_initialization(self, embedding_model):
		# Test that the model initializes correctly
		assert isinstance(embedding_model, nn.Module)
		assert hasattr(embedding_model, "embedding")
		assert embedding_model.embedding.shape == (100, 128)  # max_steps x (64*2)
		assert isinstance(embedding_model.projection1, Linear)
		assert isinstance(embedding_model.projection2, Linear)

	def test_build_embedding(self, embedding_model):
		# Test the embedding building function
		max_steps = 50
		embedding = embedding_model._build_embedding(max_steps)
		assert embedding.shape == (max_steps, 128)
		assert torch.isfinite(embedding).all()  # Check for NaN or inf values

	def test_forward_int_input(self, embedding_model):
		# Test forward pass with integer input
		output = embedding_model(torch.tensor([10]))
		assert output.shape == torch.Size([1, 512])
		assert torch.isfinite(output).all()

		output = embedding_model(torch.tensor([10, 4, 5, 8]))
		assert output.shape == torch.Size([4, 512])
		assert torch.isfinite(output).all()

	def test_forward_float_input(self, embedding_model):
		# Test forward pass with float input
		output = embedding_model(torch.tensor([0.02]))
		assert output.shape == torch.Size([1, 512])
		assert torch.isfinite(output).all()

		output = embedding_model(torch.tensor([0.1, 2.5, 10.9, 3.78]))
		assert output.shape == torch.Size([4, 512])
		assert torch.isfinite(output).all()

	def test_lerp_embedding(self, embedding_model):
		# Test linear interpolation function
		t = torch.Tensor([2.7])
		interpolated = embedding_model._lerp_embedding(t)
		assert interpolated.shape == (1, 128)


@pytest.fixture
def residual_block():
	input_condition_dim = 128
	channel_size = 256
	kernel_size = 3
	dilation = 2
	return ResidualBlock(
		input_condition_dim=input_condition_dim,
		channel_size=channel_size,
		kernel_size=kernel_size,
		dilation=dilation,
	)


def test_residual_block_initialization(residual_block):
	"""Test if the ResidualBlock is initialized correctly."""
	assert residual_block.channel_size == 256
	assert isinstance(residual_block.conditioner_projection, nn.Linear)
	assert isinstance(residual_block.dilated_conv, nn.Conv1d)
	assert isinstance(residual_block.output_projection, nn.Conv1d)

	# Check dimensions of layers
	assert residual_block.conditioner_projection.in_features == 128
	assert residual_block.conditioner_projection.out_features == 512  # channel_size * 2
	assert residual_block.dilated_conv.in_channels == 256
	assert residual_block.dilated_conv.out_channels == 512  # channel_size * 2
	assert residual_block.dilated_conv.kernel_size == (3,)
	assert residual_block.dilated_conv.dilation == (2,)


def test_residual_block_forward():
	"""Test the forward pass of the ResidualBlock."""
	# Initialize the block with specific parameters
	block = ResidualBlock(
		input_condition_dim=128, channel_size=32, kernel_size=3, dilation=1
	)

	# Create dummy input tensors
	batch_size = 4
	seq_length = 10

	x = torch.randn(batch_size, 32, seq_length)  # (B, C, L)
	diffusion_step_embedding = torch.randn(batch_size, 32)  # (B, C)
	condition_embedding = torch.randn(
		batch_size, 128, seq_length
	)  # (B, input_condition_dim, L)

	# Forward pass
	residual, skip = block(x, diffusion_step_embedding, condition_embedding)

	# Check output shapes
	assert residual.shape == (batch_size, 32, seq_length)
	assert skip.shape == (batch_size, 32, seq_length)


def test_residual_block_with_different_dimensions():
	"""Test the ResidualBlock with different input dimensions."""
	# Test with different channel sizes and sequence lengths
	for channel_size in [16, 32, 64]:
		for seq_length in [8, 16, 32]:
			block = ResidualBlock(
				input_condition_dim=48,
				channel_size=channel_size,
				kernel_size=3,
				dilation=1,
			)

			batch_size = 2
			x = torch.randn(batch_size, channel_size, seq_length)
			diffusion_step_embedding = torch.randn(batch_size, channel_size)
			condition_embedding = torch.randn(batch_size, 48, seq_length)

			residual, skip = block(x, diffusion_step_embedding, condition_embedding)

			assert residual.shape == (batch_size, channel_size, seq_length)
			assert skip.shape == (batch_size, channel_size, seq_length)


def test_residual_block_gradient_flow():
	"""Test if gradients flow through the ResidualBlock."""
	block = ResidualBlock(
		input_condition_dim=32, channel_size=16, kernel_size=3, dilation=1
	)

	batch_size = 2
	seq_length = 8

	x = torch.randn(batch_size, 16, seq_length, requires_grad=True)
	diffusion_step_embedding = torch.randn(batch_size, 16, requires_grad=True)
	condition_embedding = torch.randn(batch_size, 32, seq_length, requires_grad=True)

	residual, skip = block(x, diffusion_step_embedding, condition_embedding)

	# Compute a loss and check if gradients flow
	loss = (residual.sum() + skip.sum()) / 2
	loss.backward()

	assert x.grad is not None
	assert diffusion_step_embedding.grad is not None
	assert condition_embedding.grad is not None


def test_residual_block_normalization():
	"""Test if the normalization by sqrt(2.0) is applied correctly."""
	block = ResidualBlock(
		input_condition_dim=32, channel_size=16, kernel_size=3, dilation=1
	)

	# Create inputs where we can easily track the normalization
	batch_size = 1
	seq_length = 4

	# Create constant inputs for easier verification
	x = torch.ones(batch_size, 16, seq_length)
	diffusion_step_embedding = torch.zeros(
		batch_size, 16
	)  # Zero to simplify calculation

	# Create a condition embedding that will result in a simple output after processing
	# This is a simplified test case where we assume the gate will be close to 1 and filter close to input
	condition_embedding = (
		torch.ones(batch_size, 32, seq_length) * 5
	)  # Large positive value to make sigmoid(gate) close to 1

	with torch.no_grad():
		# Forward pass
		residual, _ = block(x, diffusion_step_embedding, condition_embedding)

		# Get the mean of the output
		mean_output = residual.mean().item()

		# The scale should be roughly 1/sqrt(2) times the input if the residual is similar to y
		# This is a very rough check and might fail depending on initialization
		assert 0.5 < mean_output < 2.0, "Output scale seems off"
