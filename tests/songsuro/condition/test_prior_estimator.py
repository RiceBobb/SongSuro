import pytest
import torch
from torch import nn

# Import the PriorEstimator class or include it directly in your test file
from songsuro.condition.prior_estimator import PriorEstimator

condition_embedding_dim = 128
output_dim = 64


class TestPriorEstimator:
	@pytest.fixture
	def prior_estimator(self):
		model = PriorEstimator(condition_embedding_dim, output_dim)
		return model

	def test_initialization(self, prior_estimator):
		"""Test that the PriorEstimator initializes correctly."""
		# Check that the layer is a Linear layer
		assert isinstance(prior_estimator.layer, nn.Linear)
		# Check that the layer has the correct dimensions
		assert prior_estimator.layer.in_features == condition_embedding_dim
		assert prior_estimator.layer.out_features == output_dim

	def test_forward_pass(self, prior_estimator):
		"""Test the forward pass of the PriorEstimator."""
		batch_size = 8
		sequence_length = 16

		# Create a random input tensor with shape (batch_size, embedding_dim, length)
		x = torch.randn(batch_size, condition_embedding_dim, sequence_length)
		# Forward pass
		output = prior_estimator(x)
		# Check output shape
		assert output.shape == (batch_size, output_dim, sequence_length)

	def test_transpose_operations(self, prior_estimator):
		"""Test that the transpose operations work correctly."""
		batch_size = 4
		sequence_length = 10

		# Set weights to a fixed value for deterministic testing
		with torch.no_grad():
			prior_estimator.layer.weight.fill_(0.1)
			prior_estimator.layer.bias.fill_(0.0)

		# Create an input with recognizable values
		x = torch.ones(batch_size, condition_embedding_dim, sequence_length)

		# Forward pass
		output = prior_estimator(x)

		# Expected output after transpose, linear layer, and transpose back
		# Linear layer with weight 0.1 and input of ones should give:
		# 0condition_embedding_dim for each element (plus bias of 0)
		expected_value = 0.1 * condition_embedding_dim

		# Check if all values in the output are close to the expected value
		assert torch.allclose(
			output, torch.full_like(output, expected_value), rtol=1e-5
		)
