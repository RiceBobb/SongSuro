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
		# Create a random input tensor with shape (batch_size, embedding_dim, length)
		x = torch.randn(batch_size, condition_embedding_dim)
		# Forward pass
		output = prior_estimator(x)
		# Check output shape
		assert output.shape == (batch_size, output_dim)
