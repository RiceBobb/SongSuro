import pytest
import torch
import torch.nn as nn
from torch.nn import Linear

from songsuro.diffusion.model import DiffusionEmbedding


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
