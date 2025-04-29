import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from songsuro.autoencoder.models import Autoencoder
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


@pytest.fixture
def ae_params():
	# Dictionary of hyperparameters to pass to the Autoencoder
	return dict(
		encoder_in_channels=1,
		encoder_out_channels=128,
		num_quantizers=8,
		codebook_size=32,
		codebook_dim=128,
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[8, 8, 2, 2],
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
		upsample_initial_channel=512,
		resblock="1",
	)


@pytest.fixture
def autoencoder(ae_params):
	return Autoencoder(**ae_params)


def test_initialization(ae_params, autoencoder):
	# Check if components are initialized correctly
	assert isinstance(autoencoder.encoder, Encoder)
	assert isinstance(autoencoder.quantizer, ResidualVectorQuantizer)
	assert isinstance(autoencoder.decoder, Generator)
	# Check if init_args are stored correctly
	for k, v in ae_params.items():
		assert autoencoder.init_args[k] == v


def test_forward_with_mocks(ae_params):
	original_init = Autoencoder.__init__
	mock_outputs = {}

	# Mock class inheriting from nn.Module
	class ModuleMock(nn.Module):
		def __init__(self, return_value=None):
			super().__init__()
			self.mock = MagicMock()
			self.return_value = return_value

		def forward(self, *args, **kwargs):
			self.mock(*args, **kwargs)
			return self.return_value

	def mock_init(self, **kwargs):
		# 1. Call the original nn.Module initialization
		nn.Module.__init__(self)
		self.init_args = kwargs

		# 2. Create module-compatible mocks
		mock_outputs["encoded_output"] = torch.randn(2, 128, 50)
		self.encoder = ModuleMock(return_value=mock_outputs["encoded_output"])

		mock_outputs["quantized_output"] = torch.randn(2, 128, 50)
		mock_outputs["commit_loss"] = torch.tensor(0.1)
		self.quantizer = ModuleMock(
			return_value=(
				mock_outputs["quantized_output"],
				mock_outputs["commit_loss"],
			)
		)

		mock_outputs["decoded_output"] = torch.randn(2, 1, 100)
		self.decoder = ModuleMock(return_value=mock_outputs["decoded_output"])

	try:
		Autoencoder.__init__ = mock_init
		autoencoder = Autoencoder(**ae_params)

		x = torch.randn(2, 1, 100)
		output, loss = autoencoder(x)  # Should work now

		# Verify calls
		autoencoder.encoder.mock.assert_called_once_with(x)
		autoencoder.quantizer.mock.assert_called_once_with(
			mock_outputs["encoded_output"]
		)
		autoencoder.decoder.mock.assert_called_once_with(
			mock_outputs["quantized_output"]
		)

	finally:
		Autoencoder.__init__ = original_init


def test_with_real_components(autoencoder):
	# Create a small input tensor
	x = torch.randn(2, 1, 32)
	output, loss = autoencoder(x)
	assert isinstance(output, torch.Tensor)
	assert isinstance(loss, torch.Tensor)
	assert output.shape[0] == 2  # Batch size
	# Output channel depends on Generator structure (e.g., 1)
	# Output length (after upsampling) should be greater than 32
	assert output.shape[2] > 32


def test_remove_weight_norm(autoencoder, monkeypatch):
	# Mock the decoder's remove_weight_norm method
	mock_remove_weight_norm = MagicMock()
	monkeypatch.setattr(
		autoencoder.decoder, "remove_weight_norm", mock_remove_weight_norm
	)
	autoencoder.remove_weight_norm()
	mock_remove_weight_norm.assert_called_once()
