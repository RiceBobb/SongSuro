import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from songsuro.autoencoder.models import Autoencoder
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class TestAutoencoder:
	@pytest.fixture
	def hps_config(self):
		# Create a mock object for hyperparameters
		hps = MagicMock()
		hps.model = MagicMock()

		# Encoder settings
		hps.model.encoder_in_channels = 1
		hps.model.encoder_out_channels = 128

		# Quantizer settings
		hps.model.num_quantizers = 8
		hps.model.codebook_size = 32
		hps.model.codebook_dim = 128

		# Generator settings
		hps.model.resblock_kernel_sizes = [3, 7, 11]
		hps.model.upsample_rates = [8, 8, 2, 2]
		hps.model.upsample_kernel_sizes = [16, 16, 4, 4]
		hps.model.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
		hps.model.upsample_initial_channel = 512
		hps.model.resblock = "1"
		hps.model.generator_input_channels = 128  # 디코더 입력 채널을 128로 설정

		return hps

	@pytest.fixture
	def autoencoder(self, hps_config):
		return Autoencoder(hps_config)

	def test_initialization(self, hps_config, autoencoder):
		# Check whether components are initialized correctly
		assert isinstance(autoencoder.encoder, Encoder)
		assert isinstance(autoencoder.quantizer, ResidualVectorQuantizer)
		assert isinstance(autoencoder.decoder, Generator)

		# Check if the hyperparameters are stored
		assert autoencoder.hps is hps_config

	def test_forward_with_mocks(self, hps_config):
		original_init = Autoencoder.__init__
		mock_outputs = {}

		# Create a custom mock class that inherits from nn.Module
		class ModuleMock(nn.Module):
			def __init__(self, return_value=None):
				super().__init__()
				self.mock = MagicMock()
				self.return_value = return_value

			def forward(self, *args, **kwargs):
				self.mock(*args, **kwargs)
				return self.return_value

		def mock_init(self, hps):
			# 1. Call original Module initialization
			original_init(self, hps)

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
			autoencoder = Autoencoder(hps_config)

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

	def test_with_real_components(self, autoencoder):
		# Create input tensor (small size for test)
		x = torch.randn(2, 1, 32)

		# Run forward pass
		output, loss = autoencoder(x)

		# Validate outputs
		assert isinstance(output, torch.Tensor)
		assert isinstance(loss, torch.Tensor)
		assert output.shape[0] == 2  # Batch size
		assert output.shape[1] == 1  # Output channels

		# Check output length (should increase due to upsampling)
		assert output.shape[2] > 32

	def test_remove_weight_norm(self, autoencoder, monkeypatch):
		# Mock the decoder's remove_weight_norm method
		mock_remove_weight_norm = MagicMock()
		monkeypatch.setattr(
			autoencoder.decoder, "remove_weight_norm", mock_remove_weight_norm
		)

		# Call the method
		autoencoder.remove_weight_norm()

		# Verify that the decoder's method was called
		mock_remove_weight_norm.assert_called_once()
