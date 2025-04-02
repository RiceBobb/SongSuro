import pytest
import torch
import torch.nn as nn
from songsuro.autoencoder.autoencoder import ResidualVectorQuantizer


class MockEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv1d(1, 256, kernel_size=3, padding=1)

	def forward(self, x):
		return self.conv(x)


class MockDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv1d(256, 1, kernel_size=3, padding=1)

	def forward(self, x):
		return self.conv(x)


class MockAutoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = MockEncoder()
		self.quantizer = ResidualVectorQuantizer()
		self.decoder = MockDecoder()

	def forward(self, x):
		encoded = self.encoder(x)
		quantized, commit_loss = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded, commit_loss


@pytest.fixture
def mock_model():
	return MockAutoencoder()


def test_autoencoder_output_shape(mock_model):
	x = torch.randn(2, 1, 128)
	output, commit_loss = mock_model(x)

	assert output.shape[0] == x.shape[0], "Batch dimension should match"
	assert output.shape[1] == x.shape[1], "Channel dimension should match"
	assert output.shape[2] == x.shape[2], "Time dimension should be preserved"


def test_commitment_loss(mock_model):
	x = torch.randn(2, 1, 128)
	_, commit_loss = mock_model(x)

	assert isinstance(
		commit_loss, torch.Tensor
	), "Commitment loss should be a torch.Tensor"
	assert commit_loss.numel() == 1, "Commitment loss should be a scalar"
	assert commit_loss.item() >= 0, "Commitment loss should be non-negative"


def test_model_components(mock_model):
	assert hasattr(mock_model, "encoder"), "Model should have an encoder"
	assert hasattr(mock_model, "quantizer"), "Model should have a quantizer"
	assert hasattr(mock_model, "decoder"), "Model should have a decoder"


def test_encoder_output_shape(mock_model):
	x = torch.randn(2, 1, 128)
	encoded = mock_model.encoder(x)

	assert encoded.shape[0] == 2, "Batch dimension should be preserved"
	assert encoded.shape[1] == 256, "Encoder should output 256 channels"
	assert encoded.shape[2] == 128, "Time dimension should be preserved"


def test_quantizer_functionality(mock_model):
	encoded = torch.randn(2, 256, 128)
	quantized, commit_loss = mock_model.quantizer(encoded)

	assert quantized.shape == encoded.shape, "Quantizer should preserve shape"
	assert isinstance(commit_loss, torch.Tensor), "Commitment loss should be a tensor"


def test_decoder_output_shape(mock_model):
	quantized = torch.randn(2, 256, 128)
	decoded = mock_model.decoder(quantized)

	assert decoded.shape[0] == 2, "Batch dimension should be preserved"
	assert decoded.shape[1] == 1, "Decoder should output 1 channel"
	assert decoded.shape[2] == 128, "Time dimension should be preserved"


def test_model_forward_pass(mock_model):
	x = torch.randn(2, 1, 128)
	try:
		output, commit_loss = mock_model(x)
	except Exception as e:
		pytest.fail(f"Forward pass failed with error: {str(e)}")
