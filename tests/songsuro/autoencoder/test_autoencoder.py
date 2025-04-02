import pytest
import torch
from songsuro.autoencoder.autoencoder import (
	ResidualVectorQuantizer,
	WaveNetEncoder,
	HiFiGANDecoder,
	Autoencoder,
)


@pytest.fixture
def mock_autoencoder():
	encoder = WaveNetEncoder()
	quantizer = ResidualVectorQuantizer()
	decoder = HiFiGANDecoder()
	return Autoencoder(encoder, quantizer, decoder)


def test_autoencoder_output_shape(mock_autoencoder):
	x = torch.randn(2, 1, 128)
	output, commit_loss = mock_autoencoder(x)

	assert output.shape[0] == x.shape[0], "Batch dimension should match"
	assert output.shape[1] == x.shape[1], "Channel dimension should match"
	assert output.shape[2] == x.shape[2], "Time dimension should be preserved"


def test_commitment_loss(mock_autoencoder):
	x = torch.randn(2, 1, 128)
	_, commit_loss = mock_autoencoder(x)

	assert isinstance(
		commit_loss, torch.Tensor
	), "Commitment loss should be a torch.Tensor"
	assert commit_loss.numel() == 1, "Commitment loss should be a scalar"
	assert commit_loss.item() >= 0, "Commitment loss should be non-negative"


def test_model_components(mock_autoencoder):
	assert hasattr(mock_autoencoder, "encoder"), "Model should have an encoder"
	assert hasattr(mock_autoencoder, "quantizer"), "Model should have a quantizer"
	assert hasattr(mock_autoencoder, "decoder"), "Model should have a decoder"


def test_encoder_output_shape(mock_autoencoder):
	x = torch.randn(2, 1, 128)
	encoded = mock_autoencoder.encoder(x)

	assert encoded.shape[0] == 2, "Batch dimension should be preserved"
	assert encoded.shape[1] == 256, "Encoder should output 256 channels"
	assert encoded.shape[2] == 128, "Time dimension should be preserved"


def test_quantizer_functionality(mock_autoencoder):
	encoded = torch.randn(2, 256, 128)
	quantized, commit_loss = mock_autoencoder.quantizer(encoded)

	assert quantized.shape == encoded.shape, "Quantizer should preserve shape"
	assert isinstance(commit_loss, torch.Tensor), "Commitment loss should be a tensor"


def test_decoder_output_shape(mock_autoencoder):
	quantized = torch.randn(2, 256, 128)
	decoded = mock_autoencoder.decoder(quantized)

	assert decoded.shape[0] == 2, "Batch dimension should be preserved"
	assert decoded.shape[1] == 1, "Decoder should output 1 channel"
	assert decoded.shape[2] == 128, "Time dimension should be preserved"


def test_model_forward_pass(mock_autoencoder):
	x = torch.randn(2, 1, 128)
	try:
		output, commit_loss = mock_autoencoder(x)
	except Exception as e:
		pytest.fail(f"Forward pass failed with error: {str(e)}")
