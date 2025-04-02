import pytest
import torch
from songsuro.autoencoder.encoder.encoder import WaveNetEncoder


@pytest.fixture
def encoder():
	return WaveNetEncoder(
		batch_size=2,
		dilations=[1, 2, 4, 8],
		filter_width=3,
		residual_channels=32,
		dilation_channels=32,
		skip_channels=64,
		latent_channels=128,
	)


def test_encoder_output_shape(encoder):
	input_data = torch.randn(2, 256, 128)  # [batch_size, channels, time_steps]

	output = encoder(input_data)

	assert output.shape == (
		2,
		256,
		128,
	), f"Expected shape (2, 1, 128), but got {output.shape}"


def test_encoder_loss(encoder):
	input_data = torch.randn(2, 1, 1000)
	target_data = torch.randn(2, 1, 1000)

	# Calculate loss
	loss = encoder.loss(input_data, target_data)

	assert loss.dim() == 0, f"Expected scalar loss, but got shape {loss.shape}"
	assert loss.item() > 0, f"Expected positive loss, but got {loss.item()}"


def test_encoder_l2_regularization(encoder):
	input_data = torch.randn(2, 1, 1000)

	loss_without_l2 = encoder.loss(input_data)

	loss_with_l2 = encoder.loss(input_data, l2_regularization_strength=0.01)

	assert (
		loss_with_l2.item() > loss_without_l2.item()
	), "L2 regularization should increase the loss"


def test_encoder_non_causal_convolution(encoder):
	input_data = torch.zeros(2, 1, 1000)
	input_data[:, :, 500] = 1.0  # 중앙에 임펄스 신호

	output = encoder(input_data)

	assert torch.any(
		output[:, :, :500] != 0
	), "Non-causal convolution should affect earlier time steps"
