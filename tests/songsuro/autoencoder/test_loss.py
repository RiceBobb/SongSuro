import torch

from songsuro.autoencoder.loss import reconstruction_loss


def test_reconstruction_loss():
	# Create dummy data
	original = torch.randn(
		10, 3, 32, 32
	)  # Example shape (batch_size, channels, height, width)
	reconstructed = torch.randn(10, 3, 32, 32)

	# Calculate loss
	loss = reconstruction_loss(original, reconstructed)

	# Check if loss is a tensor
	assert isinstance(loss, torch.Tensor)
	assert loss.item() >= 0  # Loss should be non-negative
	assert loss.shape == torch.Size([])  # Loss should be a scalar
	assert loss.item() != 0  # Loss should not be zero for random data
	assert (
		loss.item() == torch.mean(torch.abs(original - reconstructed)).item()
	)  # Check if it matches the expected value
