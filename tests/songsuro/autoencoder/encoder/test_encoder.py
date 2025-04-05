import torch

from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.encoder.vconv import VirtualConv


class TestEncoder:
	def test_initialization(self):
		# Create parent virtual conv object
		parent_vc = VirtualConv(filter_info=3, stride=1, name="parent")

		# Test initialization of the Encoder module
		encoder = Encoder(n_in=1, n_out=128, parent_vc=parent_vc)

		assert len(encoder.net) == 9  # 9 layers
		assert isinstance(encoder.vc, dict)
		assert "beg" in encoder.vc
		assert "end" in encoder.vc

		# Check VirtualConv connection structure
		assert encoder.vc["beg"].parent is parent_vc
		assert parent_vc.child is encoder.vc["beg"]

	def test_forward(self):
		# Create parent virtual conv object
		parent_vc = VirtualConv(filter_info=3, stride=1, name="parent")

		# Initialize the Encoder module
		encoder = Encoder(n_in=1, n_out=128, parent_vc=parent_vc)

		# Create input tensor (batch size 2, channel 1, time 100)
		x = torch.randn(2, 1, 100)

		# Run forward pass
		output = encoder(x)

		# Check that output is a tensor
		assert isinstance(output, torch.Tensor)

		# Check that output channel matches n_out
		assert output.shape[1] == 128

		# Check that time dimension is reduced due to downsampling
		assert output.shape[2] < 100

	def test_set_parent_vc(self):
		# Create parent virtual conv object
		parent_vc = VirtualConv(filter_info=3, stride=1, name="parent")

		# Initialize the Encoder module with no parent
		encoder = Encoder(n_in=1, n_out=128, parent_vc=None)

		# Set parent virtual conv
		encoder.set_parent_vc(parent_vc)

		# Check parent-child relationship
		assert encoder.vc["beg"].parent is parent_vc
		assert parent_vc.child is encoder.vc["beg"]

	def test_metrics_update(self):
		# Create parent virtual conv object
		parent_vc = VirtualConv(filter_info=3, stride=1, name="parent")

		# Initialize the Encoder module
		encoder = Encoder(n_in=1, n_out=128, parent_vc=parent_vc)

		# Create input tensor
		x = torch.randn(2, 1, 100)

		# Run forward pass
		_ = encoder(x)

		# Check if metrics attribute exists
		assert hasattr(encoder, "metrics")

		# Check that each layer has a 'frac_zero_act' metric
		for i in range(9):
			assert f"enc_az_{i}" in encoder.metrics
			assert (
				0 <= encoder.metrics[f"enc_az_{i}"] <= 1
			)  # Should be a ratio between 0 and 1
