import unittest
import torch

from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class TestResidualVectorQuantizer(unittest.TestCase):
	def test_initialization(self):
		rvq = ResidualVectorQuantizer(
			input_dim=128,
			num_quantizers=8,  # Use smaller value for testing
			codebook_size=32,  # Use smaller value for testing
			codebook_dim=128,
		)

		self.assertEqual(rvq.rvq.num_quantizers, 8)
		self.assertEqual(rvq.rvq.codebook_size, 32)

	def test_forward(self):
		rvq = ResidualVectorQuantizer(
			input_dim=128, num_quantizers=8, codebook_size=32, codebook_dim=128
		)

		# Create input tensor [batch, channels, seq_len]
		x = torch.randn(2, 128, 10)

		# Run forward pass
		quantized, commit_loss = rvq(x)

		# Check output shape
		self.assertEqual(quantized.shape, x.shape)
		self.assertTrue(isinstance(commit_loss, torch.Tensor))
		self.assertEqual(commit_loss.dim(), 0)  # Scalar tensor

	def test_with_encoder_output(self):
		# Create tensor that simulates actual encoder output
		encoder_output = torch.randn(2, 128, 50)  # [batch, channels, seq_len]

		# Initialize RVQ
		rvq = ResidualVectorQuantizer(
			input_dim=128, num_quantizers=8, codebook_size=32, codebook_dim=128
		)

		# Run forward pass
		quantized, commit_loss = rvq(encoder_output)

		# Check output shape
		self.assertEqual(quantized.shape, encoder_output.shape)
		self.assertTrue(isinstance(commit_loss, torch.Tensor))
		self.assertEqual(commit_loss.dim(), 0)  # Scalar tensor

	def test_quantization_effect(self):
		# Initialize RVQ
		rvq = ResidualVectorQuantizer(
			input_dim=128, num_quantizers=8, codebook_size=32, codebook_dim=128
		)

		# Create input tensor
		x = torch.randn(2, 128, 10)

		# Run forward pass
		quantized, _ = rvq(x)

		# Check that quantized output is different from input due to quantization
		self.assertFalse(torch.allclose(x, quantized))

	def test_encoder_rvq_integration(self):
		# Simulate encoder output
		encoder_output = torch.randn(2, 128, 50)

		# Initialize RVQ
		rvq = ResidualVectorQuantizer(
			input_dim=128, num_quantizers=8, codebook_size=32, codebook_dim=128
		)

		# Run forward pass
		quantized, commit_loss = rvq(encoder_output)

		# Check output shape
		self.assertEqual(quantized.shape, encoder_output.shape)
