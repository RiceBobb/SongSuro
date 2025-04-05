import torch
import torch.nn as nn

from songsuro.autoencoder.decoder.discriminator import (
	DiscriminatorP,
	MultiPeriodDiscriminator,
	DiscriminatorS,
	MultiScaleDiscriminator,
)


class TestDiscriminatorP:
	def test_initialization(self):
		# Basic initialization test
		disc = DiscriminatorP(period=2)

		assert disc.period == 2
		assert len(disc.convs) == 5
		assert isinstance(disc.conv_post, nn.Conv2d)

		# Check if all layers are initialized with weight_norm
		for layer in disc.convs:
			assert hasattr(layer, "weight_v")
		assert hasattr(disc.conv_post, "weight_v")

	def test_spectral_norm(self):
		# Test using spectral_norm
		disc = DiscriminatorP(period=2, use_spectral_norm=True)

		# Check if all layers are initialized with spectral_norm
		for layer in disc.convs:
			assert hasattr(layer, "weight_orig")
		assert hasattr(disc.conv_post, "weight_orig")

	def test_forward(self):
		# Forward pass test
		disc = DiscriminatorP(period=2)

		# Create input tensor (batch size 3, channel 1, time 100)
		x = torch.randn(3, 1, 100)

		# Execute forward pass
		output, fmap = disc(x)

		# Check output type
		assert isinstance(output, torch.Tensor)
		assert len(fmap) == 6  # 5 convs + 1 conv_post

		# Check output shape
		assert output.shape[0] == 3  # batch size

		# Check feature maps
		for i, feat in enumerate(fmap):
			assert isinstance(feat, torch.Tensor)
			assert feat.shape[0] == 3  # batch size

	def test_padding(self):
		# Padding test (when input length is not divisible by period)
		disc = DiscriminatorP(period=3)

		# Create input tensor (batch size 2, channel 1, time 101)
		x = torch.randn(2, 1, 101)

		# Execute forward pass
		output, fmap = disc(x)

		# Check output type
		assert isinstance(output, torch.Tensor)
		assert output.shape[0] == 2  # batch size


class TestMultiPeriodDiscriminator:
	def test_initialization(self):
		# Initialization test
		mpd = MultiPeriodDiscriminator()

		assert len(mpd.discriminators) == 5
		for i, d in enumerate(mpd.discriminators):
			assert isinstance(d, DiscriminatorP)
			assert d.period in [2, 3, 5, 7, 11]

	def test_forward(self):
		# Forward pass test
		mpd = MultiPeriodDiscriminator()

		# Create input tensors (batch size 2, channel 1, time 200)
		y = torch.randn(2, 1, 200)
		y_hat = torch.randn(2, 1, 200)

		# Execute forward pass
		y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)

		# Check output shapes
		assert len(y_d_rs) == 5  # 5 discriminators
		assert len(y_d_gs) == 5
		assert len(fmap_rs) == 5
		assert len(fmap_gs) == 5

		# Check each discriminator's output
		for i in range(5):
			assert isinstance(y_d_rs[i], torch.Tensor)
			assert isinstance(y_d_gs[i], torch.Tensor)
			assert y_d_rs[i].shape[0] == 2  # batch size
			assert y_d_gs[i].shape[0] == 2  # batch size

			# Check feature maps
			assert len(fmap_rs[i]) == 6  # 5 convs + 1 conv_post
			assert len(fmap_gs[i]) == 6


class TestDiscriminatorS:
	def test_initialization(self):
		# Basic initialization test
		disc = DiscriminatorS()

		assert len(disc.convs) == 7
		assert isinstance(disc.conv_post, nn.Conv1d)

		# Check if all layers are initialized with weight_norm
		for layer in disc.convs:
			assert hasattr(layer, "weight_v")
		assert hasattr(disc.conv_post, "weight_v")

	def test_spectral_norm(self):
		# Test using spectral_norm
		disc = DiscriminatorS(use_spectral_norm=True)

		# Check if all layers are initialized with spectral_norm
		for layer in disc.convs:
			assert hasattr(layer, "weight_orig")
		assert hasattr(disc.conv_post, "weight_orig")

	def test_forward(self):
		# Forward pass test
		disc = DiscriminatorS()

		# Create input tensor (batch size 3, channel 1, time 100)
		x = torch.randn(3, 1, 100)

		# Execute forward pass
		output, fmap = disc(x)

		# Check output type
		assert isinstance(output, torch.Tensor)
		assert len(fmap) == 8  # 7 convs + 1 conv_post

		# Check output shape
		assert output.shape[0] == 3  # batch size

		# Check feature maps
		for i, feat in enumerate(fmap):
			assert isinstance(feat, torch.Tensor)
			assert feat.shape[0] == 3  # batch size


class TestMultiScaleDiscriminator:
	def test_initialization(self):
		# Initialization test
		msd = MultiScaleDiscriminator()

		assert len(msd.discriminators) == 3
		assert len(msd.meanpools) == 2

		# The first discriminator uses spectral_norm
		assert hasattr(msd.discriminators[0].convs[0], "weight_orig")

		# The others use weight_norm
		assert hasattr(msd.discriminators[1].convs[0], "weight_v")
		assert hasattr(msd.discriminators[2].convs[0], "weight_v")

	def test_forward(self):
		# Forward pass test
		msd = MultiScaleDiscriminator()

		# Create input tensors (batch size 2, channel 1, time 200)
		y = torch.randn(2, 1, 200)
		y_hat = torch.randn(2, 1, 200)

		# Execute forward pass
		y_d_rs, y_d_gs, fmap_rs, fmap_gs = msd(y, y_hat)

		# Check output shapes
		assert len(y_d_rs) == 3  # 3 discriminators
		assert len(y_d_gs) == 3
		assert len(fmap_rs) == 3
		assert len(fmap_gs) == 3

		# Check each discriminator's output
		for i in range(3):
			assert isinstance(y_d_rs[i], torch.Tensor)
			assert isinstance(y_d_gs[i], torch.Tensor)
			assert y_d_rs[i].shape[0] == 2  # batch size
			assert y_d_gs[i].shape[0] == 2  # batch size

			# Check feature maps
			assert len(fmap_rs[i]) == 8  # 7 convs + 1 conv_post
			assert len(fmap_gs[i]) == 8
