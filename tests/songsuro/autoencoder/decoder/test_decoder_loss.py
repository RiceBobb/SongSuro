import torch

from songsuro.autoencoder.decoder.decoder_loss import (
	feature_loss,
	discriminator_loss,
	generator_loss,
)


class TestLossFunctions:
	def test_feature_loss(self):
		# Create test data
		fmap_r = [
			[torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
			[torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
		]
		fmap_g = [
			[torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
			[torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
		]

		# Loss between identical feature maps should be 0
		identical_loss = feature_loss(fmap_r, fmap_r)
		assert identical_loss.item() == 0.0

		# In general cases, the loss should be positive
		loss = feature_loss(fmap_r, fmap_g)
		assert loss.item() > 0.0

		# Loss should be a scalar tensor
		assert loss.shape == torch.Size([])

		# Loss should match the formula
		# $L_{fm}(G) = \mathbb{E}[\sum_{l=1}^L \frac{1}{N_l} ||D_l(y) - D_l(G(z_q))||_1]$
		direct_loss = 0
		batch_size = fmap_r[0][0].shape[0]
		for dr, dg in zip(fmap_r, fmap_g):
			for rl, gl in zip(dr, dg):
				N = rl.shape[1] * rl.shape[2]
				direct_loss += torch.sum(torch.abs(rl - gl)) / N
		direct_loss /= batch_size

		assert torch.isclose(loss, direct_loss)

	def test_discriminator_loss(self):
		# Create test data
		disc_real_outputs = [torch.rand(2, 1) for _ in range(3)]
		disc_generated_outputs = [torch.rand(2, 1) for _ in range(3)]

		# Calculate loss
		loss, r_losses, g_losses = discriminator_loss(
			disc_real_outputs, disc_generated_outputs
		)

		# Check return values
		assert isinstance(loss, torch.Tensor)
		assert len(r_losses) == len(disc_real_outputs)
		assert len(g_losses) == len(disc_generated_outputs)

		# Ideal case: real = 1, fake = 0 => loss should be 0
		ideal_real = [torch.ones_like(dr) for dr in disc_real_outputs]
		ideal_gen = [torch.zeros_like(dg) for dg in disc_generated_outputs]
		ideal_loss, ideal_r_losses, ideal_g_losses = discriminator_loss(
			ideal_real, ideal_gen
		)
		assert torch.isclose(ideal_loss, torch.tensor(0.0), atol=1e-6)
		assert all(
			torch.isclose(torch.tensor(rl), torch.tensor(0.0), atol=1e-6)
			for rl in ideal_r_losses
		)
		assert all(
			torch.isclose(torch.tensor(gl), torch.tensor(0.0), atol=1e-6)
			for gl in ideal_g_losses
		)

		# Worst case: real = 0, fake = 1 => loss should be high
		worst_real = [torch.zeros_like(dr) for dr in disc_real_outputs]
		worst_gen = [torch.ones_like(dg) for dg in disc_generated_outputs]
		worst_loss, worst_r_losses, worst_g_losses = discriminator_loss(
			worst_real, worst_gen
		)
		assert worst_loss.item() > 0.0
		assert all(rl > 0.0 for rl in worst_r_losses)
		assert all(gl > 0.0 for gl in worst_g_losses)

	def test_generator_loss(self):
		# Create test data
		disc_outputs = [torch.rand(2, 1) for _ in range(3)]

		# Calculate loss
		loss, gen_losses = generator_loss(disc_outputs)

		# Check return values
		assert isinstance(loss, torch.Tensor)
		assert len(gen_losses) == len(disc_outputs)

		# Ideal case check (generated outputs = 1)
		ideal_outputs = [torch.ones_like(dg) for dg in disc_outputs]
		ideal_loss, ideal_gen_losses = generator_loss(ideal_outputs)
		assert ideal_loss.item() == 0.0
		assert all(gl == 0.0 for gl in ideal_gen_losses)

		# Worst case check (generated outputs = 0)
		worst_outputs = [torch.zeros_like(dg) for dg in disc_outputs]
		worst_loss, worst_gen_losses = generator_loss(worst_outputs)
		assert worst_loss.item() > 0.0
		assert all(gl > 0.0 for gl in worst_gen_losses)
