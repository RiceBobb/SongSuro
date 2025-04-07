import torch


def feature_loss(fmap_r, fmap_g):
	"""
	:param fmap_r: reference(gt) feature maps
	:param fmap_g: generated feature maps
	:return: feature matching loss
	"""
	loss = 0
	batch_size = fmap_r[0][0].shape[0]
	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):  # rl, gl: [B, C, W]
			N = rl.shape[1] * rl.shape[2]  # C * W
			loss += torch.sum(torch.abs(rl - gl)) / N
	return loss / batch_size


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []
	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		# E[(D(y) - 1)^2]
		r_loss = torch.mean((dr - 1) ** 2)
		# E[(D(G(z_q)))^2]
		g_loss = torch.mean(dg**2)
		# Due to E[A + B] = E[A] + E[B]
		loss += r_loss + g_loss
		r_losses.append(r_loss.item())  # real loss per layer
		g_losses.append(g_loss.item())  # fake loss per layer

	return loss, r_losses, g_losses


def generator_loss(disc_generated_outputs):
	loss = 0
	g_losses = []
	for dg in disc_generated_outputs:
		# Computes E[(D(G(z_q)) - 1)^2] using mean over all elements
		g_loss = torch.mean((dg - 1) ** 2)
		loss += g_loss
		g_losses.append(g_loss.item())  # Store loss per layer as float

	return loss, g_losses
