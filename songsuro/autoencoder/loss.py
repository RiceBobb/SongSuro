import torch.nn as nn


def reconstruction_loss(original, reconstructed):
	recon_loss_fn = nn.L1Loss()
	return recon_loss_fn(reconstructed, original)
