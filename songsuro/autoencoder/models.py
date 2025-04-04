import torch.nn as nn


class Autoencoder(nn.Module):
	def __init__(self, encoder, quantizer, decoder):
		super(Autoencoder, self).__init__()
		self.encoder = encoder
		self.quantizer = quantizer
		self.decoder = decoder

	def forward(self, x):
		encoded = self.encoder(x)
		quantized, commit_loss = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded, commit_loss
