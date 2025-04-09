from typing import Optional

import numpy as np
import torch
from torch import nn

from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.encoder.vconv import VirtualConv
from songsuro.condition.model import ConditionalEncoder
from songsuro.diffusion.model import Denoiser


class Songsuro(nn.Module):
	def __init__(
		self,
		n_mels,
		latent_dim,
		condition_dim,
		device,
		noise_schedule=np.linspace(1e-4, 0.05, 50),
		*args,
		**kwargs,
	):
		super().__init__()
		self.device = device
		parent_vc = VirtualConv(filter_info=3, stride=1, name="parent")
		self.encoder = Encoder(n_mels, latent_dim, parent_vc)
		self.max_step_size = len(noise_schedule)
		self.noise_schedule = noise_schedule
		self.noise_level = torch.Tensor(
			np.cumprod(1 - noise_schedule).astype(np.float32)
		).to(device)

		self.denoiser = Denoiser(
			self.max_step_size,
			channel_size=latent_dim,
			condition_embedding_dim=condition_dim,
		)
		self.conditional_encoder = ConditionalEncoder(
			hidden_size=condition_dim,
			prior_output_dim=latent_dim,
		)
		self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get("fp16", False))

	def forward(self, gt_spectrogram, lyrics, step_idx: Optional[int] = None):
		"""
		Forward process (in training) of the Songsuro model.
		This is only one step training

		:return:
		"""
		for param in self.denoiser.parameters():
			param.grad = None

		for param in self.conditional_encoder.parameters():
			param.grad = None

		with torch.no_grad():
			latent = self.encoder(gt_spectrogram)

		if step_idx is None:
			step_idx = torch.randint(0, self.max_step_size, [1], device=self.device)

		with self.autocast:
			condition_embedding, prior = self.conditional_encoder(
				lyrics, gt_spectrogram
			)
			# TODO: Implement prior loss

			noise_scale = self.noise_level[step_idx].unsqueeze(1)
			noise_scale_sqrt = noise_scale**0.5

			noise = prior + torch.randn_like(
				latent, device=self.device
			)  # epsilon sampling from N(\mu, I)
			noisy_latent = (
				noise_scale_sqrt * latent + (1.0 - noise_scale) ** 0.5 * noise
			)  # x_t

			predicted = self.denoiser(noisy_latent, step_idx, condition_embedding)
			diff_loss = nn.MSELoss()(noise, predicted.squeeze(1))

		return diff_loss

	# TODO: Contrastive loss must be implemented in train.py

	def sample(self, gt_spectrogram, lyrics):
		"""
		Inference of the Songsuro model.

		:return: The generated waveform.
		"""
		with torch.no_grad():
			_ = self.encoder(gt_spectrogram)
