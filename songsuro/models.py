import numpy as np
import torch
import torchaudio
from torch import nn
import pytorch_lightning as pl

from songsuro.condition.model import ConditionalEncoder
from songsuro.diffusion.model import Denoiser
from songsuro.autoencoder.models import Autoencoder


class Songsuro(pl.LightningModule):
	def __init__(
		self,
		latent_dim,
		condition_dim,
		autoencoder_checkpoint_path: str,
		noise_schedule=np.linspace(1e-4, 0.05, 50),
		optimizer_betas=(0.8, 0.99),
		prior_lambda: float = 0.8,
		contrastive_lambda: float = 1.0,
	):
		super().__init__()
		self.autoencoder = Autoencoder()
		self.autoencoder.load_state_dict(
			torch.load(autoencoder_checkpoint_path, weights_only=True)
		)
		self.autoencoder.eval()

		self.max_step_size = len(noise_schedule)
		self.noise_schedule = noise_schedule  # beta_t
		self.noise_level = torch.Tensor(
			np.cumprod(1 - noise_schedule).astype(np.float32)
		)  # alpha_t_bar
		self.optimizer_betas = optimizer_betas
		self.prior_lambda = prior_lambda
		self.contrastive_lambda = contrastive_lambda

		self.denoiser = Denoiser(
			self.max_step_size,
			channel_size=latent_dim,
			condition_embedding_dim=condition_dim,
		)
		self.conditional_encoder = ConditionalEncoder(
			hidden_size=condition_dim,
			prior_output_dim=latent_dim,
		)

	def training_step(self, batch, batch_idx):
		lyrics = batch["lyrics"]
		gt_spectrogram = batch["mel_spectrogram"]

		for param in self.denoiser.parameters():
			param.grad = None

		for param in self.conditional_encoder.parameters():
			param.grad = None

		with torch.no_grad():  # frozen
			latent = self.autoencoder.encode(gt_spectrogram)

		step_idx = torch.randint(0, self.max_step_size, [1])

		condition_embedding, prior = self.conditional_encoder(lyrics, gt_spectrogram)
		# condition embedding shape : [B, Condition Dimension, Length]
		# prior shape : [B, latent_dim]
		prior_loss = nn.CrossEntropyLoss()(torch.mean(latent, dim=-1), prior)

		noise_scale = self.noise_level[step_idx].unsqueeze(1).type_as(gt_spectrogram)
		noise_scale_sqrt = noise_scale**0.5

		noise = prior.unsqueeze(-1) + torch.randn_like(
			latent
		)  # epsilon sampling from N(\mu, I)
		noisy_latent = (
			noise_scale_sqrt * latent + (1.0 - noise_scale) ** 0.5 * noise
		)  # x_t

		predicted = self.denoiser(noisy_latent, step_idx, condition_embedding)
		diff_loss = nn.MSELoss()(
			noise, predicted.squeeze(1)
		)  # diffusion loss (denoise loss)

		final_loss = diff_loss + (self.prior_lambda * prior_loss)
		self.log("train_loss", final_loss, on_step=True, prog_bar=True, logger=True)
		return final_loss

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			list(self.denoiser.parameters())
			+ list(self.conditional_encoder.parameters()),
			lr=2 * 1e-4,
			betas=self.optimizer_betas,
			weight_decay=0.01,
		)
		return [optimizer]

	def validation_step(self, batch, batch_idx):
		gt_spectrogram = batch["mel_spectrogram"]
		lyrics = batch["lyrics"]
		with torch.no_grad():
			pred_result = self.sample(gt_spectrogram, lyrics)
			pred_result = pred_result.squeeze(1)  # [B, L]

		# TODO: sample rates can be different among audio samples
		mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=batch["sample_rates"][0],
			n_fft=2048,
			hop_length=1024,
			f_max=8000,
		)
		pred_mel_spectrogram = mel_spectrogram_transform(
			pred_result.cpu()
		)  # Transfrom only support cpu type tensor

		spectrogram_mae = nn.L1Loss(reduction="mean")(
			pred_mel_spectrogram[..., : gt_spectrogram.shape[-1]].type_as(
				gt_spectrogram
			),
			gt_spectrogram,
		)
		self.log("val_loss", spectrogram_mae, on_step=True, prog_bar=True, logger=True)

		return {
			"loss": spectrogram_mae,
			"pred": pred_result,
		}

	def test_step(self, batch, batch_idx):
		return self.validation_step(batch, batch_idx)

	def forward(self, batch):
		gt_spectrogram = batch["mel_spectrogram"]
		lyrics = batch["lyrics"]
		with torch.no_grad():
			pred_result = self.sample(gt_spectrogram, lyrics)
			pred_result = pred_result.squeeze(1)

		return {
			"pred": pred_result,
		}

	def denoise(self, latent, condition_embedding, prior):
		x = prior.unsqueeze(-1) + torch.randn_like(latent)  # x_T # start with prior

		# Reverse process
		for step in range(len(self.noise_schedule) - 1, -1, -1):
			if step > 0:
				z = torch.randn_like(latent)
				sigma = (
					(1.0 - self.noise_level[step - 1])
					/ (1.0 - self.noise_level[step])
					* self.noise_schedule[step]
				) ** 0.5
			else:
				z = torch.zeros_like(latent)
				sigma = 0

			c1 = 1 / ((1 - self.noise_schedule[step]) ** 0.5)
			c2 = self.noise_schedule[step] / ((1 - self.noise_level[step]) ** 0.5)
			x = c1 * (x - c2 * self.denoiser(x, step, condition_embedding)) + z * sigma
			x = torch.clamp(x, -1.0, 1.0)

		return x

	def sample(self, gt_spectrogram, lyrics):
		"""
		Inference of the Songsuro model.

		:return: The generated waveform.
		"""
		with torch.no_grad():
			latent = self.autoencoder.encode(gt_spectrogram)
			# The latent is from the autoencoder encoder
			condition_embedding, prior = self.conditional_encoder(
				lyrics, gt_spectrogram
			)
			x = self.denoise(latent, condition_embedding, prior)
			# Now x is generated latent
			decoded_result = self.autoencoder.decode(x)
			return decoded_result
