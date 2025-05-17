import torch
import pytorch_lightning as pl

from songsuro.autoencoder.decoder.decoder_loss import (
	discriminator_loss,
	generator_loss,
	feature_loss,
)
from songsuro.autoencoder.decoder.discriminator import (
	MultiPeriodDiscriminator,
	MultiScaleDiscriminator,
)
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.loss import reconstruction_loss
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class Autoencoder(pl.LightningModule):
	def __init__(
		self,
		# encoder
		encoder_in_channels=128,
		encoder_out_channels=80,
		# rvq
		num_quantizers=8,
		codebook_size=1024,
		codebook_dim=128,
		# generator (= decoder)
		resblock="1",
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[16, 8, 4, 4],
		upsample_initial_channel=512,
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
		# Generator loss lambda
		lambda_recon=0.1,
		lambda_emb=0.1,
		lambda_fm=0.1,
	):
		super().__init__()
		self.lambda_recon = lambda_recon
		self.lambda_emb = lambda_emb
		self.lambda_fm = lambda_fm

		self.encoder = Encoder(
			n_in=encoder_in_channels,
			n_out=encoder_out_channels,
			parent_vc=None,
		)
		self.quantizer = ResidualVectorQuantizer(
			input_dim=encoder_out_channels,
			num_quantizers=num_quantizers,
			codebook_size=codebook_size,
			codebook_dim=codebook_dim,
		)
		self.decoder = Generator(
			generator_input_channels=encoder_out_channels,
			resblock=resblock,
			resblock_kernel_sizes=resblock_kernel_sizes,
			upsample_rates=upsample_rates,
			upsample_initial_channel=upsample_initial_channel,
			upsample_kernel_sizes=upsample_kernel_sizes,
			resblock_dilation_sizes=resblock_dilation_sizes,
		)
		self.mpd = MultiPeriodDiscriminator()
		self.msd = MultiScaleDiscriminator()

		self.automatic_optimization = False

	def training_step(self, batch, batch_idx):
		mel = batch["mel_spectrogram"]
		gt_audio = batch["audio"]

		optim_d, optim_g = self.optimizers()

		# Run autoencoder
		encoded = self.encoder(mel)
		quantized, commit_loss = self.quantizer(encoded)
		y_hat = self.decoder(quantized)

		# Discriminator optimizer step
		y_df_hat_r, y_df_hat_g, _, _ = self.mpd(gt_audio, y_hat.detach())
		loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
		y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(gt_audio, y_hat.detach())
		loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
		loss_disc_all = loss_disc_s + loss_disc_f

		optim_d.zero_grad()
		self.manual_backward(loss_disc_all)
		optim_d.step()

		# Generator optimizer step
		y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(gt_audio, y_hat)
		y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(gt_audio, y_hat)

		loss_gen_f, _ = generator_loss(y_df_hat_g)
		loss_gen_s, _ = generator_loss(y_ds_hat_g)
		loss_adv = loss_gen_f + loss_gen_s

		loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * self.lambda_fm
		loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * self.lambda_fm
		loss_fm = loss_fm_f + loss_fm_s

		loss_recon = reconstruction_loss(gt_audio, y_hat) * self.lambda_recon
		loss_emb = commit_loss * self.lambda_emb

		loss_gen_all = loss_adv + loss_fm + loss_recon + loss_emb

		optim_g.zero_grad()
		self.manual_backward(loss_gen_all)
		optim_g.step()

		self.log_dict(
			{
				"loss/g_total": loss_gen_all,
				"loss/d_total": loss_disc_all,
				"loss/adv": loss_adv,
				"loss/fm": loss_fm,
				"loss/recon": loss_recon,
				"loss/emb": loss_emb,
			},
			prog_bar=True,
		)

	def configure_optimizers(self):
		optim_g = torch.optim.AdamW(
			list(self.encoder.parameters())
			+ list(self.quantizer.parameters())
			+ list(self.decoder.parameters()),
			lr=2e-4,
			betas=(0.8, 0.99),
			weight_decay=0.01,
		)
		optim_d = torch.optim.AdamW(
			list(self.mpd.parameters()) + list(self.msd.parameters()),
			lr=2e-4,
			betas=(0.8, 0.99),
			weight_decay=0.01,
		)
		return optim_d, optim_g

	def forward(self, mel):
		encoded = self.encoder(mel)
		quantized, commit_loss = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded, commit_loss

	@torch.no_grad()
	def encode(self, mel):
		encoded = self.encoder(mel)
		return encoded

	@torch.no_grad()
	def decode(self, encoded):
		quantized, _ = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded

	@torch.no_grad()
	def sample(self, mel, device=None):
		self.eval()
		if device is not None:
			mel = mel.to(device)
		encoded = self.encoder(mel)
		quantized, _ = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded

	def remove_weight_norm(self):
		"""가중치 정규화를 제거하는 메서드 (추론 시 사용)"""
		self.decoder.remove_weight_norm()
