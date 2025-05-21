import torch
from torch import nn
import pytorch_lightning as pl
import torchaudio

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


def print_memory_stats(step_name):
	if torch.cuda.is_available():
		print(
			f"\n{step_name} - MPS 메모리 할당: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
		)


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
		upsample_rates=[8, 8, 2, 2],  # 16, 8, 4, 4 였음
		upsample_initial_channel=512,
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
		# Generator loss lambda
		lambda_recon=0.1,
		lambda_emb=0.1,
		lambda_fm=0.1,
	):
		super().__init__()
		self.encoder_in_channels = encoder_in_channels
		self.encoder_out_channels = encoder_out_channels
		self.num_quantizers = num_quantizers
		self.codebook_size = codebook_size
		self.codebook_dim = codebook_dim
		self.resblock = resblock
		self.resblock_kernel_sizes = resblock_kernel_sizes
		self.upsample_rates = upsample_rates
		self.upsample_initial_channel = upsample_initial_channel
		self.upsample_kernel_sizes = upsample_kernel_sizes
		self.resblock_dilation_sizes = resblock_dilation_sizes
		self.lambda_recon = lambda_recon
		self.lambda_emb = lambda_emb
		self.lambda_fm = lambda_fm

		self.encoder = None
		self.decoder = None
		self.quantizer = None
		self.mpd = None
		self.msd = None

		self.automatic_optimization = False

	def configure_model(self):
		if self.encoder is None:
			self.encoder = Encoder(
				n_in=self.encoder_in_channels,
				n_out=self.encoder_out_channels,
				parent_vc=None,
			)
		if self.quantizer is None:
			self.quantizer = ResidualVectorQuantizer(
				input_dim=self.encoder_out_channels,
				num_quantizers=self.num_quantizers,
				codebook_size=self.codebook_size,
				codebook_dim=self.codebook_dim,
			)
		if self.decoder is None:
			self.decoder = Generator(
				generator_input_channels=self.encoder_out_channels,
				resblock=self.resblock,
				resblock_kernel_sizes=self.resblock_kernel_sizes,
				upsample_rates=self.upsample_rates,
				upsample_initial_channel=self.upsample_initial_channel,
				upsample_kernel_sizes=self.upsample_kernel_sizes,
				resblock_dilation_sizes=self.resblock_dilation_sizes,
			)
		if self.mpd is None:
			self.mpd = MultiPeriodDiscriminator()
		if self.msd is None:
			self.msd = MultiScaleDiscriminator()

	def training_step(self, batch, batch_idx):
		mel = batch["mel_spectrogram"]
		gt_audio = batch["audio"]

		optim_d, optim_g = self.optimizers()
		print_memory_stats("Before training step")

		# Run autoencoder
		encoded = self.encoder(mel)
		print_memory_stats("After encoder")
		quantized, commit_loss = self.quantizer(encoded)
		print_memory_stats("After quantizer")
		y_hat = self.decoder(quantized)
		print_memory_stats("After decoder")

		# Discriminator optimizer step
		y_df_hat_r, y_df_hat_g, _, _ = self.mpd(gt_audio, y_hat.detach())
		loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
		y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(gt_audio, y_hat.detach())
		loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
		loss_disc_all = loss_disc_s + loss_disc_f

		optim_d.zero_grad()
		self.manual_backward(loss_disc_all)
		self.clip_gradients(optim_d, gradient_clip_val=None)
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
		self.clip_gradients(optim_g, gradient_clip_val=None)
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
			on_step=True,
			prog_bar=True,
			logger=True,
		)

	def on_train_epoch_end(self):
		scheduler_d, scheduler_g = self.schedulers()
		scheduler_d.step()
		scheduler_g.step()

	def configure_optimizers(self):
		optim_g = torch.optim.SGD(
			list(self.encoder.parameters())
			+ list(self.quantizer.parameters())
			+ list(self.decoder.parameters()),
			lr=0.01,  # SGD는 일반적으로 AdamW보다 더 큰 학습률 사용
			momentum=0.9,  # 모멘텀 추가 (AdamW의 beta1과 유사한 역할)
			weight_decay=0.01,  # 원래 weight_decay 유지
		)
		optim_d = torch.optim.SGD(
			list(self.mpd.parameters()) + list(self.msd.parameters()),
			lr=0.01,  # SGD는 일반적으로 AdamW보다 더 큰 학습률 사용
			momentum=0.9,  # 모멘텀 추가
			weight_decay=0.01,  # 원래 weight_decay 유지
		)

		scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
			optim_g,
			gamma=0.998,
			last_epoch=self.current_epoch - 1 if self.current_epoch > 0 else -1,
		)
		scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
			optim_d,
			gamma=0.998,
			last_epoch=self.current_epoch - 1 if self.current_epoch > 0 else -1,
		)

		return [optim_d, optim_g], [scheduler_d, scheduler_g]

	def validation_step(self, batch, batch_idx):
		mel = batch["mel_spectrogram"]
		res = self.forward(batch)
		y_hat = res["pred_audio"].cpu()

		# TODO: sample rates can be different among audio samples
		mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=res["sample_rates"][0],
			n_fft=2048,
			hop_length=1024,
			f_max=8000,
		)
		y_hat_mel = mel_spectrogram_transform(y_hat)

		val_loss = nn.L1Loss(reduction="mean")(
			y_hat_mel[..., : mel.shape[-1]].type_as(mel),
			mel,
		)
		self.log("val_loss", val_loss, on_step=True, prog_bar=True, logger=True)
		return {
			"loss": val_loss,
			"pred_audio": y_hat,
			"pred_mel": y_hat_mel,
		}

	def test_step(self, batch, batch_idx):
		result = self.forward(batch)
		y_hat = result["pred_audio"].cpu()

		mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=batch["sample_rates"][0],
			n_fft=2048,
			hop_length=1024,
			f_max=8000,
		)
		y_hat_mel = mel_spectrogram_transform(y_hat)
		mel = batch["mel_spectrogram"]
		l1_loss = nn.L1Loss(reduction="mean")(
			y_hat_mel[..., : mel.shape[-1]].type_as(mel),
			mel,
		)
		self.log("test_loss", l1_loss, on_step=True, prog_bar=True, logger=True)
		return {
			"loss": l1_loss,
			"pred_audio": y_hat,
			"pred_mel": y_hat_mel,
		}

	def on_validation_start(self):
		self.decoder.remove_weight_norm()

	def on_validation_end(self):
		self.decoder.add_weight_norm()

	def on_test_start(self):
		self.encoder.remove_weight_norm()

	def on_test_end(self):
		self.encoder.add_weight_norm()

	def forward(self, batch):
		mel = batch["mel_spectrogram"]

		encoded = self.encoder(mel)
		quantized, _ = self.quantizer(encoded)
		y_hat = self.decoder(quantized)

		return {
			"pred_audio": y_hat.squeeze(1),
			"sample_rates": batch["sample_rates"],
		}

	@torch.no_grad()
	def encode(self, mel):
		encoded = self.encoder(mel)
		return encoded

	@torch.no_grad()
	def decode(self, encoded):
		quantized, _ = self.quantizer(encoded)
		self.decoder.remove_weight_norm()
		decoded = self.decoder(quantized)
		return decoded
