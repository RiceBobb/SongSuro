import torch
import pytorch_lightning as pl
import torchaudio
from torch import nn

from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.loss import reconstruction_loss
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class SimpleAutoencoder(pl.LightningModule):
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
		# Loss lambda values
		lambda_recon=1.0,
		lambda_emb=1.0,
	):
		super().__init__()
		self.lambda_recon = lambda_recon
		self.lambda_emb = lambda_emb

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

	def training_step(self, batch, batch_idx):
		mel = batch["mel_spectrogram"]
		gt_audio = batch["audio"]

		# Run autoencoder
		encoded = self.encoder(mel)
		quantized, commit_loss = self.quantizer(encoded)
		y_hat = self.decoder(quantized)

		# Calculate losses
		loss_recon = reconstruction_loss(gt_audio, y_hat) * self.lambda_recon
		loss_emb = commit_loss * self.lambda_emb

		# Total loss
		total_loss = loss_recon + loss_emb

		# Log metrics
		self.log_dict(
			{
				"loss/total": total_loss,
				"loss/recon": loss_recon,
				"loss/emb": loss_emb,
			},
			on_step=True,
			prog_bar=True,
			logger=True,
		)

		return total_loss

	def on_train_epoch_end(self):
		scheduler_g = self.schedulers()
		scheduler_g.step()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			self.parameters(),
			lr=2e-4,
			betas=(0.8, 0.99),
			weight_decay=0.01,
		)

		scheduler = torch.optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=0.998,
			last_epoch=self.current_epoch - 1 if self.current_epoch > 0 else -1,
		)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "epoch",
			},
		}

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
