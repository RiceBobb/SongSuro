import torch.nn as nn
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class Autoencoder(nn.Module):
	def __init__(
		self,
		# encoder (instance 만들 때, mel_channels를 계산해서 넣어줘야 함)
		encoder_in_channels=128,
		encoder_out_channels=80,
		# rvq
		num_quantizers=8,
		codebook_size=1024,
		codebook_dim=128,
		# generator
		resblock="1",
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[8, 8, 2, 2],
		upsample_initial_channel=512,
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
	):
		super().__init__()
		self.init_args = {
			"encoder_in_channels": encoder_in_channels,
			"encoder_out_channels": encoder_out_channels,
			"num_quantizers": num_quantizers,
			"codebook_size": codebook_size,
			"codebook_dim": codebook_dim,
			"resblock": resblock,
			"resblock_kernel_sizes": resblock_kernel_sizes,
			"upsample_rates": upsample_rates,
			"upsample_initial_channel": upsample_initial_channel,
			"upsample_kernel_sizes": upsample_kernel_sizes,
			"resblock_dilation_sizes": resblock_dilation_sizes,
		}

		self._initialize()

	def _initialize(self):
		# Encoder
		encoder_out_channels = self.init_args["encoder_out_channels"]

		self.encoder = Encoder(
			n_in=self.init_args["encoder_in_channels"],
			n_out=encoder_out_channels,
			parent_vc=None,
		)

		# RVQ(Residual Vector Quantizer)
		self.quantizer = ResidualVectorQuantizer(
			input_dim=encoder_out_channels,
			num_quantizers=self.init_args["num_quantizers"],
			codebook_size=self.init_args["codebook_size"],
			codebook_dim=self.init_args["codebook_dim"],
		)

		# Generator (= Decoder)
		self.decoder = Generator(
			generator_input_channels=encoder_out_channels,
			resblock=self.init_args["resblock"],
			resblock_kernel_sizes=self.init_args["resblock_kernel_sizes"],
			upsample_rates=self.init_args["upsample_rates"],
			upsample_initial_channel=self.init_args["upsample_initial_channel"],
			upsample_kernel_sizes=self.init_args["upsample_kernel_sizes"],
			resblock_dilation_sizes=self.init_args["resblock_dilation_sizes"],
		)

	def forward(self, x):
		encoded = self.encoder(x)
		quantized, commit_loss = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded, commit_loss

	def remove_weight_norm(self):
		"""가중치 정규화를 제거하는 메서드 (추론 시 사용)"""
		self.decoder.remove_weight_norm()
