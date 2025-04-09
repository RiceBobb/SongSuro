import torch.nn as nn
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class Autoencoder(nn.Module):
	def __init__(self, hps):
		super().__init__()
		self.hps = hps

		# 인코더 초기화
		self.encoder = Encoder(
			n_in=getattr(hps.model, "encoder_in_channels", 1),  # 기본값 1
			n_out=getattr(hps.model, "encoder_out_channels", 128),  # 기본값 128
			parent_vc=None,
		)

		# 양자화기 초기화
		self.quantizer = ResidualVectorQuantizer(
			input_dim=getattr(hps.model, "encoder_out_channels", 128),
			num_quantizers=getattr(hps.model, "num_quantizers", 8),
			codebook_size=getattr(hps.model, "codebook_size", 1024),
			codebook_dim=getattr(hps.model, "codebook_dim", 128),
		)

		# 디코더(Generator) 초기화
		# Generator 클래스는 h 객체를 받으므로 hps.model을 전달
		self.decoder = Generator(hps.model)

	def forward(self, x):
		encoded = self.encoder(x)
		quantized, commit_loss = self.quantizer(encoded)
		decoded = self.decoder(quantized)
		return decoded, commit_loss

	def remove_weight_norm(self):
		"""가중치 정규화를 제거하는 메서드 (추론 시 사용)"""
		self.decoder.remove_weight_norm()
