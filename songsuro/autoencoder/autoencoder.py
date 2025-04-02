import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


# WaveNet 인코더 부분 (residual blocks)
class WaveNetEncoder(nn.Module):
	def __init__(
		self,
		layers=10,
		blocks=4,
		dilation_channels=32,
		residual_channels=32,
		skip_channels=256,
		kernel_size=2,
	):
		super(WaveNetEncoder, self).__init__()

		self.layers = layers
		self.blocks = blocks
		self.dilation_channels = dilation_channels
		self.residual_channels = residual_channels
		self.skip_channels = skip_channels
		self.kernel_size = kernel_size

		# 초기 컨볼루션 레이어
		self.start_conv = nn.Conv1d(
			in_channels=1, out_channels=residual_channels, kernel_size=1
		)

		# Residual blocks
		self.dilations = []
		self.dilated_convs = nn.ModuleList()
		self.residual_convs = nn.ModuleList()
		self.skip_convs = nn.ModuleList()

		# 각 블록과 레이어에 대한 dilation 설정
		for block in range(blocks):
			for layer in range(layers):
				dilation = 2**layer
				self.dilations.append(dilation)

				# Dilated convolutions
				self.dilated_convs.append(
					nn.Conv1d(
						in_channels=residual_channels,
						out_channels=dilation_channels,
						kernel_size=kernel_size,
						dilation=dilation,
						padding=dilation,
					)
				)

				# Residual convolutions
				self.residual_convs.append(
					nn.Conv1d(
						in_channels=dilation_channels,
						out_channels=residual_channels,
						kernel_size=1,
					)
				)

				# Skip connections
				self.skip_convs.append(
					nn.Conv1d(
						in_channels=dilation_channels,
						out_channels=skip_channels,
						kernel_size=1,
					)
				)

		# 출력 레이어
		self.final_conv = nn.Conv1d(
			in_channels=skip_channels,
			out_channels=256,  # 인코딩 차원
			kernel_size=1,
		)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.start_conv(x)
		skip = 0

		# Residual blocks 처리
		for i in range(len(self.dilated_convs)):
			residual = x

			# Dilated convolution
			x = self.dilated_convs[i](x)
			x = torch.tanh(x)

			# Residual connection
			x = self.residual_convs[i](x)
			x = x + residual[:, :, : x.size(2)]

			# Skip connection
			s = self.skip_convs[i](x)
			if skip == 0:
				skip = s
			else:
				skip = skip + s[:, :, : skip.size(2)]

		# 최종 출력
		x = self.relu(skip)
		x = self.final_conv(x)

		return x


# HiFi-GAN V1 디코더 부분
class HiFiGANDecoder(nn.Module):
	def __init__(self, input_channels=256):
		super(HiFiGANDecoder, self).__init__()

		self.input_channels = input_channels

		# 초기 컨볼루션 레이어
		self.initial_conv = nn.Conv1d(
			in_channels=input_channels, out_channels=512, kernel_size=7, padding=3
		)

		# Upsampling 블록
		self.upsamples = nn.ModuleList(
			[
				nn.ConvTranspose1d(
					in_channels=512,
					out_channels=256,
					kernel_size=16,
					stride=8,
					padding=4,
				),
				nn.ConvTranspose1d(
					in_channels=256,
					out_channels=128,
					kernel_size=16,
					stride=8,
					padding=4,
				),
				nn.ConvTranspose1d(
					in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
				),
			]
		)

		# Residual blocks
		self.resblocks1 = nn.ModuleList(
			[
				ResBlock(256, kernel_sizes=[3, 7, 11]),
				ResBlock(256, kernel_sizes=[3, 7, 11]),
				ResBlock(256, kernel_sizes=[3, 7, 11]),
			]
		)

		self.resblocks2 = nn.ModuleList(
			[
				ResBlock(128, kernel_sizes=[3, 7, 11]),
				ResBlock(128, kernel_sizes=[3, 7, 11]),
			]
		)

		self.resblocks3 = nn.ModuleList(
			[
				ResBlock(64, kernel_sizes=[3, 7, 11]),
				ResBlock(64, kernel_sizes=[3, 7, 11]),
			]
		)

		# 최종 출력 레이어
		self.final_conv = nn.Conv1d(
			in_channels=64, out_channels=1, kernel_size=7, padding=3
		)

		self.leaky_relu = nn.LeakyReLU(0.1)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.initial_conv(x)
		x = self.leaky_relu(x)

		# 첫 번째 업샘플링 및 ResBlock
		x = self.upsamples[0](x)
		x = self.leaky_relu(x)
		for resblock in self.resblocks1:
			x = resblock(x)

		# 두 번째 업샘플링 및 ResBlock
		x = self.upsamples[1](x)
		x = self.leaky_relu(x)
		for resblock in self.resblocks2:
			x = resblock(x)

		# 세 번째 업샘플링 및 ResBlock
		x = self.upsamples[2](x)
		x = self.leaky_relu(x)
		for resblock in self.resblocks3:
			x = resblock(x)

		# 최종 출력
		x = self.final_conv(x)
		x = self.tanh(x)

		return x


# HiFi-GAN 디코더의 ResBlock
class ResBlock(nn.Module):
	def __init__(self, channels, kernel_sizes):
		super(ResBlock, self).__init__()

		self.convs = nn.ModuleList()
		for kernel_size in kernel_sizes:
			self.convs.append(
				nn.Sequential(
					nn.LeakyReLU(0.1),
					nn.Conv1d(
						channels, channels, kernel_size, padding=kernel_size // 2
					),
					nn.LeakyReLU(0.1),
					nn.Conv1d(
						channels, channels, kernel_size, padding=kernel_size // 2
					),
				)
			)

	def forward(self, x):
		for conv in self.convs:
			x = x + conv(x)
		return x


class ResidualVectorQuantizer(nn.Module):
	def __init__(
		self, input_dim=256, num_quantizers=30, codebook_size=1024, codebook_dim=128
	):
		super(ResidualVectorQuantizer, self).__init__()
		self.rvq = ResidualVQ(
			dim=input_dim,
			num_quantizers=num_quantizers,
			codebook_size=codebook_size,
			codebook_dim=codebook_dim,
			commitment_weight=1.0,
		)

	def forward(self, x):
		# 입력 형태 변환: [batch, channels, seq_len] -> [batch, seq_len, channels]
		x = x.transpose(1, 2)

		# ResidualVQ 적용
		quantized, indices, commit_losses = self.rvq(x)

		# 출력 형태 복원: [batch, seq_len, channels] -> [batch, channels, seq_len]
		quantized = quantized.transpose(1, 2)

		# 모든 commitment loss 합산
		commit_loss = torch.sum(commit_losses)

		return quantized, commit_loss


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


# 모델 초기화 예시
def create_model():
	encoder = WaveNetEncoder()
	quantizer = ResidualVectorQuantizer()
	decoder = HiFiGANDecoder()
	model = Autoencoder(encoder, quantizer, decoder)
	return model
