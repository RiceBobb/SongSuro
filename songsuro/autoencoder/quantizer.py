import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class ResidualVectorQuantizer(nn.Module):
	def __init__(
		self, input_dim=128, num_quantizers=30, codebook_size=1024, codebook_dim=128
	):
		super().__init__()
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
