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
		# 메모리 최적화된 버전
		# with torch.cuda.amp.autocast():  # 혼합 정밀도 사용
		# 불필요한 변수 할당 없이 직접 permute 사용
		x_reshaped = x.permute(0, 2, 1).contiguous()

		quantized, indices, commit_losses = self.rvq(x_reshaped)

		# 원래 형태로 효율적으로 복원
		quantized = quantized.permute(0, 2, 1).contiguous()

		commit_loss = torch.sum(commit_losses)

		return quantized, commit_loss
