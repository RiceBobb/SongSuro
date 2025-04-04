import torch
import torch.nn as nn
import math


class RelativePositionBias(nn.Module):
	def __init__(self, num_buckets=32, max_distance=128, num_heads=2):
		super().__init__()
		self.num_buckets = num_buckets
		self.max_distance = max_distance
		self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

	def _relative_position_bucket(self, relative_position):
		relative_buckets = 0
		num_buckets = self.num_buckets
		max_distance = self.max_distance

		num_buckets //= 2
		relative_buckets += (relative_position > 0).long() * num_buckets
		relative_position = torch.abs(relative_position)

		max_exact = num_buckets // 2
		is_small = relative_position < max_exact

		relative_position_if_large = (
			max_exact
			+ (
				torch.log(relative_position.float() / max_exact)
				/ math.log(max_distance / max_exact)
				* (num_buckets - max_exact)
			).long()
		)
		relative_position_if_large = torch.min(
			relative_position_if_large,
			torch.full_like(relative_position_if_large, num_buckets - 1),
		)

		relative_buckets += torch.where(
			is_small, relative_position, relative_position_if_large
		)
		return relative_buckets

	def forward(self, query_length, key_length):
		context_pos = torch.arange(query_length)[:, None]
		memory_pos = torch.arange(key_length)[None, :]
		relative_pos = memory_pos - context_pos

		relative_buckets = self._relative_position_bucket(relative_pos)
		bias = self.relative_attention_bias(relative_buckets)
		return bias.permute(2, 0, 1).unsqueeze(0)


class FFTransformerBlock(nn.Module):
	def __init__(self, d_model=192, n_heads=2, conv_kernel=9, dropout=0.1):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
		self.conv1 = nn.Conv1d(
			d_model,
			d_model * 4,
			kernel_size=conv_kernel,
			padding=(conv_kernel - 1) // 2,
		)
		self.conv2 = nn.Conv1d(
			d_model * 4,
			d_model,
			kernel_size=conv_kernel,
			padding=(conv_kernel - 1) // 2,
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.rel_pos = RelativePositionBias(num_heads=n_heads)
		self.gelu = nn.GELU()

	def forward(self, x, mask=None):
		batch_size, seq_len, _ = x.shape

		# 상대 위치 인코딩 적용
		rel_bias = self.rel_pos(seq_len, seq_len).to(x.device)
		x_attn = x.permute(1, 0, 2)
		attn_out, _ = self.attn(
			x_attn, x_attn, x_attn, attn_mask=rel_bias, key_padding_mask=mask
		)
		x = x + self.dropout(attn_out.permute(1, 0, 2))
		x = self.norm1(x)

		# 컨볼루션 피드포워드
		residual = x
		x = x.permute(0, 2, 1)  # [B, C, T]
		x = self.conv1(x)
		x = self.gelu(x)
		x = self.dropout(x)
		x = self.conv2(x)
		x = x.permute(0, 2, 1)  # [B, T, C]
		x = residual + self.dropout(x)
		return self.norm2(x)


class LyricsEncoder(nn.Module):
	def __init__(self, vocab_size, d_model=192, n_layers=4):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, d_model)
		self.blocks = nn.ModuleList(
			[FFTransformerBlock(d_model=d_model) for _ in range(n_layers)]
		)
		self.prior = nn.Linear(d_model, d_model)

	def forward(self, x, mask=None):
		x = self.embed(x)
		for block in self.blocks:
			x = block(x, mask)
		return self.prior(x)
