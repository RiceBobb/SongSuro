import torch


def group_hidden_by_segs(h, seg_ids, max_len, is_BHT=False):
	"""
	Code from TCSinger (EMNLP 2024)
	https://github.com/AaronZ345/TCSinger

	:param h: [B, T, H]
	:param seg_ids: [B, T]
	:return: h_ph: [B, T_ph, H]
	"""
	if is_BHT:
		h = h.transpose(1, 2)
	B, T, H = h.shape
	h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(
		1, seg_ids[:, :, None].repeat([1, 1, H]), h
	)
	all_ones = h.new_ones(h.shape[:2])
	cnt_gby_segs = (
		h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
	)
	h_gby_segs = h_gby_segs[:, 1:]
	cnt_gby_segs = cnt_gby_segs[:, 1:]
	h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
	if is_BHT:
		h_gby_segs = h_gby_segs.transpose(1, 2)
	return h_gby_segs, cnt_gby_segs


def temporal_avg_pool(x, mask=None):
	"""
	Code from TCSinger (EMNLP 2024)
	Computes masked temporal average pooling along the sequence dimension.
	Ignores padded/masked elements using boolean mask.

	Note:
		- Designed for variable-length sequences where padding needs to be ignored
		- Maintains original feature dimension while reducing temporal dimension to 1
		- Handles zero-length sequences gracefully through PyTorch's division operation

	:param x: Input sequence tensor of shape (batch_size, seq_len, features)
	:param mask: Boolean mask tensor of shape (batch_size, seq_len, features) where True indicates masked/padded positions
	:return: Pooled output tensor of shape (batch_size, features, 1)
	"""
	len_ = (~mask).sum(dim=-1).unsqueeze(-1)
	x = x.masked_fill(mask, 0)
	x = x.sum(dim=-1).unsqueeze(-1)
	# Create a mask for zero-length sequences (all masked)
	zero_mask = len_ == 0
	# Replace zeros in len_ with ones to avoid division by zero
	len_ = torch.where(zero_mask, torch.ones_like(len_), len_)
	out = torch.div(x, len_)
	out = torch.where(zero_mask, torch.zeros_like(out), out)
	return out
