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
