# -----------------------------------------------------------
# This code is adapted from ae-wavenet: https://github.com/hrbigelow/ae-wavenet
# Original repository: https://github.com/hrbigelow/ae-wavenet
# -----------------------------------------------------------


import torch
from torch import nn
from songsuro.autoencoder.encoder.vconv import VirtualConv, output_offsets
from songsuro.autoencoder.encoder.netmisc import xavier_init
from sys import stderr


class ConvReLURes(nn.Module):
	def __init__(
		self,
		n_in_chan,
		n_out_chan,
		filter_sz,
		stride=1,
		do_res=True,
		parent_vc=None,
		name=None,
	):
		super().__init__()
		self.n_in = n_in_chan
		self.n_out = n_out_chan
		# 필터 크기와 스트라이드에 따른 패딩 계산
		if stride == 1:
			# 입력과 출력 크기를 동일하게 유지하기 위한 패딩
			padding = (filter_sz - 1) // 2
			if filter_sz % 2 == 0:  # 짝수 필터 크기인 경우 비대칭 패딩 필요
				self.asymmetric_padding = True
				self.pad_left = filter_sz // 2 - 1
				self.pad_right = filter_sz // 2
				padding = 0  # Conv1d에는 패딩 없이 수동으로 적용
			else:
				self.asymmetric_padding = False
				padding = (filter_sz - 1) // 2
		else:
			# 스트라이드가 1보다 큰 경우 다운샘플링 발생
			padding = (filter_sz - 1) // 2
			self.asymmetric_padding = False

		self.conv = nn.Conv1d(
			n_in_chan, n_out_chan, filter_sz, stride, padding=padding, bias=True
		)
		self.relu = nn.ReLU()
		self.name = name
		# self.bn = nn.BatchNorm1d(n_out_chan)

		self.vc = VirtualConv(
			filter_info=filter_sz, stride=stride, parent=parent_vc, name=name
		)

		self.do_res = do_res
		if self.do_res:
			if stride != 1:
				print(
					"Stride must be 1 for residually connected convolution", file=stderr
				)
				raise ValueError
			l_off, r_off = output_offsets(self.vc, self.vc)
			self.register_buffer("residual_offsets", torch.tensor([l_off, r_off]))
		xavier_init(self.conv)

	def forward(self, x):
		"""
		B, C, T = n_batch, n_in_chan, n_win
		x: (B, C, T)
		"""
		identity = x.clone()  # 원본 입력 저장

		if self.asymmetric_padding:
			x = nn.functional.pad(x, (self.pad_left, self.pad_right))

		pre = self.conv(x)
		act = self.relu(pre)

		if self.do_res:
			# 입력과 출력의 크기 비교
			if act.shape[2] != identity.shape[2]:
				# 크기가 다른 경우 중앙 부분 잘라내기
				if act.shape[2] < identity.shape[2]:
					# 출력이 더 작은 경우 입력 잘라내기
					diff = identity.shape[2] - act.shape[2]
					start = diff // 2
					identity = identity[:, :, start : start + act.shape[2]]
				else:
					# 입력이 더 작은 경우 출력 잘라내기
					diff = act.shape[2] - identity.shape[2]
					start = diff // 2
					act = act[:, :, start : start + identity.shape[2]]

			# 이제 크기가 같으므로 더하기
			act = act + identity

		self.frac_zero_act = (act == 0.0).sum().double() / act.nelement()
		return act


class Encoder(nn.Module):
	def __init__(self, n_in, n_out, parent_vc):
		super().__init__()

		# the "stack"
		stack_in_chan = [n_in, n_out, n_out, n_out, n_out, n_out, n_out, n_out, n_out]
		stack_filter_sz = [3, 3, 4, 3, 3, 1, 1, 1, 1]
		stack_strides = [1, 1, 2, 1, 1, 1, 1, 1, 1]
		stack_residual = [False, True, False, True, True, True, True, True, True]
		# stack_residual = [True] * 9
		# stack_residual = [False] * 9
		stack_info = zip(stack_in_chan, stack_filter_sz, stack_strides, stack_residual)
		self.net = nn.Sequential()
		self.vc = dict()

		for i, (in_chan, filt_sz, stride, do_res) in enumerate(stack_info):
			name = "CRR_{}(filter_sz={}, stride={}, do_res={})".format(
				i, filt_sz, stride, do_res
			)
			mod = ConvReLURes(in_chan, n_out, filt_sz, stride, do_res, parent_vc, name)
			self.net.add_module(str(i), mod)
			parent_vc = mod.vc

		self.vc["beg"] = self.net[0].vc
		self.vc["end"] = self.net[-1].vc

	def set_parent_vc(self, parent_vc):
		self.vc["beg"].parent = parent_vc
		parent_vc.child = self.vc["beg"]

	def update_metrics(self):
		self.metrics = {}
		for i, mod in enumerate(self.net):
			# wkey = "enc_wz_{}".format(i)
			# bkey = "enc_bz_{}".format(i)
			akey = "enc_az_{}".format(i)
			self.metrics[akey] = mod.frac_zero_act
			# self.metrics[wkey] = (mod.conv.weight == 0).sum()
			# self.metrics[bkey] = (mod.conv.bias == 0).sum()

	def forward(self, mels):
		"""
		B, M, C, T = n_batch, n_mels, n_channels, n_timesteps
		mels: (B, M, T) (torch.tensor)
		outputs: (B, C, T)
		"""
		out = self.net(mels)
		self.update_metrics()
		# out = torch.tanh(out * 10.0)
		return out
