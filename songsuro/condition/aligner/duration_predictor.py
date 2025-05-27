import torch
from songsuro.condition.aligner.length_regulator import LayerNorm


class DurationPredictor(torch.nn.Module):
	"""Duration predictor module.
	This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
	The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
	.. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
	    https://arxiv.org/pdf/1905.09263.pdf
	Note:
	    The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
	    the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
	"""

	def __init__(
		self,
		idim,
		n_layers=2,
		n_chans=384,
		kernel_size=3,
		dropout_rate=0.1,
		offset=1.0,
		padding="SAME",
	):
		"""Initilize duration predictor module.
		Args:
		    idim (int): Input dimension.
		    n_layers (int, optional): Number of convolutional layers.
		    n_chans (int, optional): Number of channels of convolutional layers.
		    kernel_size (int, optional): Kernel size of convolutional layers.
		    dropout_rate (float, optional): Dropout rate.
		    offset (float, optional): Offset value to avoid nan in log domain.
		"""
		super(DurationPredictor, self).__init__()
		self.offset = offset
		self.conv = torch.nn.ModuleList()
		self.kernel_size = kernel_size
		self.padding = padding
		for idx in range(n_layers):
			in_chans = idim if idx == 0 else n_chans
			self.conv += [
				torch.nn.Sequential(
					torch.nn.ConstantPad1d(
						((kernel_size - 1) // 2, (kernel_size - 1) // 2)
						if padding == "SAME"
						else (kernel_size - 1, 0),
						0,
					),
					torch.nn.Conv1d(
						in_chans, n_chans, kernel_size, stride=1, padding=0
					),
					torch.nn.ReLU(),
					LayerNorm(n_chans, dim=1),
					torch.nn.Dropout(dropout_rate),
				)
			]
		if hparams["dur_loss"] in ["mse", "huber"]:
			odims = 1
		elif hparams["dur_loss"] == "mog":
			odims = 15
		elif hparams["dur_loss"] == "crf":
			odims = 32
			from torchcrf import CRF

			self.crf = CRF(odims, batch_first=True)
		self.linear = torch.nn.Linear(n_chans, odims)

	def _forward(self, xs, x_masks=None, is_inference=False):
		xs = xs.transpose(1, -1)  # (B, idim, Tmax)
		for f in self.conv:
			xs = f(xs)  # (B, C, Tmax)
			if x_masks is not None:
				xs = xs * (1 - x_masks.float())[:, None, :]

		xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
		xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)
		if is_inference:
			return self.out2dur(xs), xs
		else:
			if hparams["dur_loss"] in ["mse"]:
				xs = xs.squeeze(-1)  # (B, Tmax)
		return xs

	def out2dur(self, xs):
		if hparams["dur_loss"] in ["mse"]:
			# NOTE: calculate in log domain
			xs = xs.squeeze(-1)  # (B, Tmax)
			dur = torch.clamp(
				torch.round(xs.exp() - self.offset), min=0
			).long()  # avoid negative value
		elif hparams["dur_loss"] == "mog":
			return NotImplementedError
		elif hparams["dur_loss"] == "crf":
			dur = torch.LongTensor(self.crf.decode(xs)).cuda()
		return dur

	def forward(self, xs, x_masks=None):
		"""Calculate forward propagation.
		Args:
		    xs (Tensor): Batch of input sequences (B, Tmax, idim).
		    x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
		Returns:
		    Tensor: Batch of predicted durations in log domain (B, Tmax).
		"""
		return self._forward(xs, x_masks, False)

	def inference(self, xs, x_masks=None):
		"""Inference duration.
		Args:
		    xs (Tensor): Batch of input sequences (B, Tmax, idim).
		    x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
		Returns:
		    LongTensor: Batch of predicted durations in linear domain (B, Tmax).
		"""
		return self._forward(xs, x_masks, True)
