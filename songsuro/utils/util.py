import glob
import os
import subprocess

import torch
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


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


def init_weights(m, mean=0.0, std=0.01):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
	return int((kernel_size * dilation - dilation) / 2)


def clip_grad_value(parameters, clip_value, norm_type=2):
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = list(filter(lambda p: p.grad is not None, parameters))
	norm_type = float(norm_type)
	if clip_value is not None:
		clip_value = float(clip_value)

	total_norm = 0
	for p in parameters:
		param_norm = p.grad.data.norm(norm_type)
		total_norm += param_norm.item() ** norm_type
		if clip_value is not None:
			p.grad.data.clamp_(min=-clip_value, max=clip_value)
	total_norm = total_norm ** (1.0 / norm_type)
	return total_norm


# 이 아래부터는 VISinger2에서 사용된
def load_checkpoint(checkpoint_path, model, optimizer=None):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
	iteration = checkpoint_dict["iteration"]
	learning_rate = checkpoint_dict["learning_rate"]
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint_dict["optimizer"])
	saved_state_dict = checkpoint_dict["model"]
	if hasattr(model, "module"):
		state_dict = model.module.state_dict()
	else:
		state_dict = model.state_dict()
	new_state_dict = {}
	for k, v in state_dict.items():
		try:
			new_state_dict[k] = saved_state_dict[k]
		except:
			print("error, %s is not in the checkpoint" % k)
			logger.info("%s is not in the checkpoint" % k)
			new_state_dict[k] = v
	if hasattr(model, "module"):
		model.module.load_state_dict(new_state_dict)
	else:
		model.load_state_dict(new_state_dict)
	print("load ")
	logger.info(
		"Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
	)
	return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
	logger.info(
		"Saving model and optimizer state at iteration {} to {}".format(
			iteration, checkpoint_path
		)
	)
	if hasattr(model, "module"):
		state_dict = model.module.state_dict()
	else:
		state_dict = model.state_dict()
	torch.save(
		{
			"model": state_dict,
			"iteration": iteration,
			"optimizer": optimizer.state_dict(),
			"learning_rate": learning_rate,
		},
		checkpoint_path,
	)


def summarize(
	writer,
	global_step,
	scalars={},
	histograms={},
	images={},
	audios={},
	audio_sampling_rate=22050,
):
	for k, v in scalars.items():
		writer.add_scalar(k, v, global_step)
	for k, v in histograms.items():
		writer.add_histogram(k, v, global_step)
	for k, v in images.items():
		writer.add_image(k, v, global_step, dataformats="HWC")
	for k, v in audios.items():
		writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
	f_list = glob.glob(os.path.join(dir_path, regex))
	f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
	x = f_list[-1]
	print(x)
	return x


def get_logger(model_dir, filename="train.log"):
	global logger
	logger = logging.getLogger(os.path.basename(model_dir))
	logger.setLevel(logging.DEBUG)

	formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	h = logging.FileHandler(os.path.join(model_dir, filename))
	h.setLevel(logging.DEBUG)
	h.setFormatter(formatter)
	logger.addHandler(h)
	return logger


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def check_git_hash(model_dir):
	source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	if not os.path.exists(os.path.join(source_dir, ".git")):
		logger.warn(
			"{} is not a git repository, therefore hash value comparison will be ignored.".format(
				source_dir
			)
		)
		return

	cur_hash = subprocess.getoutput("git rev-parse HEAD")

	path = os.path.join(model_dir, "githash")
	if os.path.exists(path):
		saved_hash = open(path).read()
		if saved_hash != cur_hash:
			logger.warn(
				"git hash values are different. {}(saved) != {}(current)".format(
					saved_hash[:8], cur_hash[:8]
				)
			)
	else:
		open(path, "w").write(cur_hash)


def plot_spectrogram_to_numpy(spectrogram):
	global MATPLOTLIB_FLAG
	if not MATPLOTLIB_FLAG:
		import matplotlib

		matplotlib.use("Agg")
		MATPLOTLIB_FLAG = True
		mpl_logger = logging.getLogger("matplotlib")
		mpl_logger.setLevel(logging.WARNING)
	import matplotlib.pylab as plt
	import numpy as np

	fig, ax = plt.subplots(figsize=(10, 2))
	im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
	plt.colorbar(im, ax=ax)
	plt.xlabel("Frames")
	plt.ylabel("Channels")
	plt.tight_layout()

	fig.canvas.draw()
	data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	plt.close()
	return data
