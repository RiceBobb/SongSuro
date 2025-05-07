import glob
import os
import subprocess
import sys
import logging
import warnings

import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from songsuro.data.loader.base import BaseDataLoader

from songsuro.autoencoder.loss import reconstruction_loss
from songsuro.autoencoder.decoder.decoder_loss import (
	generator_loss,
	discriminator_loss,
	feature_loss,
)

from songsuro.autoencoder.models import Autoencoder
from songsuro.autoencoder.decoder.discriminator import (
	MultiPeriodDiscriminator,
	MultiScaleDiscriminator,
)

from songsuro.preprocess import make_mel_spectrogram_from_audio


warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.append("../..")

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)

logger = logging.getLogger("SongSuro")


def main(dataset, batch_size=16, port=8001):
	"""Assume Single Node Multi GPUs Training Only"""
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = str(port)

	# 데이터셋 분할 비율 설정
	train_ratio = 0.8

	# 데이터셋 크기 계산
	dataset_size = len(dataset)
	train_size = int(train_ratio * dataset_size)
	val_size = dataset_size - train_size

	# 데이터셋 분할
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	if torch.cuda.is_available():
		n_gpus = torch.cuda.device_count()
		mp.spawn(
			run, nprocs=n_gpus, args=(n_gpus, train_dataset, val_dataset, batch_size)
		)
	else:
		cpurun(0, 1, train_dataset, val_dataset, batch_size)


def run(
	rank,
	n_gpus,
	save_dir="/root/logdir/songsuro",
	seed=1234,
	lr_decay=0.998,
	epochs=1000,
	train_dataset=None,
	valid_dataset=None,
	batch_size=2,
):
	if rank == 0:
		check_git_hash(save_dir)
		writer = SummaryWriter(log_dir=save_dir)
		writer_eval = SummaryWriter(log_dir=os.path.join(save_dir, "eval"))

	dist.init_process_group(
		backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
	)
	torch.manual_seed(seed)
	torch.cuda.set_device(rank)

	train_loader = BaseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = BaseDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

	net_g = Autoencoder().cuda(rank)
	mpd = MultiPeriodDiscriminator().cuda(rank)
	msd = MultiScaleDiscriminator().cuda(rank)

	optim_g = torch.optim.AdamW(
		net_g.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01
	)
	optim_d = torch.optim.AdamW(
		list(mpd.parameters()) + list(msd.parameters()),
		lr=2e-4,
		betas=(0.8, 0.99),
		weight_decay=0.01,
	)

	net_g = DDP(net_g, device_ids=[rank])
	mpd = DDP(mpd, device_ids=[rank])
	msd = DDP(msd, device_ids=[rank])

	try:
		_, _, _, epoch_str = load_checkpoint(
			latest_checkpoint_path(save_dir, "G_*.pth"), net_g, optim_g
		)
		_, _, _, epoch_str = load_checkpoint(
			latest_checkpoint_path(save_dir, "D_*.pth"), [mpd, msd], [optim_d]
		)
		global_step = (epoch_str - 1) * len(train_loader)
	except:
		epoch_str = 1
		global_step = 0

	scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
		optim_g, gamma=lr_decay, last_epoch=epoch_str - 2
	)
	scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
		optim_d, gamma=lr_decay, last_epoch=epoch_str - 2
	)

	for epoch in range(epoch_str, epochs + 1):
		if rank == 0:
			global_step = train_and_evaluate(
				rank,
				epoch,
				[net_g, mpd, msd],
				[optim_g, optim_d],
				[scheduler_g, scheduler_d],
				[train_loader, valid_loader],
				logger,
				[writer, writer_eval],
				global_step,
			)
		else:
			global_step = train_and_evaluate(
				rank,
				epoch,
				[net_g, mpd, msd],
				[optim_g, optim_d],
				[scheduler_g, scheduler_d],
				[train_loader, None],
				None,
				None,
				global_step,
			)
		scheduler_g.step()
		scheduler_d.step()


def cpurun(
	rank,
	n_gpus,
	save_dir="/root/logdir/songsuro",
	seed=1234,
	lr_decay=0.998,
	epochs=1000,
	train_dataset=None,
	valid_dataset=None,
	batch_size=2,
):
	if rank == 0:
		check_git_hash(save_dir)
		writer = SummaryWriter(log_dir=save_dir)
		writer_eval = SummaryWriter(log_dir=os.path.join(save_dir, "eval"))
	torch.manual_seed(seed)

	train_loader = BaseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = BaseDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

	net_g = Autoencoder()
	mpd = MultiPeriodDiscriminator()
	msd = MultiScaleDiscriminator()

	optim_g = torch.optim.AdamW(
		net_g.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01
	)
	optim_d = torch.optim.AdamW(
		list(mpd.parameters()) + list(msd.parameters()),
		lr=2e-4,
		betas=(0.8, 0.99),
		weight_decay=0.01,
	)

	try:
		_, _, _, epoch_str = load_checkpoint(
			latest_checkpoint_path(save_dir, "G_*.pth"), net_g, optim_g
		)
		_, _, _, epoch_str = load_checkpoint(
			latest_checkpoint_path(save_dir, "D_*.pth"), [mpd, msd], [optim_d]
		)
		global_step = (epoch_str - 1) * len(train_loader)
	except:
		epoch_str = 1
		global_step = 0

	scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
		optim_g, gamma=lr_decay, last_epoch=epoch_str - 2
	)
	scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
		optim_d, gamma=lr_decay, last_epoch=epoch_str - 2
	)

	for epoch in range(epoch_str, epochs + 1):
		global_step = train_and_evaluate(
			rank,
			epoch,
			[net_g, mpd, msd],
			[optim_g, optim_d],
			[scheduler_g, scheduler_d],
			[train_loader, valid_loader],
			logger,
			[writer, writer_eval],
			global_step,
		)
		scheduler_g.step()
		scheduler_d.step()


def train_and_evaluate(
	rank,
	epoch,
	nets,
	optims,
	schedulers,
	loaders,
	logger,
	writers,
	global_step,
	lambda_recon=0.1,
	lambda_emb=0.1,
	lambda_fm=0.1,
	log_interval=2000,
	eval_interval=20000,
	learning_rate=2e-4,
	save_dir="/root/logdir/songsuro",
	hop_size=512,
):
	net_g, mpd, msd = nets
	optim_g, optim_d = optims
	scheduler_g, scheduler_d = schedulers
	train_loader, eval_loader = loaders
	if writers is not None:
		writer, writer_eval = writers

	if hasattr(train_loader, "sampler"):
		train_loader.sampler.set_epoch(epoch)

	net_g.train()
	mpd.train()
	msd.train()

	for batch_idx, data_dict in enumerate(train_loader):
		mel = data_dict["mel_spectrogram"]
		wav = data_dict["wav"]

		y_hat, commit_loss = net_g(mel)
		gt = wav

		# Discriminator training
		y_df_hat_r, y_df_hat_g, _, _ = mpd(gt, y_hat.detach())
		loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
		y_ds_hat_r, y_ds_hat_g, _, _ = msd(gt, y_hat.detach())
		loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
		loss_disc_all = loss_disc_s + loss_disc_f

		optim_d.zero_grad()
		loss_disc_all.backward()
		clip_grad_value(list(mpd.parameters()) + list(msd.parameters()), None)
		optim_d.step()

		# Generator training
		y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(gt, y_hat)
		y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(gt, y_hat)

		loss_gen_f, _ = generator_loss(y_df_hat_g)
		loss_gen_s, _ = generator_loss(y_ds_hat_g)
		loss_adv = loss_gen_f + loss_gen_s

		loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * lambda_fm
		loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * lambda_fm
		loss_fm = loss_fm_f + loss_fm_s

		loss_recon = reconstruction_loss(gt, y_hat) * lambda_recon
		loss_emb = commit_loss * lambda_emb

		loss_gen_all = loss_adv + loss_fm + loss_recon + loss_emb

		optim_g.zero_grad()
		loss_gen_all.backward()
		clip_grad_value(net_g.parameters(), None)
		optim_g.step()

		if rank == 0:
			if global_step % log_interval == 0:
				lr = optim_g.param_groups[0]["lr"]
				logger.info(
					"Train Epoch: {} [{:.0f}%]".format(
						epoch, 100.0 * batch_idx / len(train_loader)
					)
				)
				logger.info(
					f"Total: {loss_gen_all.item():.3f}, "
					f"Adv: {loss_adv.item():.3f}, "
					f"FM: {loss_fm.item():.3f}, "
					f"Recon: {loss_recon.item():.3f}, "
					f"Emb: {loss_emb.item():.3f}, "
					f"Step: {global_step}, LR: {lr:.6f}"
				)
				scalar_dict = {
					"loss/total": loss_gen_all,
					"loss/adv": loss_adv,
					"loss/fm": loss_fm,
					"loss/recon": loss_recon,
					"loss/emb": loss_emb,
				}
				summarize(writer, global_step, scalars=scalar_dict)

			if global_step % eval_interval == 0:
				logger.info(["All training params(G): ", count_parameters(net_g), " M"])
				evaluate(net_g, eval_loader, writer_eval, global_step)
				save_checkpoint(
					net_g,
					optim_g,
					learning_rate,
					epoch,
					os.path.join(save_dir, "G_{}.pth".format(global_step)),
				)
				save_checkpoint(
					[mpd, msd],
					optim_d,
					learning_rate,
					epoch,
					os.path.join(save_dir, "D_{}.pth".format(global_step)),
				)
				net_g.train()

		global_step += 1

	if rank == 0:
		logger.info("====> Epoch: {}".format(epoch))
	return global_step


def evaluate(autoencoder, eval_loader, writer_eval, global_step, sample_rate=44100):
	autoencoder.eval()
	with torch.no_grad():
		for batch_idx, data_dict in enumerate(eval_loader):
			mel = data_dict["mel"]
			wav = data_dict["wav"]
			wav_lengths = data_dict["wav_lengths"]

			if use_cuda:
				wav = wav.cuda(0)
				mel = mel.cuda(0)

			wav = wav[:1]
			mel = mel[:1]

			y_hat, _ = autoencoder.sample(mel)
			y_hat_np = y_hat.squeeze(0).cpu().numpy()
			y_hat_mel = make_mel_spectrogram_from_audio(y_hat_np)

			image_dict = {
				"gen/mel": plot_spectrogram_to_numpy(y_hat_mel),
				"gt/mel": plot_spectrogram_to_numpy(mel),
			}
			audio_dict = {
				"gen/audio": torch.FloatTensor(y_hat_np).unsqueeze(0),
				"gt/audio": wav[:, :, : wav_lengths[0]],
			}

			summarize(
				writer=writer_eval,
				global_step=global_step,
				images=image_dict,
				audios=audio_dict,
				audio_sampling_rate=sample_rate,
			)
			break
	autoencoder.train()


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
