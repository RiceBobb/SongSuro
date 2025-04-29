import os
import sys
import logging
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from songsuro.dataset.autoencoder import AutoEncoderDataset

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
from songsuro.utils.util import (
	clip_grad_value,
	load_checkpoint,
	summarize,
	get_logger,
	latest_checkpoint_path,
	count_parameters,
	save_checkpoint,
	check_git_hash,
	plot_spectrogram_to_numpy,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.append("../..")

torch.backends.cudnn.benchmark = True
global_step = 0
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def main(
	port=8001,
):
	"""Assume Single Node Multi GPUs Training Only"""
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = str(port)

	if torch.cuda.is_available():
		n_gpus = torch.cuda.device_count()
		mp.spawn(run, nprocs=n_gpus, args=(n_gpus))
	else:
		cpurun(0, 1)


def run(
	rank,
	n_gpus,
	# train
	save_dir="/root/logdir/songsuro",
	seed=1234,
	lr_decay=0.998,
	epochs=1000,
	# just for commit
	dataset_constructor=None,
):
	global global_step
	if rank == 0:
		logger = get_logger(save_dir)
		check_git_hash(save_dir)
		writer = SummaryWriter(log_dir=save_dir)
		writer_eval = SummaryWriter(log_dir=os.path.join(save_dir, "eval"))

	dist.init_process_group(
		backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
	)
	torch.manual_seed(seed)
	torch.cuda.set_device(rank)

	# Load dataset
	dataset_constructor = AutoEncoderDataset(num_replicas=n_gpus, rank=rank)
	train_loader = dataset_constructor.get_train_loader()
	if rank == 0:
		valid_loader = dataset_constructor.get_valid_loader()

	# Create an Autoencoder instance
	net_g = Autoencoder().cuda(rank)

	# Create HiFi-GAN Discriminator instance
	mpd = MultiPeriodDiscriminator().cuda(rank)
	msd = MultiScaleDiscriminator().cuda(rank)

	# Set AdamW Optimizer
	# from hiddensinger paper
	optim_g = torch.optim.AdamW(
		net_g.parameters(),
		lr=2e-4,
		betas=(0.8, 0.99),
		weight_decay=0.01,
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
			train_and_evaluate(
				rank,
				epoch,
				[net_g, mpd, msd],
				[optim_g, optim_d],
				[scheduler_g, scheduler_d],
				[train_loader, valid_loader],
				logger,
				[writer, writer_eval],
			)
		else:
			train_and_evaluate(
				rank,
				epoch,
				[net_g, mpd, msd],
				[optim_g, optim_d],
				[scheduler_g, scheduler_d],
				[train_loader, None],
				None,
				None,
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
	# just for commit
	dataset_constructor=None,
):
	global global_step
	if rank == 0:
		logger = get_logger(save_dir)
		check_git_hash(save_dir)
		writer = SummaryWriter(log_dir=save_dir)
		writer_eval = SummaryWriter(log_dir=os.path.join(save_dir, "eval"))
	torch.manual_seed(seed)

	# Load Dataset
	dataset_constructor = AutoEncoderDataset(num_replicas=n_gpus, rank=rank)

	train_loader = dataset_constructor.get_train_loader()
	if rank == 0:
		valid_loader = dataset_constructor.get_valid_loader()

	# Create an Autoencoder instance
	net_g = Autoencoder()

	# Create HiFi-GAN Discriminator instance
	mpd = MultiPeriodDiscriminator()
	msd = MultiScaleDiscriminator()

	# Set AdamW Optimizer
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
		train_and_evaluate(
			rank,
			epoch,
			[net_g, mpd, msd],
			[optim_g, optim_d],
			[scheduler_g, scheduler_d],
			[train_loader, valid_loader],
			logger,
			[writer, writer_eval],
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
	# train
	lambda_recon=0.1,
	lambda_emb=0.1,
	lambda_fm=0.1,
	log_interval=2000,
	eval_interval=20000,
	learning_rate=2e-4,
	save_dir="/root/logdir/songsuro",
	# data
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
	global global_step

	# Set the module to training mode
	net_g.train()
	mpd.train()
	msd.train()

	for batch_idx, data_dict in enumerate(train_loader):
		# 오디오 데이터 로드
		# loader에 mel_spectrogram이 있다고 가정
		mel = data_dict["mel_spectrogram"]
		wav = data_dict["wav"]

		# 오토인코더 순전파 (출력 확장)
		y_hat, commit_loss = net_g(mel)
		gt = wav

		# Decoder-Discriminator 학습
		y_df_hat_r, y_df_hat_g, _, _ = mpd(gt, y_hat.detach())
		loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
		y_ds_hat_r, y_ds_hat_g, _, _ = msd(gt, y_hat.detach())
		loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
		loss_disc_all = loss_disc_s + loss_disc_f

		optim_d.zero_grad()
		loss_disc_all.backward()
		clip_grad_value(list(mpd.parameters()) + list(msd.parameters()), None)
		optim_d.step()

		# Decoder-Generator 학습
		y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(gt, y_hat)
		y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(gt, y_hat)

		# 1. 적대적 손실 (λ 없음)
		loss_gen_f, _ = generator_loss(y_df_hat_g)
		loss_gen_s, _ = generator_loss(y_ds_hat_g)
		loss_adv = loss_gen_f + loss_gen_s

		# 2. 특성 매칭 손실 (lambda_fm 적용)
		loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * lambda_fm
		loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * lambda_fm
		loss_fm = loss_fm_f + loss_fm_s

		# 3. 재구성 손실 (lambda_recon 적용)
		loss_recon = reconstruction_loss(gt, y_hat) * lambda_recon

		# 4. 임베딩 손실 (lambda_emb 적용)
		loss_emb = commit_loss * lambda_emb

		# 최종 손실
		loss_gen_all = loss_adv + loss_fm + loss_recon + loss_emb

		optim_g.zero_grad()
		loss_gen_all.backward()
		clip_grad_value(net_g.parameters(), None)
		optim_g.step()

		if rank == 0:
			if global_step % log_interval == 0:
				lr = optim_g.param_groups[0]["lr"]
				# losses 리스트를 새로운 손실 구조에 맞게 업데이트
				# just for commit
				# losses = [
				# 	loss_gen_all,
				# 	loss_adv,
				# 	loss_fm,
				# 	loss_recon,
				# 	loss_emb,
				# ]

				logger.info(
					"Train Epoch: {} [{:.0f}%]".format(
						epoch, 100.0 * batch_idx / len(train_loader)
					)
				)

				# 기존 loss_mel 제거 + 새로운 항목 추가
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
				evaluate(net_g, eval_loader, writer_eval)
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


def evaluate(
	autoencoder,
	eval_loader,
	writer_eval,
	# data
	sample_rate=44100,
):
	autoencoder.eval()
	with torch.no_grad():
		for batch_idx, data_dict in enumerate(eval_loader):
			mel = data_dict["mel"]
			wav = data_dict["wav"]
			wav_lengths = data_dict["wav_lengths"]

			if use_cuda:
				wav = wav.cuda(0)
				mel = mel.cuda(0)

			# 첫 번째 샘플만 처리
			wav = wav[:1]
			mel = mel[:1]

			# 오디오 생성 (추가된 부분)
			y_hat, y_harm, y_noise = autoencoder.sample(mel)

			# NumPy 변환 (shape 조정)
			y_hat_np = y_hat.squeeze(0).cpu().numpy()

			# 특징 추출
			y_hat_mel = make_mel_spectrogram_from_audio(y_hat_np)

			# 시각화 자료
			image_dict = {
				"gen/mel": plot_spectrogram_to_numpy(y_hat_mel),
				"gt/mel": plot_spectrogram_to_numpy(mel),
			}

			# 오디오 텐서 (차원 복원)
			audio_dict = {
				"gen/audio": torch.FloatTensor(y_hat_np).unsqueeze(0),
				"gt/audio": wav[:, :, : wav_lengths[0]],
			}

			# 로깅
			summarize(
				writer=writer_eval,
				global_step=global_step,
				images=image_dict,
				audios=audio_dict,
				audio_sampling_rate=sample_rate,
			)
			break

	# 다시 training 모드로 전환
	autoencoder.train()


if __name__ == "__main__":
	main()
