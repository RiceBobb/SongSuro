import os
from datetime import datetime
from pathlib import Path
from typing import Union

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.strategies import FSDPStrategy

from songsuro.autoencoder.models import Autoencoder
from songsuro.data.module import SongsuroDataModule


def train(
	train_root_dir: Union[str, Path],
	val_root_dir: Union[str, Path],
	batch_size: int,
	num_workers: int,
	checkpoint_path: Union[str, Path],
):
	data = SongsuroDataModule(
		train_root_dir, val_root_dir, batch_size=batch_size, num_workers=num_workers
	)

	if not os.path.isdir(checkpoint_path):
		model = Autoencoder.load_from_checkpoint(checkpoint_path)
		checkpoint_dir = os.path.dirname(checkpoint_path)
		print("Load Complete")
	else:
		print("New Model Train")
		model = Autoencoder()
		checkpoint_dir = checkpoint_path

	tqdm_cb = TQDMProgressBar(refresh_rate=10)
	ckpt_cb = ModelCheckpoint(
		dirpath=checkpoint_dir,
		filename="{epoch:02d}-{step}-{val_loss:.2f}",
		save_last=True,
		every_n_epochs=1,
	)
	wandb_logger = WandbLogger(name=str(datetime.now()), project="Songsuro-autoencoder")
	early_stop_callback = EarlyStopping(
		monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
	)

	trainer = pl.Trainer(
		accelerator="cuda",
		max_epochs=2,
		logger=wandb_logger,
		callbacks=[tqdm_cb, ckpt_cb, early_stop_callback],
		check_val_every_n_epoch=1,
		log_every_n_steps=1,
		precision="bf16-true",
		devices=2,
		strategy="fsdp",
		num_sanity_val_steps=0,
	)
	trainer.fit(model, data)


@click.command()
@click.option(
	"--train_root_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option(
	"--val_root_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--batch_size", type=int, default=4)
@click.option("--num_workers", type=int, default=6)
@click.option("--checkpoint_path", type=click.Path(dir_okay=True, file_okay=True))
def cli(
	train_root_dir: Union[str, Path],
	val_root_dir: Union[str, Path],
	batch_size: int,
	num_workers: int,
	checkpoint_path: Union[str, Path],
):
	train(train_root_dir, val_root_dir, batch_size, num_workers, checkpoint_path)


if __name__ == "__main__":
	cli()
