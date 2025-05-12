import os
from datetime import datetime
from pathlib import Path
from typing import Union

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from songsuro.data.module import SongsuroDataModule
from songsuro.models import Songsuro


@click.command()
@click.option(
	"--train_root_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option(
	"--val_root_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=6)
@click.option(
	"--autoencoder_checkpoint_path",
	type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option("--checkpoint_path", type=click.Path(dir_okay=False, file_okay=True))
def main(
	train_root_dir: Union[str, Path],
	val_root_dir: Union[str, Path],
	batch_size: int,
	num_workers: int,
	autoencoder_checkpoint_path: Union[Path, str],
	checkpoint_path: Union[str, Path],
):
	if not os.path.exists(str(checkpoint_path)):
		os.makedirs(str(checkpoint_path))

	data = SongsuroDataModule(
		train_root_dir, val_root_dir, batch_size=batch_size, num_workers=num_workers
	)

	if os.path.exists(str(checkpoint_path)):
		model = Songsuro.load_from_checkpoint(checkpoint_path)
	else:
		model = Songsuro(80, 192, autoencoder_checkpoint_path)

	tqdm_cb = TQDMProgressBar(refresh_rate=10)
	ckpt_cb = ModelCheckpoint(
		dirpath=checkpoint_path,
		filename="{epoch:02d}-{step}-{val_loss:.2f}",
		save_last=True,
		every_n_epochs=1,
	)
	wandb_logger = WandbLogger(name=str(datetime.now()), project="Songsuro-all")
	early_stop_callback = EarlyStopping(
		monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
	)

	trainer = pl.Trainer(
		accelerator="auto",
		max_epochs=2,
		log_every_n_steps=8,
		logger=wandb_logger,
		callbacks=[tqdm_cb, ckpt_cb, early_stop_callback],
		check_val_every_n_epoch=1,
		gradient_clip_val=1e9,
	)
	trainer.fit(model, data)


if __name__ == "__main__":
	main()
