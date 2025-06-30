from pathlib import Path
from typing import Union

import pytorch_lightning as pl

from songsuro.data.dataset.aihub_legacy import AIHubLegacyDataset
from songsuro.data.loader.base import BaseDataLoader


class SongsuroDataModule(pl.LightningDataModule):
	def __init__(
		self,
		train_root_dir: Union[str, Path],
		val_root_dir: Union[str, Path],
		batch_size: int,
		num_workers: int,
	):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_root_dir = train_root_dir
		self.val_root_dir = val_root_dir

	def setup(self, stage):
		if stage == "fit":
			self.train_dataset = AIHubLegacyDataset(self.train_root_dir)
			self.val_dataset = AIHubLegacyDataset(self.val_root_dir)
		if stage == "test":
			# TODO: Implement this with our own test dataset at Feature/#49
			raise NotImplementedError

	def train_dataloader(self):
		return BaseDataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=True,
		)

	def val_dataloader(self):
		return BaseDataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=False,
		)

	def test_dataloader(self):
		# TODO: Implement this with our own test dataset at Feature/#49
		raise NotImplementedError
