import torch
from torch.utils.data import DataLoader

from songsuro.dataset.base import BaseDataset


class AutoEncoderDataset(BaseDataset):
	def __init__(
		self,
		num_replicas=1,
		rank=1,
		batch_size=8,
		# Not Complete
		train_dataset=None,
		valid_dataset=None,
	):
		super().__init__()
		self.num_replicas = num_replicas
		self.rank = rank
		self.batch_size = batch_size
		self._train_dataset = train_dataset
		self._valid_dataset = valid_dataset

	def _collate_fn(self, batch):
		# 아직 로직이 없으니, 그냥 그대로 반환
		return batch

	def _init_data_loaders(self):
		train_sampler = torch.utils.data.distributed.DistributedSampler(
			self._train_dataset,
			num_replicas=self.num_replicas,
			rank=self.rank,
			shuffle=True,
		)

		self.train_loader = DataLoader(
			self._train_dataset,
			num_workers=4,
			shuffle=False,
			batch_size=self.batch_size,
			pin_memory=True,
			drop_last=True,
			collate_fn=self._collate_fn,
			sampler=train_sampler,
		)

		self.valid_loader = DataLoader(
			self._valid_dataset,
			num_workers=1,
			shuffle=False,
			batch_size=1,
			pin_memory=True,
			drop_last=True,
			collate_fn=self._collate_fn,
		)

	def get_train_loader(self):
		return self.train_loader

	def get_valid_loader(self):
		return self.valid_loader
