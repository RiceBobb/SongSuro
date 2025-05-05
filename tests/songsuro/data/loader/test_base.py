import os
import pathlib

import torch
import pytest

from songsuro.data.dataset.aihub import AIHubDataset
from songsuro.data.loader.base import BaseDataLoader


root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
data_dir = os.path.join(root_dir, "resources", "ai_hub_data_sample")


class TestBaseDataLoader:
	@pytest.fixture
	def dataset(self):
		return AIHubDataset(data_dir)

	def test_initialization(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		assert dataloader.pad_mode == "constant"
		assert dataloader.pad_value == -1
		assert dataloader.batch_size == 2

	def test_padding_function(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		# Test padding with a simple tensor
		audio = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
		padded = dataloader._pad_audio(audio, max_length=10)

		assert padded.shape == (3, 10)
		assert torch.equal(padded[:, :2], audio)
		assert torch.all(padded[:, 2:] == -1)

	def test_batch_creation(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		batches = list(dataloader)
		assert len(batches) == 3

		# Test first batch
		batch1 = batches[0]
		assert "audio" in batch1
		assert "audio_lengths" in batch1
		assert "sample_rates" in batch1
		assert "f0" in batch1
		assert "metadata" in batch1

		# Check shapes and values for first batch
		assert batch1["audio"].shape[0] == 2  # batch size
		assert batch1["audio"].shape[1] == 1  # mono audio
		assert batch1["audio"].dim() == 3
		assert batch1["audio_lengths"].shape[0] == 2
		assert batch1["audio_lengths"].dim() == 1
		assert batch1["sample_rates"].shape[0] == 2
		assert batch1["sample_rates"].dim() == 1

		# Verify padding in first batch
		assert batch1["audio"][0, 0, -1] == -1 or batch1["audio"][1, 0, -1] == -1

		batch2 = batches[1]
		assert batch2["audio"].shape[0] == 2  # batch size
		assert batch2["audio"].shape[1] == 1  # mono audio
		assert batch2["audio"].dim() == 3
		assert batch2["audio_lengths"].shape[0] == 2
		assert batch2["audio_lengths"].dim() == 1
		assert batch2["sample_rates"].shape[0] == 2
		assert batch2["sample_rates"].dim() == 1

		# Verify padding in first batch
		assert batch2["audio"][0, 0, -1] == -1 or batch2["audio"][1, 0, -1] == -1
