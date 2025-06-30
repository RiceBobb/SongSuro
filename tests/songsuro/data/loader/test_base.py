import os
import pathlib

import torch
import pytest

from songsuro.data.dataset.aihub_legacy import AIHubLegacyDataset
from songsuro.data.loader.base import BaseDataLoader


root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
data_dir = os.path.join(root_dir, "resources", "ai_hub_legacy_data_sample")


class TestBaseDataLoader:
	@pytest.fixture
	def dataset(self):
		return AIHubLegacyDataset(data_dir)

	def test_initialization(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		assert dataloader.pad_mode == "constant"
		assert dataloader.pad_value == 0
		assert dataloader.batch_size == 2

	def test_padding_function(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		# Test padding with a simple tensor
		audio = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
		padded = dataloader._pad_tensor(audio, max_length=10)

		assert padded.shape == (3, 10)
		assert torch.equal(padded[:, :2], audio)
		assert torch.all(padded[:, 2:] == 0)

		audio2 = torch.tensor(
			[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
		)
		padded = dataloader._pad_tensor(audio2, max_length=10)

		assert padded.shape == (2, 3, 10)

	def test_batch_creation(self, dataset):
		dataloader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

		for batch in dataloader:
			assert "audio" in batch
			assert "audio_lengths" in batch
			assert "sample_rates" in batch
			assert "f0" in batch
			assert "metadata" in batch
			assert "f0_lengths" in batch
			assert "mel_spectrogram" in batch

			# Check shapes and values for first batch
			assert batch["audio"].shape[0] == 2  # batch size
			assert batch["audio"].dim() == 2
			assert batch["audio_lengths"].shape[0] == 2
			assert batch["audio_lengths"].dim() == 1
			assert batch["sample_rates"].shape[0] == 2
			assert batch["sample_rates"].dim() == 1

			assert batch["mel_spectrogram"].dim() == 3
			assert batch["mel_spectrogram"].shape[0] == 2
			assert batch["mel_spectrogram"].shape[1] == 128

			assert batch["f0"].dim() == 2
			assert batch["f0"].shape[0] == 2

			# Verify padding in first batch
			assert batch["audio"][0, -1] == 0 or batch["audio"][1, -1] == 0
