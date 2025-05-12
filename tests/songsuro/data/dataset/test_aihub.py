import pathlib

import pytest
import os
import torch

from songsuro.data.dataset.aihub import AIHubDataset


root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
data_dir = os.path.join(root_dir, "resources", "ai_hub_data_sample")


class TestAIHubDataset:
	@pytest.fixture
	def dataset(self):
		# Use the existing root_dir variable that contains the dataset
		return AIHubDataset(data_dir)

	def test_dataset_initialization(self, dataset):
		"""Test that the dataset is initialized correctly."""
		assert isinstance(dataset, AIHubDataset)
		assert dataset.root_dir == data_dir
		assert len(dataset.wav_file_list) > 0
		assert all(f.endswith(".wav") for f in dataset.wav_file_list)

	def test_dataset_length(self, dataset):
		"""Test that the dataset length matches the number of wav files."""
		assert len(dataset) == len(dataset.wav_file_list)

	def test_getitem(self, dataset):
		"""Test that __getitem__ returns the correct structure."""
		# Get the first item
		item = dataset[0]

		# Check the return type
		assert isinstance(item, dict)

		# Check audio tensor
		assert isinstance(item["audio"], torch.Tensor)
		assert item["audio"].dim() == 2  # [channels, time]

		assert item["sample_rate"] == 44_100

		# Check mel spectrogram
		assert isinstance(item["mel_spectrogram"], torch.Tensor)
		assert item["mel_spectrogram"].dim() == 3  # [channels, mels, time]

		# Check filepaths
		assert os.path.exists(item["audio_filepath"])
		assert os.path.exists(item["label_filepath"])
		assert item["audio_filepath"].endswith(".wav")
		assert item["label_filepath"].endswith(".json")

		# Check lyrics
		assert isinstance(item["lyrics"], str)

		# Check F0
		assert isinstance(item["f0"], torch.Tensor)
		assert item["f0"].dim() == 1  # [length]

		# Check metadata
		assert isinstance(item["metadata"], dict)
		assert "gender" in item["metadata"]
		assert "age_group" in item["metadata"]
		assert "genre" in item["metadata"]
		assert "timbre" in item["metadata"]
		assert "singer_id" in item["metadata"]

	def test_extract_metadata_from_path(self, dataset):
		"""Test metadata extraction from filepath."""
		# Get a sample filepath
		sample_filepath = dataset.wav_file_list[0]

		# Extract metadata
		metadata = dataset.extract_metadata_from_path(sample_filepath)

		# Check metadata structure
		assert isinstance(metadata, dict)
		assert "root_type" in metadata
		assert "gender" in metadata
		assert "age_group" in metadata
		assert "genre" in metadata
		assert "timbre" in metadata
		assert "singer_id" in metadata
		assert "filepath" in metadata

		# Check that the filepath in metadata matches the relative path
		rel_path = os.path.relpath(sample_filepath, start=dataset.root_dir)
		assert metadata["filepath"] == rel_path

	def test_multiple_items(self, dataset):
		"""Test getting multiple items from the dataset."""
		# Test at least 3 items if available
		num_items = min(3, len(dataset))

		for i in range(num_items):
			item = dataset[i]
			assert isinstance(item, dict)
			assert os.path.exists(item["audio_filepath"])
			assert os.path.exists(item["label_filepath"])
