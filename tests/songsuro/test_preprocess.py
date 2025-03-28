import os
import pathlib

import pytest
import numpy as np

from songsuro.preprocess import load_audio, make_spectrogram, make_mel_spectrogram

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")


class TestAudioProcessing:
	@pytest.fixture
	def sample_audio_file(self):
		"""Create a temporary sine wave audio file for testing."""
		yield os.path.join(resource_dir, "sample_only_voice.wav")

	def test_load_audio_success(self, sample_audio_file):
		"""Test loading audio successfully."""
		audio_data = load_audio(sample_audio_file)

		# Check if audio data is loaded correctly
		assert isinstance(audio_data, np.ndarray)
		assert len(audio_data) > 0
		assert not np.isnan(audio_data).any()

	def test_load_audio_custom_sr(self, sample_audio_file):
		"""Test loading audio with custom sampling rate."""
		custom_sr = 16000
		audio_data = load_audio(sample_audio_file, sr=custom_sr)

		# Original file is about 34 second at 24000 Hz, so at 16000 Hz it should be shorter
		expected_length = int(16000 * (24000 / 16000)) * 35
		assert len(audio_data) <= expected_length

	def test_load_audio_file_not_found(self):
		"""Test loading non-existent audio file."""
		with pytest.raises(FileNotFoundError):
			load_audio("non_existent_file.wav")

	def test_make_spectrogram(self, sample_audio_file):
		"""Test creating power spectrogram."""
		audio_data = load_audio(sample_audio_file)
		spectrogram = make_spectrogram(audio_data)

		# Check spectrogram properties
		assert isinstance(spectrogram, np.ndarray)
		assert spectrogram.ndim == 2
		assert spectrogram.shape[0] == 1025  # n_fft // 2 + 1 for default n_fft=2048
		assert spectrogram.shape[1] > 0
		assert np.all(spectrogram >= 0)  # Power spectrogram should be non-negative

	def test_make_spectrogram_custom_params(self, sample_audio_file):
		"""Test creating spectrogram with custom parameters."""
		audio_data = load_audio(sample_audio_file)
		n_fft = 1024
		hop_length = 512
		spectrogram = make_spectrogram(audio_data, n_fft=n_fft, hop_length=hop_length)

		# Check spectrogram dimensions with custom parameters
		assert spectrogram.shape[0] == n_fft // 2 + 1

	def test_make_mel_spectrogram(self, sample_audio_file):
		"""Test creating mel spectrogram from power spectrogram."""
		audio_data = load_audio(sample_audio_file)
		power_spec = make_spectrogram(audio_data)
		mel_spec = make_mel_spectrogram(power_spec)

		# Check mel spectrogram properties
		assert isinstance(mel_spec, np.ndarray)
		assert mel_spec.ndim == 2
		assert mel_spec.shape[0] == 128  # Default n_bins
		assert mel_spec.shape[1] == power_spec.shape[1]
		assert np.all(mel_spec >= 0)  # Mel spectrogram should be non-negative

	def test_make_mel_spectrogram_custom_params(self, sample_audio_file):
		"""Test creating mel spectrogram with custom parameters."""
		audio_data = load_audio(sample_audio_file)
		power_spec = make_spectrogram(audio_data)

		sr = 16000
		n_bins = 64
		fmax = 4000
		mel_spec = make_mel_spectrogram(power_spec, sr=sr, n_bins=n_bins, fmax=fmax)

		# Check mel spectrogram dimensions with custom parameters
		assert mel_spec.shape[0] == n_bins
		assert mel_spec.shape[1] == power_spec.shape[1]
