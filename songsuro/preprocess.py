import os

import librosa
import numpy as np
from librosa import feature as lf


def load_audio(path, sr: int = 24_000):
	"""
	Safely load audio from path and downsample to sr.

	:param path: The path to the audio file.
	:param sr: The sampling rate.
	:return: audio instance load by librosa.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(path)

	audio_data, _ = librosa.load(path, sr=sr)
	return audio_data


def make_spectrogram(audio_data, n_fft: int = 2048, hop_length: int = 1024):
	x = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
	result = np.abs(x) ** 2  # Change to power spectrogram
	return result


def make_mel_spectrogram(
	power_spectrogram, sr: int = 24_000, n_bins: int = 128, fmax: int = 8000
):
	mel_spectrogram = lf.melspectrogram(
		S=power_spectrogram, sr=sr, n_mels=n_bins, fmax=fmax
	)
	return mel_spectrogram
