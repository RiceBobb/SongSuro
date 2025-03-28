import os

import librosa
import numpy as np
from librosa import feature as lf
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import soundfile as sf


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


def extract_f0_from_file(filepath: str):
	"""
	Extract F0 from audio file

	:param filepath: The path to the audio file.
	:return: Pitch_values which is F0 and fs which is the sampling rate of input audio.
	"""
	if not os.path.exists(filepath):
		raise FileNotFoundError(filepath)

	signal = basic.SignalObj(filepath)
	# Extract pitch using YAAPT algorithm
	pitch = pYAAPT.yaapt(signal)
	# Upsampled pitch data (Same length with original audio)
	pitch_values = pitch.values_interp
	fs = signal.fs
	return pitch_values, int(fs)


def synthesize_audio_from_f0(pitch_values, fs: int, save_path: str = None):
	# 음정 데이터를 이용해 위상 누적 계산: phi[n] = phi[n-1] + 2*pi*(f[n]/fs)
	phase = np.cumsum(2 * np.pi * pitch_values / fs)
	# 합성 음성 생성: 각 시점의 주파수를 반영하는 사인파
	synthesized = np.sin(phase)
	# 음정이 0인 (비음성) 구간은 무음으로 처리
	synthesized[pitch_values == 0] = 0

	if save_path is not None:
		sf.write(save_path, synthesized, fs)

	return synthesized


def quantize_mel_scale(mel_pitch_values, levels=127, min_val=133, max_val=571):
	clipped_values = np.clip(mel_pitch_values, min_val, max_val)
	quantized_values = np.round(
		(clipped_values - min_val) / (max_val - min_val) * (levels - 1)
	)
	return quantized_values.astype(int)


def hz_to_mel(frequency):
	return 2595 * np.log10(1 + frequency / 700)


# Preprocess F0 : extract F0 => hz_to_mel => quantize_mel_scale => hz to frame (최빈값 필터) => one-hot encoding (0~127)
