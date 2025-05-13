import os

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from librosa import feature as lf
from scipy import stats


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


def make_mel_spectrogram_from_audio(
	audio_np,
	n_fft: int = 2048,
	hop_length: int = 1024,
	sr: int = 24_000,
	n_bins: int = 128,
	fmax: int = 8000,
):
	"""
	Generate mel spectrogram from raw audio data.

	:param audio_np: Raw audio data as a NumPy array
	:param n_fft: Number of FFT components
	:param hop_length: Hop length for STFT
	:param sr: Sampling rate of the audio
	:param n_bins: Number of mel bins
	:param fmax: Maximum frequency for mel filterbank
	:return: Mel spectrogram as a NumPy array
	"""
	# Create spectrogram using STFT
	spec = make_spectrogram(audio_np, n_fft=n_fft, hop_length=hop_length)

	# Convert spectrogram to mel scale
	mel_spec = make_mel_spectrogram(spec, sr=sr, n_bins=n_bins, fmax=fmax)

	return mel_spec


def extract_f0_from_file(filepath: str, silence_threshold_db: int = -50):
	"""
	Extract F0 from the input audio file

	:param filepath: The path to the audio file.
	:param silence_threshold_db: The silence threshold in decibel.
	    Default is -50.
	:return: Pitch_values which is F0 and fs which is the sampling rate of input audio.
	    In the output pitch_values, the zero value represents the 'non-voice' value.
	"""
	if not os.path.exists(filepath):
		raise FileNotFoundError(filepath)

	signal = basic.SignalObj(filepath)
	# Extract pitch using the YAAPT algorithm
	pitch = pYAAPT.yaapt(signal)
	# Upsampled pitch data (Same length with original audio)
	pitch_values = pitch.values_interp
	fs = signal.fs

	has_sound = detect_silence(signal.data, threshold_db=silence_threshold_db)
	has_sound_resampled = (
		np.interp(
			np.linspace(0, 1, len(pitch_values)),
			np.linspace(0, 1, len(has_sound)),
			has_sound.astype(float),
		)
		> 0.5
	)

	pitch_values_masked = pitch_values.copy()
	pitch_values_masked[~has_sound_resampled] = 0

	return pitch_values_masked, int(fs)


def extract_f0_from_tensor(
	waveform: torch.Tensor, sample_rate: int, silence_threshold_db: int = -50
):
	f0 = torchaudio.functional.detect_pitch_frequency(
		waveform, sample_rate=sample_rate, frame_time=0.02
	)  # default frame time = 0.01
	return f0


def detect_silence(audio_signal, frame_length=1024, hop_length=512, threshold_db=-40):
	"""
	Distinguishes between silent and non-silent segments in an audio signal.

	:param audio_signal: Array representing the audio signal
	:param frame_length: Frame length
	:param hop_length: Step size between frames
	:param threshold_db: Threshold in decibels (dB) for detecting silence
	:return: Boolean array indicating whether each frame contains sound (True) or is silent (False)
	"""
	from librosa import feature as lf

	# Compute RMS energy
	rms = lf.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]

	# Convert to decibels
	db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)

	# Classify frames based on the threshold_db
	has_sound = db > threshold_db

	return has_sound


def trim_silence(waveform: torch.Tensor, sample_rate: int):
	trimmed_front = torchaudio.functional.vad(waveform, sample_rate=sample_rate)
	trimmed_both = torch.flip(
		torchaudio.functional.vad(torch.flip(trimmed_front, [-1]), sample_rate), [-1]
	)
	return trimmed_both


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
	# Clip non-zero values and leave zeros as is
	clipped_values = np.where(
		mel_pitch_values != 0, np.clip(mel_pitch_values, min_val, max_val), 0
	)

	# Quantize non-zero values and leave zeros as is
	quantized_values = np.where(
		clipped_values != 0,
		np.round((clipped_values - min_val) / (max_val - min_val) * (levels - 1) + 1),
		0,
	)

	return quantized_values.astype(int)


def quantize_mel_scale_torch(mel_pitch_values, levels=127, min_val=133, max_val=571):
	# Clip non-zero values and leave zeros as is
	clipped_values = torch.where(
		mel_pitch_values != 0,
		torch.clamp(mel_pitch_values, min_val, max_val),
		torch.tensor(0, dtype=mel_pitch_values.dtype),
	)

	# Quantize non-zero values and leave zeros as is
	quantized_values = torch.where(
		clipped_values != 0,
		torch.round(
			(clipped_values - min_val) / (max_val - min_val) * (levels - 1) + 1
		),
		torch.tensor(0, dtype=mel_pitch_values.dtype),
	)

	return quantized_values.to(torch.int64)


def hz_to_mel(frequency: np.ndarray):
	"""
	Change hz to the mel scale.
	If the frequency value is 0 (silent), returns 0.
	"""
	return np.where(frequency != 0, 2595 * np.log10(1 + frequency / 700), 0)


def hz_to_mel_torch(frequency: torch.Tensor):
	"""
	Change hz to the mel scale using torch.
	If the frequency value is 0 (silent), returns 0.
	"""
	return torch.where(
		frequency != 0, 2595 * torch.log10(1 + frequency / 700), torch.tensor(0.0)
	)


def mode_window_filter(arr: np.ndarray, window_size: int):
	valid_length = len(arr) - (len(arr) % window_size)
	arr_trimmed = arr[:valid_length]

	frames = arr_trimmed.reshape(-1, window_size)
	filtered_audio, _ = stats.mode(frames, axis=1, keepdims=False)
	return filtered_audio


# Preprocess F0 : extract F0 => hz_to_mel => quantize_mel_scale => hz to frame (최빈값 필터)
