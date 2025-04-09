import os
import pathlib

import pytest
import numpy as np
import soundfile as sf
from unittest.mock import patch

from songsuro.preprocess import (
	load_audio,
	make_spectrogram,
	make_mel_spectrogram,
	make_mel_spectrogram_from_audio,
	extract_f0_from_file,
	synthesize_audio_from_f0,
	hz_to_mel,
	quantize_mel_scale,
	mode_window_filter,
	detect_silence,
	convert_g2p,
)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")

KO_MANGGASONG = """
맨까 새끼들 부들부들하구나
억까를 해봐도 우린 골 넣지
니네가 아무리 맹구다 어쩐다고 놀려도
아아~ 즐겁구나 명 절 이~(짜스!)[9]

맨까 새끼들 부들부들하구나
이번 시즌 잘하잖아 억까하지 마
이겨도 지롤 져도 지롤 뭐만 하면 리그컵
아아~ 리그컵도 축 군 데~(컴온!!)

맨까 새끼들 부들부들하구나
돌아온 미친 폼 누가 막을래?
더보기 리그 탈출직전 돌아와요 맨유 팬
아아~ 기대된다 챔 스 가~ Siuuuuuuu![원곡]
"""

KO_LIJUNGDDAK = """
안하긴뭘안해~~ 반갑습니다~~
이피엘에서 우승못하는팀 누구야? 소리질러~~!!
리중딱 리중딱 신나는노래~ 나도한번 불러본다~~(박수) (박수) (박수)
짠리잔짠~~ 우리는 우승하기 싫~어~ 왜냐면 우승하기 싫은팀이니깐~
20년 내~내~ 프리미어리그~ 우승도 못하는 우리팀이다. 리중딱 리중딱 신나는노래 ~~~
나도한번불러본다~ 리중딱 리중딱 신나는노래 ~~
가슴치며 불러본다~ 리중딱 노래가사는~ 생활과 정보가 있는노래 중딱이~~와 함께라면 제~라드도함께 우승못한다.
"""

EN_DIEWITHSMILE = """
I, I just woke up from a dream
Where you and I had to say goodbye
And I don't know what it all means
But since I survived, I realized

Wherever you go, that's where I'll follow
Nobody's promised tomorrow
So I'ma love you every night like it's the last night
Like it's the last night

If the world was ending
I'd wanna be next to you
If the party was over
And our time on Earth was through
I'd wanna hold you just for a while
And die with a smile
If the world was ending
I'd wanna be next to you

Ooh, lost, lost in the words that we scream
I don't even wanna do this anymore
'Cause you already know what you mean to me
And our love's the only war worth fighting for

Wherever you go, that's where I'll follow
Nobody's promised tomorrow
So I'ma love you every night like it's the last night
Like it's the last night
"""


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

	def test_make_mel_spectrogram_from_audio(self, sample_audio_file):
		"""Test creating mel spectrogram from raw audio data."""
		audio_data = load_audio(sample_audio_file)
		n_fft = 2048
		hop_length = 512
		sr = 24000
		n_bins = 128
		fmax = 8000

		mel_spec = make_mel_spectrogram_from_audio(
			audio_data, n_fft, hop_length, sr, n_bins, fmax
		)

		# Check mel spectrogram properties
		assert isinstance(mel_spec, np.ndarray)
		assert mel_spec.ndim == 2
		assert mel_spec.shape[0] == n_bins
		assert mel_spec.shape[1] > 0
		assert np.all(mel_spec >= 0)

	def test_extract_f0_from_file(self, sample_audio_file):
		# Test if the function runs without errors
		pitch_values, fs = extract_f0_from_file(sample_audio_file)

		# Check if pitch_values and fs are not None
		assert pitch_values is not None
		assert fs is not None

		# Check if pitch_values is a numpy array
		assert isinstance(pitch_values, np.ndarray)

		# Check if fs is an integer
		assert isinstance(fs, int)

		# Check if pitch_values has the same length as the audio file
		audio_length = len(sf.read(sample_audio_file)[0])
		assert len(pitch_values) == audio_length
		assert 0 in np.unique(pitch_values)

	def test_synthesize_audio_from_f0(self, sample_audio_file, tmp_path):
		# Extract F0 from the sample file
		pitch_values, fs = extract_f0_from_file(sample_audio_file)

		# Test synthesize_audio_from_f0 without saving
		synthesized = synthesize_audio_from_f0(pitch_values, fs)

		# Check if synthesized is a numpy array
		assert isinstance(synthesized, np.ndarray)

		# Check if synthesized has the same length as pitch_values
		assert len(synthesized) == len(pitch_values)

		# Test synthesize_audio_from_f0 with saving
		save_path = tmp_path / "synthesized_audio.wav"
		_ = synthesize_audio_from_f0(pitch_values, fs, str(save_path))

		# Check if the file was created
		assert save_path.exists()

		# Check if the saved file has the correct sample rate and length
		saved_audio, saved_fs = sf.read(str(save_path))
		assert saved_fs == fs
		assert len(saved_audio) == len(pitch_values)

	def test_extract_f0_from_file_not_found(self):
		# Test if FileNotFoundError is raised for non-existent file
		with pytest.raises(FileNotFoundError):
			extract_f0_from_file("non_existent_file.wav")

	def test_synthesize_audio_from_f0_zero_pitch(self, tmp_path):
		# Test synthesize_audio_from_f0 with zero pitch values
		pitch_values = np.zeros(1000)
		fs = 44100
		save_path = tmp_path / "zero_pitch_audio.wav"

		synthesized = synthesize_audio_from_f0(pitch_values, fs, str(save_path))

		# Check if synthesized audio is all zeros
		assert np.allclose(synthesized, np.zeros_like(synthesized))

		# Check if the file was created and contains all zeros
		assert save_path.exists()
		saved_audio, _ = sf.read(str(save_path))
		assert np.allclose(saved_audio, np.zeros_like(saved_audio))

	def test_quantized_f0(self, sample_audio_file):
		# Extract F0 from the sample file
		pitch_values, fs = extract_f0_from_file(sample_audio_file)

		mel_pitch_values = hz_to_mel(pitch_values)

		quantized_f0 = quantize_mel_scale(mel_pitch_values)

		assert isinstance(quantized_f0, np.ndarray)
		assert quantized_f0.ndim == 1
		assert quantized_f0.shape[0] == pitch_values.shape[0]

	def test_quantized_f0_sample(self):
		sample_pitch_values = np.array([132, 572, 0, 349, 0])
		quantized_f0 = quantize_mel_scale(sample_pitch_values)
		assert isinstance(quantized_f0, np.ndarray)
		assert np.allclose(np.array([1, 127, 0, 63, 0]), quantized_f0)

	def test_mode_window_filter(self, sample_audio_file):
		pitch_values, fs = extract_f0_from_file(sample_audio_file)

		mel_pitch_values = hz_to_mel(pitch_values)

		quantized_f0 = quantize_mel_scale(mel_pitch_values)

		frame_duration_ms = 20
		indices_per_frame = int((frame_duration_ms / 1000) * fs)
		frame_quantized_f0 = mode_window_filter(quantized_f0, indices_per_frame)

		assert isinstance(frame_quantized_f0, np.ndarray)
		assert frame_quantized_f0.ndim == 1
		assert pitch_values.shape[0] / fs == pytest.approx(
			frame_quantized_f0.shape[0] * 20 / 1000, rel=0.02
		)
		assert 0 in np.unique(frame_quantized_f0)

	def test_hz_to_mel(self):
		frequency = np.array([0, 6300, 0, 6300])
		mel_value = hz_to_mel(frequency)
		assert np.allclose(np.array([0, 2595, 0, 2595]), mel_value)

	def test_detect_silence(self, sample_audio_file):
		pitch_values, fs = extract_f0_from_file(sample_audio_file)
		loaded_audio = load_audio(sample_audio_file, fs)
		has_sound = detect_silence(loaded_audio)
		has_sound_resampled = (
			np.interp(
				np.linspace(0, 1, len(pitch_values)),
				np.linspace(0, 1, len(has_sound)),
				has_sound.astype(float),
			)
			> 0.5
		)

		assert len(has_sound_resampled) == pitch_values.shape[0]

	def test_detect_silence_zero_audio(self):
		zero_audio = np.zeros(1000)
		has_sound = detect_silence(zero_audio)
		assert np.allclose(np.zeros_like(has_sound), has_sound)

	def test_convert_g2p(self):
		ko_lyrics = [KO_LIJUNGDDAK, KO_MANGGASONG]
		en_lyrics = [EN_DIEWITHSMILE]

		phonemes_en = convert_g2p(en_lyrics, language="en")
		phonemes_ko = convert_g2p(ko_lyrics, language="ko")

		self.validate_phonemes(phonemes_ko, ko_lyrics)
		self.validate_phonemes(phonemes_en, en_lyrics)

	def test_convert_g2p_invalid_language(self):
		lyrics = ["Hello world"]
		with pytest.raises(ValueError, match="Unsupported language: es"):
			convert_g2p(lyrics, language="es")

	def test_convert_g2p_empty_lst_logs_warning(self):
		with patch("logging.warning") as mock_warning:
			convert_g2p([])  # 빈 리스트 전달
			mock_warning.assert_called_once_with(
				"No phonemes were generated. Please check the input lyrics."
			)

	def validate_phonemes(self, phonemes, lyrics):
		assert isinstance(phonemes, list), "Phonemes should be a list."
		assert len(phonemes) == len(
			lyrics
		), "Phonemes length should match lyrics length."
		assert all(
			isinstance(p, str) for lst in phonemes for p in lst
		), "All phonemes should be strings."
