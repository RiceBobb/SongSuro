import pytest
import torch
import torchaudio
import numpy as np
from songsuro.evaluation.melody_evaluator import MelodyLoss


# 테스트용 오디오 생성 함수
def generate_sine_wave(duration=1.0, freq=440, sample_rate=16000, noise_level=0.0):
	t = torch.linspace(0, duration, int(sample_rate * duration))
	signal = 0.5 * torch.sin(2 * np.pi * freq * t)
	if noise_level > 0:
		signal += noise_level * torch.randn_like(t)
	return signal


# Fixture: 동일한 오디오 쌍
@pytest.fixture
def identical_audio(tmp_path):
	sr = 16000
	audio = generate_sine_wave(freq=440, sample_rate=sr)
	gt_path = tmp_path / "gt.wav"
	gen_path = tmp_path / "gen.wav"
	torchaudio.save(str(gt_path), audio.unsqueeze(0), sr)
	torchaudio.save(str(gen_path), audio.unsqueeze(0), sr)
	return gt_path, gen_path


# Fixture: 다른 오디오 쌍
@pytest.fixture
def different_audio(tmp_path):
	sr = 16000
	gt_audio = generate_sine_wave(freq=440, sample_rate=sr)
	gen_audio = generate_sine_wave(freq=442, sample_rate=sr, noise_level=0.01)
	gt_path = tmp_path / "gt.wav"
	gen_path = tmp_path / "gen.wav"
	torchaudio.save(str(gt_path), gt_audio.unsqueeze(0), sr)
	torchaudio.save(str(gen_path), gen_audio.unsqueeze(0), sr)
	return gt_path, gen_path


# 1. 동일 오디오 테스트: 모든 지표 0 근접
def test_identical_audio(identical_audio):
	gt_path, gen_path = identical_audio
	evaluator = MelodyLoss(gt_path, gen_path)
	results = evaluator.evaluate(device="cpu")

	assert (
		results["Spectrogram_MAE"] < 1e-5
	), f"MAE should be near 0, got {results['MAE']}"
	assert (
		results["Pitch_Error"] < 20
	), f"Pitch error should be under 20, got {results['Pitch_Error']}"
	assert (
		results["Periodicity_Error"] < 20
	), f"Periodicity error should be under 20, got {results['Periodicity_Error']}"
	assert (
		abs(results["VUV_F1"] - 1.0) < 1e-5
	), f"VUV F1 should be 1.0, got {results['VUV_F1']}"


# 2. 다른 오디오 테스트: 지표 변화 확인
def test_different_audio(different_audio):
	gt_path, gen_path = different_audio
	evaluator = MelodyLoss(gt_path, gen_path)
	results = evaluator.evaluate(device="cpu")

	assert results["Spectrogram_MAE"] > 0.1, f"MAE should be >0.1, got {results['MAE']}"
	assert (
		results["Pitch_Error"] > 0
	), f"Pitch error should be >0, got {results['Pitch_Error']}"
	assert (
		results["Periodicity_Error"] > 0
	), f"Periodicity error should be >0, got {results['Periodicity_Error']}"
	assert (
		0.8 <= results["VUV_F1"] <= 1.0
	), f"VUV F1 should be high, got {results['VUV_F1']}"


# 3. 모양 불일치 시 오류 발생 테스트
def test_shape_mismatch_error(tmp_path):
	sr = 16000
	# 길이가 다른 오디오 생성
	gt_audio = generate_sine_wave(duration=1.0, sample_rate=sr)
	gen_audio = generate_sine_wave(duration=0.5, sample_rate=sr)

	gt_path = tmp_path / "gt.wav"
	gen_path = tmp_path / "gen.wav"
	torchaudio.save(str(gt_path), gt_audio.unsqueeze(0), sr)
	torchaudio.save(str(gen_path), gen_audio.unsqueeze(0), sr)

	evaluator = MelodyLoss(gt_path, gen_path)

	# Spectrogram MAE 계산 시 모양 불일치 오류 발생 확인
	with pytest.raises(AssertionError, match="Spectrogram shapes do not match"):
		evaluator.compute_spectrogram_mae()

	# Pitch/Periodicity 계산 시 오류 발생 확인
	with pytest.raises(AssertionError, match="Pitch shape do not match"):
		evaluator.compute_pitch_periodicity(sample_rate=sr)
