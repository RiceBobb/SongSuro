import torch
import pytest
import numpy as np
from songsuro.evaluation.metrics import (
	compute_spectrogram_mae,
	compute_pitch_periodicity,
	compute_vuv_f1,
)


@pytest.fixture
def sine_wave_samples():
	sample_rate = 24000
	t = torch.linspace(0, 1, sample_rate)
	sine_440 = torch.sin(2 * np.pi * 440 * t)  # 440Hz
	sine_880 = torch.sin(2 * np.pi * 880 * t)  # 880Hz
	return sine_440, sine_880


@pytest.fixture
def noisy_sine_samples():
	sample_rate = 24000
	t = torch.linspace(0, 1, sample_rate)
	sine = torch.sin(2 * np.pi * 440 * t)
	noise = 0.05 * torch.randn(sample_rate)
	noisy = sine + noise
	return sine, noisy


def test_compute_spectrogram_mae(sine_wave_samples):
	gt, gen = sine_wave_samples
	gt_mel = gt.unsqueeze(0)
	gen_mel = gen.unsqueeze(0)
	mae = compute_spectrogram_mae(gt_mel, gen_mel)
	expected = torch.mean(torch.abs(gt_mel - gen_mel)).item()
	assert np.isclose(mae, expected), f"Expected {expected}, got {mae}"


def test_compute_pitch_periodicity(noisy_sine_samples):
	gt, gen = noisy_sine_samples
	pitch_error, periodicity_error, voiced_mask, periodicity_gt, periodicity_gen = (
		compute_pitch_periodicity(gt, gen, device="cpu")
	)
	assert isinstance(pitch_error, float)
	assert isinstance(periodicity_error, float)
	assert isinstance(voiced_mask, torch.Tensor)
	assert isinstance(periodicity_gt, torch.Tensor)
	assert isinstance(periodicity_gen, torch.Tensor)
	assert periodicity_gt.shape == periodicity_gen.shape
	assert voiced_mask.dtype == torch.bool


def test_compute_vuv_f1(noisy_sine_samples):
	gt, gen = noisy_sine_samples
	_, _, _, periodicity_gt, periodicity_gen = compute_pitch_periodicity(
		gt, gen, device="cpu"
	)
	f1 = compute_vuv_f1(periodicity_gt[0], periodicity_gen[0])
	assert 0 <= f1 <= 1
