import torch
from songsuro.evaluation.metrics import (
	compute_spectrogram_mae,
	compute_pitch_periodicity,
	PitchEvaluator,
	RMSE,
	L1,
)


# Dummy data for testing
def make_dummy_tensors():
	wav_mel_gt = torch.ones(80, 100)
	wav_mel_gen = torch.ones(80, 100) * 1.1
	return wav_mel_gt, wav_mel_gen


def make_pitch_periodicity_tensors():
	# Simulated pitch and periodicity (no NaNs, all voiced)
	true_pitch = torch.tensor([[100.0, 200.0, 300.0, 400.0, 500.0]])
	true_periodicity = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
	pred_pitch = torch.tensor([[110.0, 210.0, 310.0, 390.0, 480.0]])
	pred_periodicity = torch.tensor([[0.85, 0.75, 0.65, 0.55, 0.45]])
	return true_pitch, true_periodicity, pred_pitch, pred_periodicity


def test_compute_spectrogram_mae():
	wav_mel_gt, wav_mel_gen = make_dummy_tensors()
	mae = compute_spectrogram_mae(wav_mel_gt, wav_mel_gen)
	assert isinstance(mae, float)
	assert mae > 0


def test_compute_pitch_periodicity_shapes(monkeypatch):
	# Patch torchcrepe.predict to return dummy values
	def dummy_predict(wav, sr, hop, fmin, fmax, model, return_periodicity, device):
		shape = (1, 10)
		return torch.ones(shape), torch.ones(shape) * 0.5

	monkeypatch.setattr("torchcrepe.predict", dummy_predict)

	wav_mel_gt, wav_mel_gen = make_dummy_tensors()
	pitch_gt, periodicity_gt, pitch_gen, periodicity_gen = compute_pitch_periodicity(
		wav_mel_gt[0], wav_mel_gen[0]
	)
	assert (
		pitch_gt.shape
		== periodicity_gt.shape
		== pitch_gen.shape
		== periodicity_gen.shape
	)


def test_pitch_evaluator_update_and_call():
	evaluator = PitchEvaluator()
	true_pitch, true_periodicity, pred_pitch, pred_periodicity = (
		make_pitch_periodicity_tensors()
	)
	evaluator.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)
	results = evaluator()
	assert all(
		key in results for key in ["pitch", "periodicity", "f1", "precision", "recall"]
	)
	assert results["pitch"] >= 0
	assert results["periodicity"] >= 0


def test_rmse():
	rmse = RMSE()
	x = torch.tensor([1.0, 2.0, 3.0, 4.0])
	y = torch.tensor([1.1, 1.9, 3.1, 3.9])
	rmse.update(x, y)
	val = rmse()
	assert isinstance(val, float)
	assert val >= 0


def test_l1():
	l1 = L1()
	x = torch.tensor([1.0, 2.0, 3.0, 4.0])
	y = torch.tensor([1.1, 1.9, 3.1, 3.9])
	l1.update(x, y)
	val = l1()
	assert isinstance(val, float)
	assert val >= 0
