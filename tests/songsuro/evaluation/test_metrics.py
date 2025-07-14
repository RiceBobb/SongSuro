import torch
import pytest
from songsuro.evaluation.metrics import PitchEvaluator, RMSE


@pytest.fixture
def sample_pitch_periodicity():
	# Create dummy pitch and periodicity tensors (shape: [batch, length])
	true_pitch = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
	true_periodicity = torch.tensor([[0.8, 0.9, 0.7, 0.6]])
	pred_pitch = torch.tensor([[110.0, 190.0, 310.0, 390.0]])
	pred_periodicity = torch.tensor([[0.75, 0.85, 0.65, 0.55]])
	return true_pitch, true_periodicity, pred_pitch, pred_periodicity


def test_pitch_evaluator_update_and_call(sample_pitch_periodicity):
	true_pitch, true_periodicity, pred_pitch, pred_periodicity = (
		sample_pitch_periodicity
	)
	evaluator = PitchEvaluator()
	evaluator.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)
	results = evaluator()
	# Check that all expected metrics are present and are floats
	assert set(results.keys()) == {"pitch", "periodicity", "f1", "precision", "recall"}
	for key in results:
		assert isinstance(results[key], float)


def test_pitch_evaluator_reset(sample_pitch_periodicity):
	true_pitch, true_periodicity, pred_pitch, pred_periodicity = (
		sample_pitch_periodicity
	)
	evaluator = PitchEvaluator()
	evaluator.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)
	evaluator.reset()
	# After reset, all counters and totals should be zero
	assert evaluator.count == 0
	assert evaluator.voiced == 0
	assert evaluator.pitch_total == 0.0
	assert evaluator.periodicity_total == 0.0
	assert evaluator.true_positives == 0
	assert evaluator.false_positives == 0
	assert evaluator.false_negatives == 0


def test_rmse_update_and_call():
	rmse = RMSE()
	x = torch.tensor([1.0, 2.0, 3.0])
	y = torch.tensor([1.0, 2.5, 2.5])
	rmse.update(x, y)
	result = rmse()
	# RMSE should be a float and non-negative
	assert isinstance(result, float)
	assert result >= 0


def test_rmse_reset():
	rmse = RMSE()
	x = torch.tensor([1.0, 2.0, 3.0])
	y = torch.tensor([1.0, 2.5, 2.5])
	rmse.update(x, y)
	rmse.reset()
	assert rmse.count == 0
	assert rmse.total == 0.0
