# -----------------------------------------------------------
# This code is adapted from CARGAN referenced by HiddenSinger: https://github.com/descriptinc/cargan/blob/master/cargan/evaluate/objective/metrics.py
# Original repository: https://github.com/descriptinc/cargan
# -----------------------------------------------------------

import torch
import torchcrepe


###############################################################################
# Pitch metrics
###############################################################################


def compute_spectrogram_mae(
	wav_mel_gt: torch.tensor = None, wav_mel_gen: torch.tensor = None
):
	assert wav_mel_gt.shape == wav_mel_gen.shape, "Spectrogram shapes do not match."
	mae = torch.mean(torch.abs(wav_mel_gt - wav_mel_gen)).item()
	return mae


def compute_pitch_periodicity(
	wav_mel_gt: torch.tensor = None,
	wav_mel_gen: torch.tensor = None,
	sample_rate: int = 24_000,
	hop_length: int = 1024,
	fmin: int = 50,
	fmax: int = 1100,
	threshold=0.5,
	device="cpu",
):
	wav_gt = wav_mel_gt.unsqueeze(0).to(device)
	wav_gen = wav_mel_gen.unsqueeze(0).to(device)

	pitch_gt, periodicity_gt = torchcrepe.predict(
		wav_gt,
		sample_rate,
		hop_length,
		fmin,
		fmax,
		model="full",
		return_periodicity=True,
		device=device,
	)
	pitch_gen, periodicity_gen = torchcrepe.predict(
		wav_gen,
		sample_rate,
		hop_length,
		fmin,
		fmax,
		model="full",
		return_periodicity=True,
		device=device,
	)

	assert pitch_gt.shape == pitch_gen.shape, "Pitch shape do not match."
	assert (
		periodicity_gt.shape == periodicity_gen.shape
	), "Periodicity shape do not match."

	assert (
		pitch_gt.shape == periodicity_gt.shape
	), "Pitch and periodicity shapes do not match."

	return (pitch_gt, periodicity_gt, pitch_gen, periodicity_gen)


class PitchEvaluator:
	def __init__(self):
		self.threshold = torchcrepe.threshold.Hysteresis()
		self.reset()

	def __call__(self):
		pitch_rmse = torch.sqrt(self.pitch_total / self.voiced)
		periodicity_rmse = torch.sqrt(self.periodicity_total / self.count)
		precision = self.true_positives / (self.true_positives + self.false_positives)
		recall = self.true_positives / (self.true_positives + self.false_negatives)
		f1 = 2 * precision * recall / (precision + recall)
		return {
			"pitch": pitch_rmse.item(),
			"periodicity": periodicity_rmse.item(),
			"f1": f1.item(),
			"precision": precision.item(),
			"recall": recall.item(),
		}

	def reset(self):
		self.count = 0
		self.voiced = 0
		self.pitch_total = 0.0
		self.periodicity_total = 0.0
		self.true_positives = 0
		self.false_positives = 0
		self.false_negatives = 0

	def update(self, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
		# Threshold
		true_threshold = self.threshold(true_pitch, true_periodicity)
		pred_threshold = self.threshold(pred_pitch, pred_periodicity)
		true_voiced = ~torch.isnan(true_threshold)
		pred_voiced = ~torch.isnan(pred_threshold)

		# Update periodicity rmse
		self.count += true_pitch.shape[1]
		self.periodicity_total += (true_periodicity - pred_periodicity).pow(2).sum()

		# Update pitch rmse
		voiced = true_voiced & pred_voiced
		self.voiced += voiced.sum()
		difference_cents = 1200 * (
			torch.log2(true_pitch[voiced]) - torch.log2(pred_pitch[voiced])
		)
		self.pitch_total += difference_cents.pow(2).sum()

		# Update voiced/unvoiced precision and recall
		self.true_positives += (true_voiced & pred_voiced).sum()
		self.false_positives += (~true_voiced & pred_voiced).sum()
		self.false_negatives += (true_voiced & ~pred_voiced).sum()


###############################################################################
# Waveform metrics
###############################################################################


class RMSE:
	def __init__(self):
		self.reset()

	def __call__(self):
		return torch.sqrt(self.total / self.count).item()

	def reset(self):
		self.count = 0
		self.total = 0.0

	def update(self, x, y):
		self.count += x.numel()
		self.total += ((x - y) ** 2).sum()


class L1:
	def __init__(self):
		self.reset()

	def __call__(self):
		return (self.total / self.count).item()

	def reset(self):
		self.count = 0
		self.total = 0.0

	def update(self, x, y):
		self.count += x.numel()
		self.total += torch.abs(x - y).sum()
