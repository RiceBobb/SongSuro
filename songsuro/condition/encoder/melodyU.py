import torch

from songsuro.preprocess import (
	trim_silence,
	extract_f0_from_tensor,
	hz_to_mel_torch,
	quantize_mel_scale_torch,
)


def preprocess_f0(waveform: torch.Tensor, sample_rate: int):
	LEVELS = 127
	MEL_MIN_VAL = 133
	MEL_MAX_VAL = 571

	trimmed_waveform = trim_silence(waveform, sample_rate)
	pitch = extract_f0_from_tensor(trimmed_waveform, sample_rate)

	mel_f0 = hz_to_mel_torch(pitch)
	if mel_f0.ndim > 1:
		mel_f0 = torch.mean(mel_f0, dim=0)

	quantized_f0 = quantize_mel_scale_torch(
		mel_f0, levels=LEVELS, min_val=MEL_MIN_VAL, max_val=MEL_MAX_VAL
	)
	return quantized_f0
