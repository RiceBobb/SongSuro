import os

from songsuro.preprocess import (
	extract_f0_from_file,
	hz_to_mel,
	quantize_mel_scale,
	mode_window_filter,
)


def preprocess_f0(audio_filepath: str):
	if not os.path.exists(audio_filepath):
		raise FileNotFoundError(audio_filepath)

	LEVELS = 127
	MEL_MIN_VAL = 133
	MEL_MAX_VAL = 571

	f0_signal, fs = extract_f0_from_file(audio_filepath)
	mel_f0 = hz_to_mel(f0_signal)
	quantized_f0 = quantize_mel_scale(
		mel_f0, levels=LEVELS, min_val=MEL_MIN_VAL, max_val=MEL_MAX_VAL
	)
	frame_duration_ms = 20
	indices_per_frame = int((frame_duration_ms / 1000) * fs)

	frame_quantized_f0 = mode_window_filter(quantized_f0, indices_per_frame)
	return frame_quantized_f0
