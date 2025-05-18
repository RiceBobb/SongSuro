import os
import pathlib

import pytest
import torch
import torchaudio

from songsuro.condition.encoder.melodyU import preprocess_f0


root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def sample_audio_file():
	yield os.path.join(resource_dir, "sample_only_voice.wav")


def test_mode_window_filter(sample_audio_file):
	waveform, fs = torchaudio.load(sample_audio_file)
	frame_quantized_f0 = preprocess_f0(waveform, fs)

	assert isinstance(frame_quantized_f0, torch.Tensor)
	assert frame_quantized_f0.ndim == 1
	assert frame_quantized_f0.shape[0] < 2000
