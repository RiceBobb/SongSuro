import pytest
import torch
import os
import pathlib
import torchaudio
from songsuro.preprocess import preprocess_f0

from songsuro.condition.model import ConditionalEncoder

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
resource_dir = os.path.join(root_dir, "resources")


class TestConditionalEncoder:
	@pytest.fixture(
		params=[
			{
				"batch_size": 2,
				"note_duration_seq_len": 100,
				"max_lyrics_seq_len": 20,
				"f0_seq_len": 200,
			},
			{
				"batch_size": 4,
				"note_duration_seq_len": 300,
				"max_lyrics_seq_len": 30,
				"f0_seq_len": 400,
			},
			{
				"batch_size": 8,
				"note_duration_seq_len": 512,
				"max_lyrics_seq_len": 50,
				"f0_seq_len": 800,
			},
		]
	)
	def generate_param_test_combo(self, request):
		return request.param

	@pytest.fixture
	def conditional_encoder(self):
		mock_latent_dim = 80
		return ConditionalEncoder(
			lyrics_input_channel=512,
			melody_input_channel=128,
			prior_output_dim=mock_latent_dim,
		)

	def sample_note_durations(self, batch_size, note_duration_seq_len):
		durations = torch.randint(
			1, 9, (batch_size, note_duration_seq_len), dtype=torch.long
		)
		duration_tensor_lengths = torch.randint(
			1, note_duration_seq_len + 1, (batch_size,), dtype=torch.long
		)
		# durations: (batch, max_seq_len), lengths: (batch,)
		return durations, duration_tensor_lengths

	def sample_lyrics_and_lengths(self, batch_size, max_lyrics_seq_len):
		min_seq_len = 10

		# 각 배치별 실제 길이 랜덤 생성
		lengths = torch.randint(min_seq_len, max_lyrics_seq_len + 1, (batch_size,))
		lyrics = torch.zeros((batch_size, max_lyrics_seq_len), dtype=torch.long)

		for i, length in enumerate(lengths):
			lyrics[i, :length] = torch.randint(
				1, 100, (length,), dtype=torch.long
			)  # 0은 패딩이므로 1부터

		# lyrics: (batch, max_lyrics_seq_len), lengths: (batch,)
		return lyrics, lengths

	def sample_audio_file(self, batch_size):
		"""Provide multiple audio file paths as a list."""
		wav_files = [
			os.path.join(resource_dir, f)
			for f in os.listdir(resource_dir)
			if f.endswith(".wav")
		]

		files = wav_files[:batch_size]
		yield files

	def pad_sequence(self, sequences, batch_first=True, padding_value=0):
		# You can use torch.nn.utils.rnn.pad_sequence
		lengths = [seq.shape[-1] for seq in sequences]
		max_len = max(lengths)
		padded = []
		for seq in sequences:
			pad_size = max_len - seq.shape[-1]
			if pad_size > 0:
				seq = torch.cat(
					[seq, torch.full((pad_size,), padding_value, dtype=seq.dtype)]
				)
			padded.append(seq)
		return torch.stack(padded), torch.tensor(lengths)

	def sample_quantized_f0_batch(self, batch_size):
		# sample_audio_files: 오디오 파일 경로 리스트
		f0_list = []
		lengths = []
		sample_audio_files = self.sample_audio_file(batch_size=batch_size)
		for audio_file in sample_audio_files:
			waveform, fs = torchaudio.load(audio_file)
			quantized_f0 = preprocess_f0(waveform, fs)
			f0_list.append(quantized_f0)
			lengths.append(quantized_f0.shape[-1])

		# 패딩해서 batch tensor로 만듦
		batch_f0, batch_lengths = self.pad_sequence(
			f0_list, batch_first=True, padding_value=0
		)
		# batch_f0: (batch, max_length), batch_lengths: (batch,)
		return batch_f0, batch_lengths

	# TODO: Lyrics expand to frame level
	def test_expand_embeddings_to_frame_level(
		self, conditional_encoder, generate_param_test_combo
	):
		batch_size = generate_param_test_combo["batch_size"]
		note_duration_seq_len = generate_param_test_combo["note_duration_seq_len"]

		sample_note_durations, sample_lengths = self.sample_note_durations(
			batch_size, note_duration_seq_len
		)
		batch_size, seq_len = sample_note_durations.shape
		mock_hidden = 192
		mock_embeddings = torch.randn(batch_size, mock_hidden, seq_len)

		expanded = conditional_encoder.expand_embeddings_to_frame_level(
			mock_embeddings, sample_note_durations
		)

		# 전체 shape 확인
		assert expanded.shape[0] == batch_size
		assert expanded.shape[1] == mock_hidden
		max_total_frames = max(
			sample_note_durations[b].sum().item() for b in range(batch_size)
		)
		assert expanded.shape[2] == max_total_frames

		# 각 배치별 실제 프레임 수 확인
		for b in range(batch_size):
			true_length = sample_note_durations[b].sum().item()
			# 패딩 전 구간은 값이 존재, 패딩 구간은 0
			assert torch.all(expanded[b, :, true_length:] == 0)
			assert expanded[b, :, :true_length].shape == (mock_hidden, true_length)
			assert isinstance(expanded, torch.Tensor)

	def test_sample_lyrics_and_lengths(self, generate_param_test_combo):
		batch_size = generate_param_test_combo["batch_size"]
		max_lyrics_seq_len = generate_param_test_combo["max_lyrics_seq_len"]

		lyrics, lengths = self.sample_lyrics_and_lengths(batch_size, max_lyrics_seq_len)
		assert lyrics.shape[0] == lengths.shape[0]
		assert lyrics.shape[1] >= lengths.max()
		print(f"\nBatch size: {lyrics.shape[0]}, Max seq len: {lyrics.shape[1]}")
		print(f"Lengths: {lengths}")

	def test_initialization(self, conditional_encoder):
		"""Test if the encoder initializes correctly with all components."""
		assert hasattr(conditional_encoder, "lyrics_encoder")
		assert hasattr(conditional_encoder, "enhanced_condition_encoder")
		assert hasattr(conditional_encoder, "prior_estimator")

	def test_forward_with_tensors(
		self,
		conditional_encoder,
		generate_param_test_combo,
	):
		"""Test forward pass with tensor inputs."""
		batch_size = generate_param_test_combo["batch_size"]
		note_duration_seq_len = generate_param_test_combo["note_duration_seq_len"]
		max_lyrics_seq_len = generate_param_test_combo["max_lyrics_seq_len"]

		sample_lyrics, sample_lyrics_length = self.sample_lyrics_and_lengths(
			batch_size, max_lyrics_seq_len
		)
		sample_f0, sample_f0_length = self.sample_quantized_f0_batch(batch_size)
		sample_note_durations, sample_note_durations_length = (
			self.sample_note_durations(batch_size, note_duration_seq_len)
		)

		batch_size = sample_lyrics.size(0)

		condition_embedding, prior = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
			note_durations=sample_note_durations,
			note_durations_lengths=sample_note_durations_length,
		)

		# Check output shapes and types
		assert isinstance(condition_embedding, torch.Tensor)
		assert isinstance(prior, torch.Tensor)
		assert condition_embedding.dim() == 3  # (batch, channels, seq_len)
		assert prior.dim() == 2  # Assuming prior has shape (batch, dim, seq_len)

		assert condition_embedding.size(0) == batch_size
		assert prior.size(0) == batch_size

	def test_input_output_consistency(
		self, conditional_encoder, generate_param_test_combo
	):
		"""Test that the same input produces the same output."""
		batch_size = generate_param_test_combo["batch_size"]
		max_lyrics_seq_len = generate_param_test_combo["max_lyrics_seq_len"]
		note_duration_seq_len = generate_param_test_combo["note_duration_seq_len"]

		sample_lyrics, sample_lyrics_length = self.sample_lyrics_and_lengths(
			batch_size, max_lyrics_seq_len
		)
		sample_f0, sample_f0_length = self.sample_quantized_f0_batch(batch_size)
		sample_note_duration, sample_note_duration_length = self.sample_note_durations(
			batch_size, note_duration_seq_len
		)

		embedding1, prior1 = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
			note_durations=sample_note_duration,
			note_durations_lengths=sample_note_duration_length,
		)
		embedding2, prior2 = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
			note_durations=sample_note_duration,
			note_durations_lengths=sample_note_duration_length,
		)

		torch.testing.assert_close(embedding1, embedding2)
		torch.testing.assert_close(prior1, prior2)
