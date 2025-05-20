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
	@pytest.fixture
	def conditional_encoder(self):
		mock_latent_dim = 80
		return ConditionalEncoder(
			lyrics_input_channel=512,
			melody_input_channel=128,
			prior_output_dim=mock_latent_dim,
		)

	@pytest.fixture(params=[(2, 20), (4, 30), (8, 50)])
	def sample_lyrics_and_lengths(self, request):
		batch_size, max_seq_len = request.param
		min_seq_len = 10

		# 각 배치별 실제 길이 랜덤 생성
		lengths = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,))
		lyrics = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

		for i, length in enumerate(lengths):
			lyrics[i, :length] = torch.randint(
				1, 100, (length,), dtype=torch.long
			)  # 0은 패딩이므로 1부터

		# lyrics: (batch, max_seq_len), lengths: (batch,)
		return lyrics, lengths

	@pytest.fixture
	def sample_audio_file(self):
		"""Provide multiple audio file paths as a list."""
		yield [
			os.path.join(resource_dir, "sample_only_voice.wav"),
			os.path.join(resource_dir, "sample_song.wav"),
		]

	def pad_sequence(self, sequences, batch_first=True, padding_value=0):
		# torch.nn.utils.rnn.pad_sequence를 써도 됩니다.
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

	@pytest.fixture
	def sample_quantized_f0_batch(self, sample_audio_file):
		# sample_audio_files: 오디오 파일 경로 리스트
		f0_list = []
		lengths = []
		for audio_file in sample_audio_file:
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

	def test_sample_lyrics_and_lengths(self, sample_lyrics_and_lengths):
		lyrics, lengths = sample_lyrics_and_lengths
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
		sample_lyrics_and_lengths,
		sample_quantized_f0_batch,
	):
		"""Test forward pass with tensor inputs."""
		sample_lyrics, sample_lyrics_length = sample_lyrics_and_lengths
		sample_f0, sample_f0_length = sample_quantized_f0_batch

		batch_size = sample_lyrics.size(0)

		condition_embedding, prior = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
		)

		# Check output shapes and types
		assert isinstance(condition_embedding, torch.Tensor)
		assert isinstance(prior, torch.Tensor)
		assert condition_embedding.dim() == 3  # (batch, channels, seq_len)
		assert prior.dim() == 2  # Assuming prior has shape (batch, dim, seq_len)

		assert condition_embedding.size(0) == batch_size
		assert prior.size(0) == batch_size

	def test_input_output_consistency(
		self, conditional_encoder, sample_lyrics_and_lengths, sample_quantized_f0_batch
	):
		"""Test that the same input produces the same output."""
		sample_lyrics, sample_lyrics_length = sample_lyrics_and_lengths
		sample_f0, sample_f0_length = sample_quantized_f0_batch

		embedding1, prior1 = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
		)
		embedding2, prior2 = conditional_encoder(
			sample_lyrics,
			sample_lyrics_length,
			quantized_f0=sample_f0,
			quantized_f0_lengths=sample_f0_length,
		)

		torch.testing.assert_close(embedding1, embedding2)
		torch.testing.assert_close(prior1, prior2)
