from torch import nn
from typing import Optional
import torch

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator
from songsuro.preprocess import preprocess_f0


class ConditionalEncoder(nn.Module):
	def __init__(
		self,
		lyrics_input_channel: int,
		melody_input_channel: int,
		hidden_size: Optional[int] = 192,
		prior_output_dim: Optional[int] = 1,
		device=None,
	):
		super().__init__()
		self.lyrics_encoder = FFTEncoder(lyrics_input_channel)
		self.melody_encoder = FFTEncoder(melody_input_channel)

		# self.timbre_encoder = TimbreEncoder(hidden_size=hidden_size, vq_input_dim=80)
		# self.style_encoder = StyleEncoder(hidden_size=hidden_size)

		self.enhanced_condition_encoder = FFTEncoder()
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

		self.device = (
			device
			if device is not None
			else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		)

	def forward(self, lyrics, audio_filepath=None, quantized_f0=None):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Tokenized lyrics sequence. Should be a tensor or can be converted to one.
		:param audio_filepath: Audio file path for melodyU encoding. Audio will be preprocessed quantized f0.
		:param quantized_f0: Pre-processed quantized f0 data. If provided, audio_filepath is ignored.
		:return: conditional embedding vector and prior
		"""
		if not isinstance(lyrics, torch.Tensor):
			lyrics = torch.tensor(lyrics).to(self.device)
		lyrics_lengths = torch.tensor([lyrics.shape[1]]).to(self.device)

		if quantized_f0 is None and audio_filepath is not None:
			quantized_f0 = preprocess_f0(audio_filepath)

		if not isinstance(quantized_f0, torch.Tensor):
			quantized_f0 = torch.tensor(quantized_f0).unsqueeze(0)

		quantized_f0 = quantized_f0.to(self.device)
		quantized_f0_lengths = torch.tensor([quantized_f0.shape[1]]).to(self.device)

		# lyrics and melodyU are encoded with FFTEncoder and then mean pooled
		# TODO: lyrics expand진행 -> 두 Representation 합치기전 음표길이에 따라 프레임 레벨로 확장되어야한다.
		# diffsinger의 lenght regulator를 통해서 언어적 hiddn sequence를 melodyU의 melspectogram에 맞춰서 길이로 확장
		# https://arxiv.org/pdf/2105.02446 -> model structure의 encoder 부분 참고하기
		# 왜 nn.Embedding에 토큰과 같은 자연어 같은걸 통과시키는게 아니라 숫자형태인 토큰을 통과시키는 거지?
		# TODO: nn.Embedding에 대한 이해도 필요했었음!! -> 벨로그 체크

		lyrics_embedding = torch.nn.functional.adaptive_avg_pool1d(
			self.lyrics_encoder(lyrics, lyrics_lengths), 1
		)

		melodyU_embedding = torch.nn.functional.adaptive_avg_pool1d(
			self.melody_encoder(quantized_f0, quantized_f0_lengths), 1
		)

		# timbre_embedding = self.timbre_encoder(lyrics)
		# style_embedding = self.style_encoder(lyrics)

		# TODO: min pooling은 lyrics expand후 melodyU와 합쳐진 후에 진행해야함
		summation_embedding = (
			lyrics_embedding + melodyU_embedding
			# + timbre_embedding + style_embedding
		)

		summation_lengths = torch.tensor([summation_embedding.shape[2]]).to(self.device)

		# TODO: 아래 참고 하기 -> expand 부분 참고 어쩌면 mean pooling이 아닐수도
		enhanced_condition_embedding = self.enhanced_condition_encoder(
			summation_embedding, summation_lengths
		)
		prior = self.prior_estimator(enhanced_condition_embedding)

		return enhanced_condition_embedding, prior

	def to(self, device):
		"""
		Move the model to the specified device.
		"""
		self.device = device
		return super().to(device)
