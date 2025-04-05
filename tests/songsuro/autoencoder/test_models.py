import pytest
import torch
from unittest.mock import MagicMock

from songsuro.autoencoder.models import Autoencoder
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer


class TestAutoencoder:
	@pytest.fixture
	def generator_config(self):
		# Create a mock object for Generator configuration
		h = MagicMock()
		h.resblock_kernel_sizes = [3, 7, 11]
		h.upsample_rates = [8, 8, 2, 2]
		h.upsample_kernel_sizes = [16, 16, 4, 4]
		h.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
		h.upsample_initial_channel = 512
		h.resblock = "1"
		h.input_channels = 128  # 중요: 디코더 입력 채널을 128로 설정
		return h

	@pytest.fixture
	def decoder(self, generator_config):
		return Generator(generator_config)

	@pytest.fixture
	def encoder(self):
		return Encoder(n_in=1, n_out=128, parent_vc=None)

	@pytest.fixture
	def quantizer(self):
		return ResidualVectorQuantizer(
			input_dim=128,
			num_quantizers=8,  # Use smaller value for testing
			codebook_size=32,  # Use smaller value for testing
			codebook_dim=128,
		)

	@pytest.fixture
	def autoencoder(self, encoder, quantizer, decoder):
		return Autoencoder(encoder, quantizer, decoder)

	def test_initialization(self, encoder, quantizer, decoder, autoencoder):
		# 컴포넌트가 올바르게 할당되었는지 확인
		assert autoencoder.encoder is encoder
		assert autoencoder.quantizer is quantizer
		assert autoencoder.decoder is decoder

	def test_forward_with_mocks(self):
		# 모킹된 컴포넌트 생성
		mock_encoder = MagicMock()
		mock_quantizer = MagicMock()
		mock_decoder = MagicMock()

		# 모의 출력 설정
		encoded_output = torch.randn(2, 128, 50)
		mock_encoder.return_value = encoded_output

		quantized_output = torch.randn(2, 128, 50)
		commit_loss = torch.tensor(0.1)
		mock_quantizer.return_value = (quantized_output, commit_loss)

		decoded_output = torch.randn(2, 1, 100)
		mock_decoder.return_value = decoded_output

		# 모킹된 컴포넌트로 autoencoder 생성
		autoencoder = Autoencoder(mock_encoder, mock_quantizer, mock_decoder)

		# 입력 텐서 생성
		x = torch.randn(2, 1, 100)

		# 순전파 실행
		output, loss = autoencoder(x)

		# 각 컴포넌트가 올바른 입력으로 호출되었는지 확인
		mock_encoder.assert_called_once_with(x)
		mock_quantizer.assert_called_once_with(encoded_output)
		mock_decoder.assert_called_once_with(quantized_output)

		# 출력이 예상대로인지 확인
		assert torch.equal(output, decoded_output)
		assert torch.equal(loss, commit_loss)

	def test_with_real_components(self, autoencoder):
		# 입력 텐서 생성 (작은 크기로 테스트)
		x = torch.randn(2, 1, 32)

		# 순전파 실행
		output, loss = autoencoder(x)

		# 출력 검증
		assert isinstance(output, torch.Tensor)
		assert isinstance(loss, torch.Tensor)
		assert output.shape[0] == 2  # 배치 크기
		assert output.shape[1] == 1  # 출력 채널

		# 업샘플링 비율에 따른 출력 길이 확인
		expected_length = 32
		for rate in [8, 8, 2, 2]:  # h.upsample_rates
			expected_length *= rate
		assert output.shape[2] == expected_length
