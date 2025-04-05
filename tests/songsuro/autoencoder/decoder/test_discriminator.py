import torch
import torch.nn as nn

from songsuro.autoencoder.decoder.discriminator import (
	DiscriminatorP,
	MultiPeriodDiscriminator,
	DiscriminatorS,
	MultiScaleDiscriminator,
)


class TestDiscriminatorP:
	def test_initialization(self):
		# 기본 초기화 테스트
		disc = DiscriminatorP(period=2)

		assert disc.period == 2
		assert len(disc.convs) == 5
		assert isinstance(disc.conv_post, nn.Conv2d)

		# 모든 레이어가 weight_norm으로 초기화되었는지 확인
		for layer in disc.convs:
			assert hasattr(layer, "weight_v")
		assert hasattr(disc.conv_post, "weight_v")

	def test_spectral_norm(self):
		# spectral_norm 사용 테스트
		disc = DiscriminatorP(period=2, use_spectral_norm=True)

		# 모든 레이어가 spectral_norm으로 초기화되었는지 확인
		for layer in disc.convs:
			assert hasattr(layer, "weight_orig")
		assert hasattr(disc.conv_post, "weight_orig")

	def test_forward(self):
		# 순전파 테스트
		disc = DiscriminatorP(period=2)

		# 입력 텐서 생성 (배치 크기 3, 채널 1, 시간 100)
		x = torch.randn(3, 1, 100)

		# 순전파 실행
		output, fmap = disc(x)

		# 출력 형태 확인
		assert isinstance(output, torch.Tensor)
		assert len(fmap) == 6  # 5개 conv + 1개 conv_post

		# 출력 크기 확인
		assert output.shape[0] == 3  # 배치 크기

		# feature map 확인
		for i, feat in enumerate(fmap):
			assert isinstance(feat, torch.Tensor)
			assert feat.shape[0] == 3  # 배치 크기

	def test_padding(self):
		# 패딩 테스트 (기간으로 나누어 떨어지지 않는 경우)
		disc = DiscriminatorP(period=3)

		# 입력 텐서 생성 (배치 크기 2, 채널 1, 시간 101)
		x = torch.randn(2, 1, 101)

		# 순전파 실행
		output, fmap = disc(x)

		# 출력 형태 확인
		assert isinstance(output, torch.Tensor)
		assert output.shape[0] == 2  # 배치 크기


class TestMultiPeriodDiscriminator:
	def test_initialization(self):
		# 초기화 테스트
		mpd = MultiPeriodDiscriminator()

		assert len(mpd.discriminators) == 5
		for i, d in enumerate(mpd.discriminators):
			assert isinstance(d, DiscriminatorP)
			assert d.period in [2, 3, 5, 7, 11]

	def test_forward(self):
		# 순전파 테스트
		mpd = MultiPeriodDiscriminator()

		# 입력 텐서 생성 (배치 크기 2, 채널 1, 시간 200)
		y = torch.randn(2, 1, 200)
		y_hat = torch.randn(2, 1, 200)

		# 순전파 실행
		y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)

		# 출력 형태 확인
		assert len(y_d_rs) == 5  # 5개 판별자
		assert len(y_d_gs) == 5
		assert len(fmap_rs) == 5
		assert len(fmap_gs) == 5

		# 각 판별자의 출력 확인
		for i in range(5):
			assert isinstance(y_d_rs[i], torch.Tensor)
			assert isinstance(y_d_gs[i], torch.Tensor)
			assert y_d_rs[i].shape[0] == 2  # 배치 크기
			assert y_d_gs[i].shape[0] == 2  # 배치 크기

			# feature map 확인
			assert len(fmap_rs[i]) == 6  # 5개 conv + 1개 conv_post
			assert len(fmap_gs[i]) == 6


class TestDiscriminatorS:
	def test_initialization(self):
		# 기본 초기화 테스트
		disc = DiscriminatorS()

		assert len(disc.convs) == 7
		assert isinstance(disc.conv_post, nn.Conv1d)

		# 모든 레이어가 weight_norm으로 초기화되었는지 확인
		for layer in disc.convs:
			assert hasattr(layer, "weight_v")
		assert hasattr(disc.conv_post, "weight_v")

	def test_spectral_norm(self):
		# spectral_norm 사용 테스트
		disc = DiscriminatorS(use_spectral_norm=True)

		# 모든 레이어가 spectral_norm으로 초기화되었는지 확인
		for layer in disc.convs:
			assert hasattr(layer, "weight_orig")
		assert hasattr(disc.conv_post, "weight_orig")

	def test_forward(self):
		# 순전파 테스트
		disc = DiscriminatorS()

		# 입력 텐서 생성 (배치 크기 3, 채널 1, 시간 100)
		x = torch.randn(3, 1, 100)

		# 순전파 실행
		output, fmap = disc(x)

		# 출력 형태 확인
		assert isinstance(output, torch.Tensor)
		assert len(fmap) == 8  # 7개 conv + 1개 conv_post

		# 출력 크기 확인
		assert output.shape[0] == 3  # 배치 크기

		# feature map 확인
		for i, feat in enumerate(fmap):
			assert isinstance(feat, torch.Tensor)
			assert feat.shape[0] == 3  # 배치 크기


class TestMultiScaleDiscriminator:
	def test_initialization(self):
		# 초기화 테스트
		msd = MultiScaleDiscriminator()

		assert len(msd.discriminators) == 3
		assert len(msd.meanpools) == 2

		# 첫 번째 판별자는 spectral_norm 사용
		assert hasattr(msd.discriminators[0].convs[0], "weight_orig")

		# 나머지 판별자는 weight_norm 사용
		assert hasattr(msd.discriminators[1].convs[0], "weight_v")
		assert hasattr(msd.discriminators[2].convs[0], "weight_v")

	def test_forward(self):
		# 순전파 테스트
		msd = MultiScaleDiscriminator()

		# 입력 텐서 생성 (배치 크기 2, 채널 1, 시간 200)
		y = torch.randn(2, 1, 200)
		y_hat = torch.randn(2, 1, 200)

		# 순전파 실행
		y_d_rs, y_d_gs, fmap_rs, fmap_gs = msd(y, y_hat)

		# 출력 형태 확인
		assert len(y_d_rs) == 3  # 3개 판별자
		assert len(y_d_gs) == 3
		assert len(fmap_rs) == 3
		assert len(fmap_gs) == 3

		# 각 판별자의 출력 확인
		for i in range(3):
			assert isinstance(y_d_rs[i], torch.Tensor)
			assert isinstance(y_d_gs[i], torch.Tensor)
			assert y_d_rs[i].shape[0] == 2  # 배치 크기
			assert y_d_gs[i].shape[0] == 2  # 배치 크기

			# feature map 확인
			assert len(fmap_rs[i]) == 8  # 7개 conv + 1개 conv_post
			assert len(fmap_gs[i]) == 8
