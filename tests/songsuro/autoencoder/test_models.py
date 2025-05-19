from pathlib import Path

import pytest
import torch

from songsuro.autoencoder.decoder.discriminator import (
	MultiPeriodDiscriminator,
	MultiScaleDiscriminator,
)
from songsuro.autoencoder.models import Autoencoder
from songsuro.autoencoder.encoder.encoder import Encoder
from songsuro.autoencoder.decoder.generator import Generator
from songsuro.autoencoder.quantizer import ResidualVectorQuantizer
from songsuro.data.dataset.aihub import AIHubDataset
from songsuro.data.loader.base import BaseDataLoader
from tests.util import is_github_action

tests_dir = Path(__file__).parent.parent.parent
resources_dir = tests_dir / "resources"


@pytest.fixture
def ae_params():
	# Dictionary of hyperparameters to pass to the Autoencoder
	return dict(
		encoder_in_channels=128,
		encoder_out_channels=80,
		num_quantizers=8,
		codebook_size=1024,
		codebook_dim=128,
		resblock_kernel_sizes=[3, 7, 11],
		upsample_rates=[16, 8, 4, 4],
		upsample_kernel_sizes=[16, 16, 4, 4],
		resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
		upsample_initial_channel=512,
		resblock="1",
		lambda_recon=0.1,
		lambda_emb=0.1,
		lambda_fm=0.1,
	)


@pytest.fixture
def autoencoder(ae_params):
	return Autoencoder(**ae_params)


@pytest.fixture
def dataloader():
	test_dataset = AIHubDataset(str(resources_dir / "ai_hub_data_sample"))
	loader = BaseDataLoader(test_dataset, batch_size=2, num_workers=1)
	return loader


def test_initialization(ae_params, autoencoder):
	# Check if components are initialized correctly
	assert isinstance(autoencoder.encoder, Encoder)
	assert isinstance(autoencoder.quantizer, ResidualVectorQuantizer)
	assert isinstance(autoencoder.decoder, Generator)
	assert isinstance(autoencoder.mpd, MultiPeriodDiscriminator)
	assert isinstance(autoencoder.msd, MultiScaleDiscriminator)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_autoencoder_validation_step(autoencoder, dataloader):
	batch = next(iter(dataloader))
	result = autoencoder.validation_step(batch, 0)

	assert "loss" in result.keys()
	assert "pred_audio" in result.keys()
	assert "pred_mel" in result.keys()

	assert isinstance(result["loss"], torch.Tensor)
	assert isinstance(result["pred_audio"], torch.Tensor)
	assert result["pred_audio"].ndim == 2
	assert isinstance(result["pred_mel"], torch.Tensor)
	assert result["pred_mel"].ndim == 3


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_autoencoder_test_step(autoencoder, dataloader):
	batch = next(iter(dataloader))
	result = autoencoder.test_step(batch, 0)

	assert "loss" in result.keys()
	assert "pred_audio" in result.keys()
	assert "pred_mel" in result.keys()

	assert isinstance(result["loss"], torch.Tensor)
	assert isinstance(result["pred_audio"], torch.Tensor)
	assert result["pred_audio"].ndim == 2
	assert isinstance(result["pred_mel"], torch.Tensor)
	assert result["pred_mel"].ndim == 3
	assert result["pred_mel"].shape[0] == 2
