import tempfile
from pathlib import Path

import pytest
import torch
from unittest.mock import patch

from songsuro.autoencoder.models import Autoencoder
from songsuro.condition.model import ConditionalEncoder
from songsuro.data.dataset.aihub import AIHubDataset
from songsuro.data.loader.base import BaseDataLoader
from songsuro.models import Songsuro


tests_dir = Path(__file__).parent.parent
resources_dir = tests_dir / "resources"


def mock_conditional_encoder_forward(self, lyrics, original_audio):
	batch_size = original_audio.shape[0]
	return (
		torch.randn((batch_size, self.hidden_size, 500)),  # conditional embedding
		torch.randn((batch_size, self.prior_output_dim)),
	)  # prior


@pytest.fixture
def mock_autoencoder_checkpoint_path():
	with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
		# Save mock autoencoder
		mock_autoencoder = Autoencoder()
		torch.save(mock_autoencoder.state_dict(), tmp.name)
		yield tmp.name


@pytest.fixture
def songsuro_dataloader():
	test_dataset = AIHubDataset(str(resources_dir / "ai_hub_data_sample"))
	loader = BaseDataLoader(test_dataset, batch_size=2, num_workers=1)
	return loader


def test_songsuro_initialization(tmp_path, mock_autoencoder_checkpoint_path):
	"""Test that the Songsuro model can be instantiated correctly."""
	# Initialize the model
	model = Songsuro(
		latent_dim=80,
		condition_dim=128,
		autoencoder_checkpoint_path=str(mock_autoencoder_checkpoint_path),
	)

	# Check that components are initialized
	assert model.autoencoder is not None
	assert model.denoiser is not None
	assert model.conditional_encoder is not None
	assert model.noise_schedule is not None
	assert model.noise_level is not None


@patch.object(
	ConditionalEncoder,
	"forward",
	mock_conditional_encoder_forward,
)
def test_songsuro_training_step(songsuro_dataloader, mock_autoencoder_checkpoint_path):
	"""Test the training step of the Songsuro model."""
	# Initialize model
	model = Songsuro(
		latent_dim=80,
		condition_dim=128,
		autoencoder_checkpoint_path=mock_autoencoder_checkpoint_path,
	)

	# Get a batch from the dataloader
	batch = next(iter(songsuro_dataloader))

	# Run training step
	loss = model.training_step(batch, 0)

	# Check that loss is computed and is a tensor
	assert isinstance(loss, torch.Tensor)
	assert loss.ndim == 0  # Scalar tensor
	assert not torch.isnan(loss)
	assert not torch.isinf(loss)


@patch.object(
	ConditionalEncoder,
	"forward",
	mock_conditional_encoder_forward,
)
def test_songsuro_validation_step(
	songsuro_dataloader, mock_autoencoder_checkpoint_path
):
	"""Test the validation step of the Songsuro model."""
	# Initialize model
	model = Songsuro(
		latent_dim=80,
		condition_dim=128,
		autoencoder_checkpoint_path=mock_autoencoder_checkpoint_path,
	)

	# Get a batch
	batch = next(iter(songsuro_dataloader))

	# Run validation step
	result = model.validation_step(batch, 0)

	# Check the output structure
	assert isinstance(result, dict)
	assert "loss" in result
	assert "pred" in result
	assert isinstance(result["loss"], torch.Tensor)
	assert isinstance(result["pred"], torch.Tensor)


@pytest.mark.skip
@patch.object(
	ConditionalEncoder,
	"forward",
	mock_conditional_encoder_forward,
)
def test_songsuro_test_step(songsuro_dataloader, mock_autoencoder_checkpoint_path):
	"""Test the test step of the Songsuro model."""
	# Initialize model
	model = Songsuro(
		latent_dim=80,
		condition_dim=128,
		autoencoder_checkpoint_path=mock_autoencoder_checkpoint_path,
	)

	# Get a batch
	batch = next(iter(songsuro_dataloader))

	# Run test step
	result = model.test_step(batch, 0)

	# Verify test_step returns the same as validation_step
	assert isinstance(result, dict)
	assert "loss" in result
	assert "pred" in result
	assert isinstance(result["loss"], torch.Tensor)
	assert isinstance(result["pred"], torch.Tensor)


@patch.object(
	ConditionalEncoder,
	"forward",
	mock_conditional_encoder_forward,
)
def test_songsuro_forward(songsuro_dataloader, mock_autoencoder_checkpoint_path):
	"""Test the forward method of the Songsuro model."""
	# Initialize model
	model = Songsuro(
		latent_dim=80,
		condition_dim=128,
		autoencoder_checkpoint_path=mock_autoencoder_checkpoint_path,
	)
	# Get a batch
	batch = next(iter(songsuro_dataloader))

	# Run forward pass
	result = model(batch)

	# Check output
	assert isinstance(result, dict)
	assert "pred" in result
	assert isinstance(result["pred"], torch.Tensor)
	assert result["pred"].ndim == 2
