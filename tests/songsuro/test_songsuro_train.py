import tempfile
from pathlib import Path

import pytest
import torch
from songsuro.autoencoder.models import Autoencoder
from songsuro.train import train


@pytest.mark.skip
def test_train():
	root_dir = Path(__file__).parent.parent.parent
	data_dir = root_dir / "tests" / "resources" / "ai_hub_data_sample"
	checkpoint_dir = root_dir / "train_checkpoint"

	with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
		# Save mock autoencoder
		mock_autoencoder = Autoencoder()
		torch.save(mock_autoencoder.state_dict(), tmp.name)

		train(str(data_dir), str(data_dir), 4, 6, tmp.name, str(checkpoint_dir))
