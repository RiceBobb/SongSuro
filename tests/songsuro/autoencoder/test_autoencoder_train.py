from pathlib import Path


from songsuro.autoencoder.train import train


# @pytest.mark.skip
def test_autoencoder_train():
	root_dir = Path(__file__).parent.parent.parent.parent
	data_dir = root_dir / "tests" / "resources" / "ai_hub_data_sample"
	checkpoint_dir = root_dir / "autoencoder_checkpoint"
	train(str(data_dir), str(data_dir), 1, 6, checkpoint_dir)
