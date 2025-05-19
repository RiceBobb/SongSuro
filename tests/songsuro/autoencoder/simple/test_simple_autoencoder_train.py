import os
from datetime import datetime
from pathlib import Path

import torch

from songsuro.autoencoder.train import train


# @pytest.mark.skip
def autoencoder_train():
	torch.cuda.memory._record_memory_history(max_entries=200_000)
	root_dir = Path(__file__).parent.parent.parent.parent.parent
	data_dir = root_dir / "tests" / "resources" / "ai_hub_data_sample"
	checkpoint_dir = root_dir / "simple_autoencoder_checkpoint"
	try:
		train(str(data_dir), str(data_dir), 4, 6, checkpoint_dir)
		torch.cuda.memory._dump_snapshot(
			os.path.join(
				root_dir,
				f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-autoencoder.pickle",
			),
		)
	except:
		torch.cuda.memory._dump_snapshot(
			os.path.join(
				root_dir,
				f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-autoencoder.pickle",
			),
		)
	torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
	autoencoder_train()
