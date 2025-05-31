import pathlib
import tempfile

import pandas as pd
from dotenv import load_dotenv

from songsuro.data.dataset.aihub_lyrics_preprocess import build_lyrics_csv

root_dir = pathlib.Path(__file__).parent.parent.parent.parent
resource_dir = root_dir / "resources"


def test_build_lyrics_csv():
	load_dotenv()

	with tempfile.NamedTemporaryFile(suffix=".csv") as f:
		build_lyrics_csv(str(resource_dir / "ai_hub_data_sample"), f.name)

		result_df = pd.read_csv(f.name)
		assert isinstance(result_df, pd.DataFrame)
		assert set(result_df.columns) == {
			"json_filepath",
			"filename",
			"phoneme",
			"grapheme",
		}
