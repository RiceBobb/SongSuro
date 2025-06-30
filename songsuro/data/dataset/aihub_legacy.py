import glob
import json
import os

import torchaudio
from torch.utils.data import Dataset

from songsuro.condition.encoder.melodyU import preprocess_f0


class AIHubLegacyDataset(Dataset):
	def __init__(self, root_dir: str):
		"""

		:param root_dir: The root directory of the AI hub dataset.
			Have to contain '라벨링데이터' and '원천데이터' folders in the root_dir.
		"""
		self.root_dir = root_dir
		self.wav_file_list = []
		for wav_file in glob.iglob(
			os.path.join(root_dir, "원천데이터", "**", "*.wav"), recursive=True
		):
			self.wav_file_list.append(wav_file)

	def __len__(self):
		return len(self.wav_file_list)

	def __getitem__(self, idx):
		wav_filepath = self.wav_file_list[idx]
		metadata = self.extract_metadata_from_path(wav_filepath)
		audio, sample_rate = torchaudio.load(wav_filepath)
		mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=sample_rate,
			n_fft=2048,
			hop_length=1024,
			f_max=8000,
		)
		mel_spectrogram = mel_spectrogram_transform(audio)

		# F0 Contour
		f0 = preprocess_f0(audio, sample_rate)

		# Lyrics & Label
		rel_path = os.path.relpath(wav_filepath, start=self.root_dir)
		parts = rel_path.split(os.sep)
		label_path = os.path.join(
			self.root_dir,
			"라벨링데이터",
			parts[1],
			parts[2],
			parts[3],
			parts[4],
			parts[5],
			parts[-1].split(".")[0] + ".json",
		)
		with open(label_path, "r") as f:
			label = json.load(f)
		lyrics_list = list(
			map(lambda x: x["lyric"] if x["lyric"] else " ", label["notes"])
		)
		lyrics = "".join(lyrics_list)

		return {
			"audio": audio,
			"sample_rate": sample_rate,
			"mel_spectrogram": mel_spectrogram,
			"audio_filepath": wav_filepath,
			"label_filepath": label_path,
			"lyrics": lyrics,
			"f0": f0,
			"metadata": metadata,
		}

	def extract_metadata_from_path(self, filepath):
		# Get relative path from the root directory
		rel_path = os.path.relpath(filepath, start=self.root_dir)
		parts = rel_path.split(os.sep)

		return {
			"root_type": parts[0],
			"gender": parts[1],
			"age_group": parts[2],
			"genre": parts[3],
			"timbre": parts[4],
			"singer_id": parts[5],
			"filepath": rel_path,
		}
