from typing import List

import torch
from torch.utils.data import DataLoader, default_collate


class BaseDataLoader(DataLoader):
	def __init__(
		self,
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=6,
		pad_mode="constant",
		pad_value=0,
		**kwargs,
	):
		self.pad_mode = pad_mode
		self.pad_value = pad_value

		super().__init__(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			collate_fn=self._collate_fn,
			**kwargs,
		)

	def _pad_tensor(self, tensor, max_length: int):
		padding_size = max_length - tensor.shape[-1]
		padded_tensor = torch.nn.functional.pad(
			tensor, (0, padding_size), mode=self.pad_mode, value=self.pad_value
		)
		return padded_tensor

	def _pad_to_same_length(self, tensor_list: List[torch.Tensor]):
		max_length = max(elem.shape[-1] for elem in tensor_list)
		length_list = []
		padded_tensor_list = []
		for elem in tensor_list:
			length_list.append(elem.shape[-1])
			padded_tensor_list.append(self._pad_tensor(elem, max_length))

		result = torch.stack(padded_tensor_list)
		return torch.squeeze(result, dim=1), length_list

	def _collate_fn(self, batch):
		audios = list(map(lambda x: x["audio"], batch))
		sample_rates = list(map(lambda x: x["sample_rate"], batch))
		mels = list(map(lambda x: x["mel_spectrogram"], batch))
		f0s = list(map(lambda x: x["f0"], batch))

		batch_audios, batch_audio_lengths = self._pad_to_same_length(audios)
		batch_mels, batch_mel_lengths = self._pad_to_same_length(mels)
		batch_f0s, batch_f0s_lengths = self._pad_to_same_length(f0s)

		result = {
			"audio": batch_audios,  # [Batch, Length]
			"audio_lengths": torch.tensor(batch_audio_lengths),  # [B]
			"sample_rates": torch.tensor(sample_rates),  # [B]
			"mel_spectrogram": batch_mels,  # [B, C, L]
			"f0": batch_f0s,  # [B, L]
			"mel_spectrogram_lengths": torch.tensor(batch_mel_lengths),
			"f0_lengths": torch.tensor(batch_mel_lengths),
		}

		# Collate other data other than audio and sample rate
		other_data = {}
		for key in batch[0].keys():
			if key not in ["audio", "sample_rate", "mel_spectrogram", "f0"]:
				other_data[key] = default_collate([item[key] for item in batch])

		result.update(other_data)
		return result
