import torch
from torch.utils.data import DataLoader, default_collate


class BaseDataLoader(DataLoader):
	def __init__(
		self,
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=0,
		pad_mode="constant",
		pad_value=-1,
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

	def _pad_audio(self, audio, max_length: int):
		padding_size = max_length - audio.shape[-1]
		padded_audio = torch.nn.functional.pad(
			audio, (0, padding_size), mode=self.pad_mode, value=self.pad_value
		)
		return padded_audio

	def _collate_fn(self, batch):
		audios = list(map(lambda x: x["audio"], batch))
		sample_rates = list(map(lambda x: x["sample_rate"], batch))

		max_length = max(waveform.shape[-1] for waveform in audios)

		batch_waveforms = []
		batch_lengths = []

		for audio in audios:
			length = audio.shape[-1]
			batch_lengths.append(length)
			batch_waveforms.append(self._pad_audio(audio, max_length))

		batch_waveforms = torch.stack(batch_waveforms)
		batch_lengths = torch.tensor(batch_lengths)
		batch_sample_rates = torch.tensor(sample_rates)

		result = {
			"audio": batch_waveforms,  # [Batch, Channel, Length]
			"audio_lengths": batch_lengths,  # [B]
			"sample_rates": batch_sample_rates,  # [B]
		}

		# Collate other data other than audio and sample rate
		other_data = {}
		for key in batch[0].keys():
			if key not in ["audio", "sample_rate"]:
				other_data[key] = default_collate([item[key] for item in batch])

		result.update(other_data)
		return result
