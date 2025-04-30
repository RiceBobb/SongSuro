from typing import List

import torch
from transformers import AutoTokenizer

from songsuro.utils.g2p import NeuralG2P, SUPPORT_LANGUAGE


class LyricsEncoder:
	def __init__(self, language: SUPPORT_LANGUAGE = "kor"):
		# TODO: add real model
		self.tokenizer = AutoTokenizer.from_pretrained(
			"vinai/xphonebert-base"
		)  # I will use this tokenizer anyway (for phoneme)
		self.g2p_model = NeuralG2P()
		self.language = language

	def inference(self, lyrics_list: List[str]) -> torch.Tensor:
		_ = self.g2p_model.encode(lyrics_list, self.language)
		return torch.Tensor([])
