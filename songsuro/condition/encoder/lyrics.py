from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from songsuro.utils.g2p import NeuralG2P, SUPPORT_LANGUAGE


class LyricsEncoder:
	def __init__(self, language: SUPPORT_LANGUAGE = "kor"):
		self.xphonebert_model = AutoModel.from_pretrained("vinai/xphonebert-base")
		self.tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
		self.g2p_model = NeuralG2P()
		self.language = language

	def inference(self, lyrics_list: List[str]) -> torch.Tensor:
		phoneme_list = self.g2p_model.encode(lyrics_list, self.language)

		tokens = self.tokenizer(phoneme_list, return_tensors="pt", padding=True)
		result = self.xphonebert_model(**tokens)

		return torch.mean(result.last_hidden_state, dim=1)
