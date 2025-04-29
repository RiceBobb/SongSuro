from typing import List, Literal

from transformers import T5ForConditionalGeneration, AutoTokenizer

from songsuro.utils.util import normalize_string, nested_map


class NeuralG2P:
	def __init__(self):
		self.model = T5ForConditionalGeneration.from_pretrained(
			"charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
		)
		self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

	def encode(self, texts: List[str], language: Literal["eng-us", "eng-uk", "kor"]):
		normalized_texts = list(map(normalize_string, texts))
		if language in ["eng-us", "eng-uk"]:
			split_words = list(map(lambda x: x.split(), normalized_texts))
		elif language == "kor":
			# TODO: implement ko-kiwi
			split_words = []
		else:
			raise ValueError("Language must be 'eng-us', 'eng-uk' or 'kor'")

		split_words = nested_map(split_words, lambda x: f"<{language}>: " + x)

		result = []
		for words in split_words:
			tokens = self.tokenizer(
				words, padding=True, add_special_tokens=False, return_tensors="pt"
			)

			prediction = self.model.generate(**tokens, num_beams=1, max_length=50)
			phonemes = self.tokenizer.batch_decode(
				prediction.tolist(), skip_special_tokens=True
			)
			result.append(" ".join(phonemes))

		return result
