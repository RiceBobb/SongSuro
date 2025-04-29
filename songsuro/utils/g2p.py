import logging
from typing import List, Literal

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from g2p_en import G2p as G2pEn
from g2pk import G2p as G2pKo

from songsuro.utils.util import normalize_string, nested_map


logger = logging.getLogger("SongSuro")


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
			# TODO: implement ko-kiwi - Feature/#38
			split_words = []
			raise NotImplementedError
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


def naive_g2p(lyrics_list: List[str], language: str = "ko") -> List[str]:
	"""
	Converting grapheme to phoneme using g2p library.

	:param lyrics_list: The list of lyrics.
	:param language: The language of the lyrics. Default is 'ko'.
	"""
	phonemes_lst = []

	if language == "ko":
		g2p = G2pKo()
	elif language == "en":
		g2p = G2pEn()
	else:
		raise ValueError(f"Unsupported language: {language}")

	for lyric in tqdm(lyrics_list):
		phonemes_lst.append(g2p(lyric))

	if len(phonemes_lst) == 0:
		logger.warning("No phonemes were generated. Please check the input lyrics.")

	return phonemes_lst
