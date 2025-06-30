import pytest
import torch

from songsuro.condition.encoder.pre_trained_lyrics import PretrainedLyricsEncoder


@pytest.mark.skip()
def test_inference_korean():
	encoder = PretrainedLyricsEncoder("kor")
	lyrics = [
		"리중딱 리중딱 신나는 노래 나도 한 번 불러 본다",
		"최강 두산 오명진 안타를 날려라 최강 두산 오명진 홈런을 날려라",
		"최강 두산 케이브 최강 두산 케이브 승리를 위해 모두 외쳐라"
		"승리를 위해 힘차게 날려라 오오 오오오 오오오 허경민",
	]
	result = encoder.inference(lyrics)
	assert isinstance(result, torch.Tensor)
	assert result.shape == (len(lyrics), 768)


@pytest.mark.skip()
def test_inference_english():
	encoder = PretrainedLyricsEncoder("eng-us")
	lyrics = [
		"Home run Minesota Twins Byoung Hooo",
		"Tommy Hyun Soo Edman Home Run",
		"For the king for the sword for the bears",
	]
	result = encoder.inference(lyrics)
	assert isinstance(result, torch.Tensor)
	assert result.shape == (len(lyrics), 768)


def cosine_similarity(tensor1, tensor2):
	# Normalize the tensors
	tensor1_norm = tensor1 / tensor1.norm(dim=-1, keepdim=True)
	tensor2_norm = tensor2 / tensor2.norm(dim=-1, keepdim=True)
	# Compute cosine similarity
	similarity = (tensor1_norm * tensor2_norm).sum(dim=-1)
	return similarity


@pytest.mark.skip()
def test_similarity_en():
	encoder = PretrainedLyricsEncoder("eng-us")
	lyrics = [
		"I scream you scream everybody ice cream",
		"Ice cream you screen everybody I scream",
		"Havertz is the king of the Arsenal Football Club",
	]
	result = encoder.inference(lyrics)

	embeddings = torch.split(result, 1, dim=0)
	sim1 = cosine_similarity(embeddings[0].squeeze(0), embeddings[1].squeeze(0))
	sim2 = cosine_similarity(embeddings[0].squeeze(0), embeddings[2].squeeze(0))

	assert sim1 > sim2
