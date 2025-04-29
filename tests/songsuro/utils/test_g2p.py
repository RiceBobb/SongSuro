import pytest

from songsuro.utils.g2p import NeuralG2P


@pytest.fixture
def g2p_model():
	return NeuralG2P()


def test_g2p_en(g2p_model):
	lyrics = [
		"Home run Minesota Twins Byoung Hooo",
		"Tommy Hyun Soo Edman Home Run",
		"For the king for the sword for the bears",
	]
	result = g2p_model.encode(texts=lyrics, language="eng-us")
	assert len(result) == len(lyrics)
	assert isinstance(result[0], str)
