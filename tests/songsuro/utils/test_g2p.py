import pytest
from unittest.mock import patch

from songsuro.utils.g2p import NeuralG2P, naive_g2p


KO_MANGGASONG = """
맨까 새끼들 부들부들하구나
억까를 해봐도 우린 골 넣지
니네가 아무리 맹구다 어쩐다고 놀려도
아아~ 즐겁구나 명 절 이~(짜스!)[9]

맨까 새끼들 부들부들하구나
이번 시즌 잘하잖아 억까하지 마
이겨도 지롤 져도 지롤 뭐만 하면 리그컵
아아~ 리그컵도 축 군 데~(컴온!!)

맨까 새끼들 부들부들하구나
돌아온 미친 폼 누가 막을래?
더보기 리그 탈출직전 돌아와요 맨유 팬
아아~ 기대된다 챔 스 가~ Siuuuuuuu![원곡]
"""

KO_LIJUNGDDAK = """
안하긴뭘안해~~ 반갑습니다~~
이피엘에서 우승못하는팀 누구야? 소리질러~~!!
리중딱 리중딱 신나는노래~ 나도한번 불러본다~~(박수) (박수) (박수)
짠리잔짠~~ 우리는 우승하기 싫~어~ 왜냐면 우승하기 싫은팀이니깐~
20년 내~내~ 프리미어리그~ 우승도 못하는 우리팀이다. 리중딱 리중딱 신나는노래 ~~~
나도한번불러본다~ 리중딱 리중딱 신나는노래 ~~
가슴치며 불러본다~ 리중딱 노래가사는~ 생활과 정보가 있는노래 중딱이~~와 함께라면 제~라드도함께 우승못한다.
"""

EN_DIEWITHSMILE = """
I, I just woke up from a dream
Where you and I had to say goodbye
And I don't know what it all means
But since I survived, I realized

Wherever you go, that's where I'll follow
Nobody's promised tomorrow
So I'ma love you every night like it's the last night
Like it's the last night

If the world was ending
I'd wanna be next to you
If the party was over
And our time on Earth was through
I'd wanna hold you just for a while
And die with a smile
If the world was ending
I'd wanna be next to you

Ooh, lost, lost in the words that we scream
I don't even wanna do this anymore
'Cause you already know what you mean to me
And our love's the only war worth fighting for

Wherever you go, that's where I'll follow
Nobody's promised tomorrow
So I'ma love you every night like it's the last night
Like it's the last night
"""


@pytest.fixture
def g2p_model():
	return NeuralG2P()


def test_g2p_en(g2p_model):
	lyrics = [
		"Home run Minesota Twins Byoung Hooo",
		"Tommy Hyun Soo Edman Home Run",
		"For the king for the sword for the bears",
		EN_DIEWITHSMILE,
	]
	result = g2p_model.encode(texts=lyrics, language="eng-us")
	assert len(result) == len(lyrics)
	assert isinstance(result[0], str)


def test_convert_g2p():
	ko_lyrics = [KO_LIJUNGDDAK, KO_MANGGASONG]
	en_lyrics = [EN_DIEWITHSMILE]

	phonemes_en = naive_g2p(en_lyrics, language="en")
	phonemes_ko = naive_g2p(ko_lyrics, language="ko")

	validate_phonemes(phonemes_ko, ko_lyrics)
	validate_phonemes(phonemes_en, en_lyrics)


def test_convert_g2p_invalid_language():
	lyrics = ["Hello world"]
	with pytest.raises(ValueError, match="Unsupported language: es"):
		naive_g2p(lyrics, language="es")


def test_convert_g2p_empty_lst_logs_warning():
	with patch("logging.Logger.warning") as mock_warning:
		naive_g2p([])  # empty list
		mock_warning.assert_called_once_with(
			"No phonemes were generated. Please check the input lyrics."
		)


def validate_phonemes(phonemes, lyrics):
	assert isinstance(phonemes, list), "Phonemes should be a list."
	assert len(phonemes) == len(lyrics), "Phonemes length should match lyrics length."
	assert all(
		isinstance(p, str) for lst in phonemes for p in lst
	), "All phonemes should be strings."
