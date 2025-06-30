import pytest
from unittest.mock import patch
from songsuro.evaluation.lyrics_evaluator import LyricsEMEvaluator


@pytest.fixture
def evaluator():
	return LyricsEMEvaluator(model_name="large-v3")


def test_transcribe_audio_mocked(evaluator):
	# Mock transcribe to return a dict as Whisper does
	with patch.object(
		evaluator.model, "transcribe", return_value={"text": "test lyrics"}
	):
		lyrics = evaluator.transcribe_audio("dummy.wav")
		assert lyrics == "test lyrics"


def test_normalize_text(evaluator):
	text = "Hello, World!  "
	expected = "hello world"
	assert evaluator.normalize_text(text) == expected


def test_evaluate_exact_match(evaluator):
	with patch.object(
		evaluator.model, "transcribe", return_value={"text": "hello world"}
	):
		is_match, lyrics = evaluator.evaluate("dummy.wav", "hello world")
		assert is_match
		assert lyrics == "hello world"


def test_evaluate_not_match(evaluator):
	with patch.object(
		evaluator.model, "transcribe", return_value={"text": "hello world"}
	):
		is_match, lyrics = evaluator.evaluate("dummy.wav", "goodbye world")
		assert not is_match


def test_evaluate_normalized_match(evaluator):
	with patch.object(
		evaluator.model, "transcribe", return_value={"text": "Hello, World!"}
	):
		is_match, lyrics = evaluator.evaluate(
			"dummy.wav", "hello world", normalize=True
		)
		assert is_match
		assert evaluator.normalize_text(lyrics) == "hello world"
