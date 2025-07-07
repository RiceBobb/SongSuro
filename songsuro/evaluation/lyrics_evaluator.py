import whisper
import ssl

# SSL Certificate problem workaround (for environments with self-signed certificates)
ssl._create_default_https_context = ssl._create_unverified_context


class LyricsEMEvaluator:
	def __init__(self, model_name="base"):
		self.model = whisper.load_model(model_name)

	def transcribe_audio(self, audio_path, language=None):
		"""
		Extract lyrics from an audio file using Whisper.
		If you want to specify a language, replace `language` with the desired language code (e.g., 'en' for English).
		You can refer to the https://github.com/openai/whisper/blob/main/whisper/tokenizer.py for supported languages.
		"""
		if language is None:
			result = self.model.transcribe(audio=audio_path)
		else:
			result = self.model.transcribe(audio=audio_path, language=language)
		return result["text"]

	def normalize_text(self, text):
		"""
		Normalize text: lowercase, strip punctuation, and remove extra spaces.
		Adjust as needed.
		"""
		text = text.lower().strip()
		# Remove punctuation, special characters, and extra spaces
		import re

		text = re.sub(r"[^\w\s]", "", text)
		text = re.sub(r"\s+", " ", text)
		return text

	def evaluate(self, audio_path, reference_lyrics, normalize=True):
		"""
		Args:
		    audio_path: Path to the audio file
		    reference_lyrics: Reference lyrics (str)
		    normalize: Whether to ignore case/punctuation/whitespace (default: True)
		Returns:
		    (bool, str): (Exact match or not, transcribed lyrics)
		"""
		transcribed_lyrics = self.transcribe_audio(audio_path)
		if normalize:
			transcribed_lyrics = self.normalize_text(transcribed_lyrics)
			reference_lyrics = self.normalize_text(reference_lyrics)
		is_match = transcribed_lyrics == reference_lyrics
		return is_match, transcribed_lyrics
