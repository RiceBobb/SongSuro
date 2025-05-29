import torch
from torch import nn

from songsuro.condition.encoder.fft import FFTEncoder
from songsuro.condition.prior_estimator import PriorEstimator


class ConditionalEncoder(nn.Module):
	def __init__(
		self,
		lyrics_input_channel: int,
		melody_input_channel: int,
		prior_output_dim: int,
		hidden_size: int = 192,
	):
		super().__init__()

		self.lyrics_encoder = FFTEncoder(lyrics_input_channel)
		# self.melody_encoder = FFTEncoder(melody_input_channel)
		# self.melodyU_encoder = FFTEncoder(melody_input_channel)

		# self.timbre_encoder = TimbreEncoder(hidden_size=hidden_size, vq_input_dim=80)
		# self.style_encoder = StyleEncoder(hidden_size=hidden_size)

		self.enhanced_condition_encoder = FFTEncoder()
		self.prior_estimator = PriorEstimator(hidden_size, prior_output_dim)

	def forward(
		self,
		lyrics,
		lyrics_lengths: torch.Tensor,
		quantized_f0,
		quantized_f0_lengths: torch.Tensor,
		note_durations: torch.Tensor,
		note_durations_lengths: torch.Tensor,
	):
		"""
		Forward pass of the conditional encoder.
		Given original audio and the lyrics, it generates embedding vectors with timbre, lyrics, melody and style.

		:param lyrics: Tokenized lyrics sequence. Should be a tensor or can be converted to one.
		:param lyrics_lengths: Lengths of the lyrics sequences. The shape is [batch_size]

		:param quantized_f0: Pre-processed quantized f0 data.
		:param quantized_f0_lengths: Lengths of the quantized f0 sequences. The shape is [batch_size].

		:param note_durations: The duration of the note in frames. The shape is [batch_size].
		:param note_durations_lengths: Lengths of the note durations sequences. The shape is [batch_size].

		:return: conditional embedding vector and prior
		"""
		if not isinstance(lyrics, torch.Tensor):
			raise TypeError("Input lyrics must be a tensor")

		lyrics_embedding = self.lyrics_encoder(lyrics, lyrics_lengths)

		# if not isinstance(quantized_ f0, torch.Tensor):
		#     raise TypeError("Input lyrics must be a tensor")
		# quantized_f0 = torch.tensor(quantized_f0).unsqueeze(0)

		# TODO: We need to apply pitch embedding and then encode it with FFTEncoder
		# melodyU_embedding = self.melodyU_encoder(quantized_f0, quantized_f0_lengths)
		# timbre_embedding = self.timbre_encoder(lyrics)
		# style_embedding = self.style_encoder(lyrics)

		"""TODO: Expand lyrics and melody embedding to match the length of frame-level based on note durations of mel-spectrogram.
			-> Furthermore, We can use length regulator to improve matching phoneme and note duration! (DiffSinger)
		"""
		lyrics_embedding, embedding_length = self.expand_embeddings_to_frame_level(
			lyrics_embedding, note_durations
		)
		# melody_embedding = self.enhanced_condition_encoder(melody_embedding, note_durations)

		summation_embedding = (
			lyrics_embedding
			# + melodyU_embedding
			# + timbre_embedding + style_embedding
		)

		# Summation length will be frame-level based on the pitch duration
		enhanced_condition_embedding = self.enhanced_condition_encoder(
			summation_embedding, embedding_length
		)

		prior = self.prior_estimator(torch.mean(enhanced_condition_embedding, -1))

		return enhanced_condition_embedding, prior

	def expand_embeddings_to_frame_level(self, embeddings, durations):
		"""
		Expand [batch_size, hidden, seq_len] embeddings to [batch_size, hidden, total_frames] by repeating along time axis.
		Args:
		    embeddings: torch tensor of shape [batch_size, hidden, seq_len]
		    durations:  torch tensor of shape [batch_size, seq_len], number of frames for each time step
		Returns:
		    Expanded torch tensor: [batch_size, hidden, max_total_frames], max_length
		"""
		batch_size, hidden_size, seq_len = embeddings.shape
		expanded_embeddings = []

		# Iterate over each batch and expand the embeddings
		for b in range(batch_size):
			expanded_seq = []
			for t in range(seq_len):
				repeated = (
					embeddings[b, :, t].unsqueeze(1).repeat(1, durations[b, t].item())
				)
				expanded_seq.append(repeated)
			expanded_seq = torch.cat(expanded_seq, dim=1)  # [hidden, total_frames]
			expanded_embeddings.append(expanded_seq)

		# Pad the expanded embeddings to have the same length
		max_length = max(e.shape[1] for e in expanded_embeddings)
		padded_embeddings = torch.zeros(
			(batch_size, hidden_size, max_length), dtype=embeddings.dtype
		)
		for i, e in enumerate(expanded_embeddings):
			padded_embeddings[i, :, : e.shape[1]] = e
		return padded_embeddings, max_length
