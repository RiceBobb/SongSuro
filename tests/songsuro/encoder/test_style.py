import pytest
import torch
from songsuro.encoder.style import StyleEncoder, weights_init


class TestStyleEncoder:
	@pytest.fixture
	def hparams(self):
		return {
			"hidden_size": 192,
			"vq_ph_channel": 64,
			"vq": "cvq",
			"vq_ph_codebook_dim": 512,
			"vq_ph_beta": 0.25,
		}

	@pytest.fixture
	def model(self, hparams):
		return StyleEncoder(hparams)

	@pytest.fixture
	def mel_input(self):
		batch_size = 2
		n_mels = 80
		time_length = 100
		# Random values in a typical mel-spectrogram range
		return torch.randn(batch_size, n_mels, time_length) * 4 - 2

	@pytest.fixture
	def nonpadding(self, mel_input):
		# Create a mask for non-padding regions
		batch_size, _, time_length = mel_input.shape
		mask = torch.ones((batch_size, 1, time_length))
		# Simulate some padding in the second sample
		if batch_size > 1:
			pad_length = 20
			mask[1, :, time_length - pad_length :] = 0
		return mask

	@pytest.fixture
	def mel2ph(self, mel_input):
		# Create mel2ph mapping (each mel frame to corresponding phoneme)
		batch_size, _, time_length = mel_input.shape
		mel2ph = torch.zeros(batch_size, time_length, dtype=torch.long)
		# Simulate a sequence of phonemes
		max_ph_length = 30
		ph_lengths = torch.tensor([max_ph_length, max_ph_length - 5])

		# Create realistic mel2ph mapping
		for b in range(batch_size):
			ph_len = ph_lengths[b]
			frames_per_ph = time_length // ph_len
			for i in range(ph_len):
				start_idx = i * frames_per_ph
				end_idx = (i + 1) * frames_per_ph if i < ph_len - 1 else time_length
				mel2ph[b, start_idx:end_idx] = i + 1  # 1-indexed

		return mel2ph

	@pytest.fixture
	def ph_nonpadding(self, mel2ph):
		# Create phoneme-level padding mask
		batch_size, time_length = mel2ph.shape
		max_ph_length = mel2ph.max().item()
		mask = torch.zeros(batch_size, 1, max_ph_length)

		for b in range(batch_size):
			ph_length = mel2ph[b].max().item()
			mask[b, :, :ph_length] = 1

		return mask

	@pytest.fixture
	def ph_lengths(self, mel2ph):
		# Get phoneme lengths for each batch
		batch_size = mel2ph.shape[0]
		ph_lengths = torch.zeros(batch_size, dtype=torch.long)

		for b in range(batch_size):
			ph_lengths[b] = mel2ph[b].max().item()

		return ph_lengths

	def test_weights_init(self):
		# Test the weights_init function
		conv = torch.nn.Conv1d(10, 10, 3)
		weights_before = conv.weight.clone()
		weights_init(conv)
		assert not torch.allclose(weights_before, conv.weight)
		assert torch.all(conv.bias == 0)

	def test_encoder_initialization(self, hparams):
		# Test that the encoder initializes correctly
		encoder = StyleEncoder(hparams)
		assert encoder.hidden_size == hparams["hidden_size"]
		assert encoder.vq_ph_channel == hparams["vq_ph_channel"]
		assert isinstance(encoder.ph_conv_in, torch.nn.Conv1d)
		assert isinstance(encoder.ph_encoder, torch.nn.Module)
		assert isinstance(encoder.ph_postnet, torch.nn.Module)

	def test_encode_ph_vqcode(
		self, model, mel_input, nonpadding, mel2ph, ph_nonpadding
	):
		# Test the encode_ph_vqcode method
		max_ph_length = mel2ph.max().item()

		# Forward pass
		ph_vqcode = model.encode_ph_vqcode(
			mel_input, nonpadding, mel2ph, max_ph_length, ph_nonpadding
		)

		# Check output shape and type
		assert isinstance(ph_vqcode, torch.Tensor)
		assert ph_vqcode.dim() == 2
		assert ph_vqcode.shape[0] == mel_input.shape[0]
		assert ph_vqcode.shape[1] == max_ph_length
		assert ph_vqcode.dtype == torch.long

	def test_vqcode_to_latent(
		self, model, mel_input, nonpadding, mel2ph, ph_nonpadding
	):
		# Test the vqcode_to_latent method
		max_ph_length = mel2ph.max().item()

		# Get vqcode first
		ph_vqcode = model.encode_ph_vqcode(
			mel_input, nonpadding, mel2ph, max_ph_length, ph_nonpadding
		)

		# Convert back to latent
		ph_z_q_x_bar = model.vqcode_to_latent(ph_vqcode)

		# Check output shape
		assert ph_z_q_x_bar.shape[0] == mel_input.shape[0]
		assert ph_z_q_x_bar.shape[1] == model.hidden_size
		assert ph_z_q_x_bar.shape[2] == max_ph_length

	def test_encode_style(
		self, model, mel_input, nonpadding, mel2ph, ph_nonpadding, ph_lengths
	):
		# Test the encode_style method

		# Forward pass
		ph_z_q_x_st, vq_loss, indices = model.encode_style(
			mel_input, nonpadding, mel2ph, ph_nonpadding, ph_lengths
		)

		# Check output shapes and types
		assert ph_z_q_x_st.shape[0] == mel_input.shape[0]
		assert ph_z_q_x_st.shape[1] == model.hidden_size
		assert ph_z_q_x_st.shape[2] == ph_lengths.max().item()

		assert isinstance(vq_loss, torch.Tensor)
		assert vq_loss.dim() == 0  # scalar

		assert indices.shape[0] == mel_input.shape[0]
		assert indices.shape[1] == ph_lengths.max().item()
		assert indices.dtype == torch.long

	def test_end_to_end(
		self, model, mel_input, nonpadding, mel2ph, ph_nonpadding, ph_lengths
	):
		# Test the full pipeline: encode style -> get vqcode -> decode back

		# 1. Encode style
		ph_z_q_x_st, vq_loss, indices = model.encode_style(
			mel_input, nonpadding, mel2ph, ph_nonpadding, ph_lengths
		)

		# 2. Convert indices back to latent
		ph_z_q_x_bar = model.vqcode_to_latent(indices)

		# 3. Check that the reconstructed latent is close to the original
		assert ph_z_q_x_st.shape == ph_z_q_x_bar.shape
		# The reconstructed latent should be identical to the original since VQ is deterministic
		assert torch.allclose(ph_z_q_x_st, ph_z_q_x_bar, rtol=1e-5, atol=1e-7)
