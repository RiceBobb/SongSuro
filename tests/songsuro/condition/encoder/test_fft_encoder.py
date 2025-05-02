import torch
import pytest

from songsuro.condition.encoder.fft import FFTEncoder


@pytest.mark.parametrize(
	"batch_size, seq_len, input_vec_size, hidden_channels",
	[
		(2, 16, 100, 192),
		(4, 32, 50, 256),
	],
)
def test_fft_encoder_forward_shape(
	batch_size, seq_len, input_vec_size, hidden_channels
):
	x = torch.randint(0, input_vec_size, (batch_size, seq_len), dtype=torch.long)
	x_lengths = torch.randint(
		seq_len // 2, seq_len + 1, (batch_size,), dtype=torch.long
	)

	default_model = FFTEncoder(
		input_vec_size=input_vec_size,
		out_channels=hidden_channels,
		window_size=None,
		block_length=None,
	)

	model = FFTEncoder(
		input_vec_size=input_vec_size,
		out_channels=hidden_channels,
		hidden_channels=hidden_channels,
		filter_channels=hidden_channels * 4,
		n_heads=2,
		n_layers=2,
		kernel_size=9,
		p_dropout=0.1,
		window_size=4,
		block_length=None,
		mean_only=False,
		gin_channels=0,
	)

	default_output = default_model(x, x_lengths)
	output = model(x, x_lengths)

	# Output shape: [batch, hidden, time(seq_len)]
	assert default_output.shape[0] == batch_size
	assert default_output.shapep[1] == hidden_channels
	assert default_output.shape[2] == seq_len

	assert output.shape[0] == batch_size
	assert output.shape[1] == hidden_channels
	assert output.shape[2] == seq_len


def test_fft_encoder_padding_mask():
	batch_size, seq_len, input_vec_size, hidden_channels = 2, 10, 50, 64
	x = torch.randint(0, input_vec_size, (batch_size, seq_len), dtype=torch.long)
	x_lengths = torch.tensor([6, 10], dtype=torch.long)  # 첫 번째 샘플은 패딩 포함

	model = FFTEncoder(
		input_vec_size=input_vec_size,
		out_channels=hidden_channels,
		hidden_channels=hidden_channels,
		filter_channels=hidden_channels * 4,
		filter_channels_dp=hidden_channels * 4,
		n_heads=2,
		n_layers=2,
		kernel_size=3,
		p_dropout=0.0,
		window_size=2,
		block_length=None,
		mean_only=False,
		gin_channels=0,
	)

	output = model(x, x_lengths)
	# 패딩 위치는 0이어야 함
	mask = torch.arange(seq_len).unsqueeze(0) < x_lengths.unsqueeze(1)
	mask = mask.unsqueeze(1).to(output.dtype)  # [batch, 1, seq_len]
	# 패딩 위치의 값이 0인지 확인
	assert torch.all(output[0, :, 6:] == 0)


def test_fft_encoder_gradients():
	batch_size, seq_len, input_vec_size, hidden_channels = 2, 8, 30, 32
	x = torch.randint(0, input_vec_size, (batch_size, seq_len), dtype=torch.long)
	x_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

	model = FFTEncoder(
		input_vec_size=input_vec_size,
		out_channels=hidden_channels,
		hidden_channels=hidden_channels,
		filter_channels=hidden_channels * 4,
		filter_channels_dp=hidden_channels * 4,
		n_heads=2,
		n_layers=4,
		kernel_size=3,
		p_dropout=0.0,
		window_size=1,
		block_length=None,
		mean_only=False,
		gin_channels=0,
	)

	output = model(x, x_lengths)
	loss = output.sum()
	loss.backward()

	for name, param in model.named_parameters():
		if param.requires_grad:
			assert param.grad is not None, f"Gradient missing for {name}"
