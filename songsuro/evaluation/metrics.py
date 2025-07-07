import torch
import torchcrepe
from sklearn.metrics import f1_score


def compute_spectrogram_mae(
	wav_mel_gt: torch.tensor = None, wav_mel_gen: torch.tensor = None
):
	assert wav_mel_gt.shape == wav_mel_gen.shape, "Spectrogram shapes do not match."
	mae = torch.mean(torch.abs(wav_mel_gt - wav_mel_gen)).item()
	return mae


def compute_pitch_periodicity(
	wav_mel_gt: torch.tensor = None,
	wav_mel_gen: torch.tensor = None,
	sample_rate: int = 24_000,
	hop_length: int = 1024,
	fmin: int = 50,
	fmax: int = 1100,
	threshold=0.5,
	device="cpu",
):
	wav_gt = wav_mel_gt.unsqueeze(0).to(device)
	wav_gen = wav_mel_gen.unsqueeze(0).to(device)

	pitch_gt, periodicity_gt = torchcrepe.predict(
		wav_gt,
		sample_rate,
		hop_length,
		fmin,
		fmax,
		model="full",
		return_periodicity=True,
		device=device,
	)
	pitch_gen, periodicity_gen = torchcrepe.predict(
		wav_gen,
		sample_rate,
		hop_length,
		fmin,
		fmax,
		model="full",
		return_periodicity=True,
		device=device,
	)

	assert pitch_gt.shape == pitch_gen.shape, "Pitch shape do not match."
	assert (
		periodicity_gt.shape == periodicity_gen.shape
	), "Periodicity shape do not match."

	assert (
		pitch_gt.shape == periodicity_gt.shape
	), "Pitch and periodicity shapes do not match."

	# Voiced mask: periodicity > threshold (threshold는 데이터에 따라 조정)
	voiced_mask = (periodicity_gt > threshold)[0]
	# Pitch error (cent 단위, voiced 구간만)
	pitch_error = torch.sqrt(
		torch.mean(
			(
				1200
				* (
					torch.log2(pitch_gt[0, voiced_mask])
					- torch.log2(pitch_gen[0, voiced_mask])
				)
			)
			** 2
		)
	).item()
	# Periodicity error (voiced 구간만)
	periodicity_error = torch.sqrt(
		torch.mean(
			(periodicity_gt[0, voiced_mask] - periodicity_gen[0, voiced_mask]) ** 2
		)
	).item()
	return (
		pitch_error,
		periodicity_error,
		voiced_mask,
		periodicity_gt,
		periodicity_gen,
	)


def compute_vuv_f1(periodicity_gt, periodicity_gen, threshold=0.5):
	# Voiced/unvoiced 판정
	vuv_gt = (periodicity_gt > threshold).cpu().numpy().astype(int)
	vuv_syn = (periodicity_gen > threshold).cpu().numpy().astype(int)
	# F1 score 계산
	f1 = f1_score(vuv_gt, vuv_syn)
	return f1


"""
Sample usage in melody evaluation:

def evaluate(wav_mel_gt: torch.tensor = None, wav_mel_gen: torch.tensor = None, device="cpu"):
    # 스펙트로그램 MAE
    mae = compute_spectrogram_mae(wav_mel_gt, wav_mel_gen)
    # Pitch, Periodicity, V/UV
    pitch_error, periodicity_error, voiced_mask, periodicity_gt, periodicity_gen = (
        compute_pitch_periodicity(wav_mel_gt, wav_mel_gen, device=device)
    )
    # V/UV F1
    f1 = compute_vuv_f1(periodicity_gt[0], periodicity_gen[0])
    return {
        "Spectrogram_MAE": mae,
        "Pitch_Error": pitch_error,
        "Periodicity_Error": periodicity_error,
        "VUV_F1": f1,
    }
"""
