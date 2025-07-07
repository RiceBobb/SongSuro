import torch
import torchcrepe
from sklearn.metrics import f1_score
from songsuro.preprocess import load_audio, make_spectrogram, make_mel_spectrogram


class MelodyLoss:
	def __init__(self, wav_gt_path=None, wav_gen_path=None):
		if wav_gt_path is not None and wav_gen_path is not None:
			self.wav_gen = load_audio(wav_gen_path)
			self.wav_gt = load_audio(wav_gt_path)
		else:
			raise ValueError(
				"wav_gt_path and wav_gen_path must be provided to initialize MelodyEvaluator."
			)

	def compute_spectrogram_mae(self):
		spec_wav_gt = make_spectrogram(self.wav_gt)
		spec_wav_gen = make_spectrogram(self.wav_gen)

		S_gt = torch.from_numpy(make_mel_spectrogram(spec_wav_gt)).float()
		S_syn = torch.from_numpy(make_mel_spectrogram(spec_wav_gen)).float()

		assert S_gt.shape == S_syn.shape, "Spectrogram shapes do not match."

		# # 길이 맞추기 -> why? is it necessary?
		# min_len = min(S_gt.shape[-1], S_syn.shape[-1])
		# S_gt, S_syn = S_gt[..., :min_len], S_syn[..., :min_len]

		# MAE 계산
		mae = torch.mean(torch.abs(S_gt - S_syn)).item()
		return mae

	def compute_pitch_periodicity(
		self,
		sample_rate: int = 24_000,
		hop_length: int = 1024,
		fmin: int = 50,
		fmax: int = 1100,
		threshold=0.5,
		device="cpu",
	):
		# TODO: 리샘플은 왜 나와? 논문대로 구현한게 맞나 validation

		wav_gt = torch.from_numpy(self.wav_gt).float().unsqueeze(0).to(device)
		wav_gen = torch.from_numpy(self.wav_gen).float().unsqueeze(0).to(device)

		# 피치 및 주기성 추출
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

		# # 길이 맞추기
		# min_len = min(pitch_gt.shape[-1], pitch_syn.shape[-1])
		# pitch_gt, pitch_syn = pitch_gt[..., :min_len], pitch_syn[..., :min_len]
		# periodicity_gt, periodicity_syn = periodicity_gt[..., :min_len], periodicity_syn[..., :min_len]

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

	def compute_vuv_f1(self, periodicity_gt, periodicity_gen, threshold=0.5):
		# Voiced/unvoiced 판정
		vuv_gt = (periodicity_gt > threshold).cpu().numpy().astype(int)
		vuv_syn = (periodicity_gen > threshold).cpu().numpy().astype(int)
		# F1 score 계산
		f1 = f1_score(vuv_gt, vuv_syn)
		return f1

	def evaluate(self, device="cpu"):
		# 스펙트로그램 MAE
		mae = self.compute_spectrogram_mae()
		# Pitch, Periodicity, V/UV
		pitch_error, periodicity_error, voiced_mask, periodicity_gt, periodicity_gen = (
			self.compute_pitch_periodicity(device=device)
		)
		# V/UV F1
		f1 = self.compute_vuv_f1(periodicity_gt[0], periodicity_gen[0])
		return {
			"Spectrogram_MAE": mae,
			"Pitch_Error": pitch_error,
			"Periodicity_Error": periodicity_error,
			"VUV_F1": f1,
		}
