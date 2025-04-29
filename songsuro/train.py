# Copyright 2025 Rice-Bobb Organization. All Rights Reserved.
# Under MIT License
import os
from typing import Optional

import numpy as np
import torch.optim
import wandb
from torch import nn
from tqdm import tqdm

from songsuro.models import Songsuro
from songsuro.utils.util import nested_map


# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class SongsuroLearner:
	def __init__(
		self,
		model_dir,
		dataset,
		model_kwargs,
		learning_rate=5 * 1e-5,
		betas=(0.8, 0.99),
		prior_lambda: float = 0.5,
		contrastive_lambda: float = 1.0,
		max_grad_norm: Optional[float] = None,
		**kwargs,
	):
		"""
		Initialize Songsuro learner.

		:param model_dir: Directory to save or load the model
		:param dataset: Dataset to use
		:param learning_rate: The learning rate of AdamW Optimizer
			Default is 5 * 1e-5
		:param betas: The betas of AdamW Optimizer
			Default is (0.8, 0.99)
		:param prior_lambda: The prior loss weight to the Songsuro training
			Default is 0.5
		:param contrastive_lambda: The contrastive loss weight to the Songsuro training
			Default is 1.0
		:param max_grad_norm: The maximum gradient norm
		:param model_kwargs: The parameters of Songsuro model
		"""
		os.makedirs(model_dir, exist_ok=True)
		self.model_dir = model_dir
		self.dataset = dataset
		self.device = model_kwargs["device"]
		self.model_kwargs = model_kwargs
		self.model = Songsuro(**model_kwargs).to(self.device)
		self.optimizer = torch.optim.AdamW(
			self.model.parameters(),
			lr=learning_rate,
			betas=betas,
		)
		self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get("fp16", False))
		self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get("fp16", False))
		self.step = 0

		self.prior_lambda = prior_lambda
		self.contrastive_lambda = contrastive_lambda
		self.max_grad_norm = max_grad_norm

	def state_dict(self):
		if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
			model_state = self.model.module.state_dict()
		else:
			model_state = self.model.state_dict()
		return {
			"step": self.step,
			"model": {
				k: v.cpu() if isinstance(v, torch.Tensor) else v
				for k, v in model_state.items()
			},
			"optimizer": {
				k: v.cpu() if isinstance(v, torch.Tensor) else v
				for k, v in self.optimizer.state_dict().items()
			},
			"model_params": self.model_kwargs,
			"scaler": self.scaler.state_dict(),
		}

	def load_state_dict(self, state_dict):
		if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
			self.model.module.load_state_dict(state_dict["model"])
		else:
			self.model.load_state_dict(state_dict["model"])
		self.optimizer.load_state_dict(state_dict["optimizer"])
		self.scaler.load_state_dict(state_dict["scaler"])
		self.step = state_dict["step"]

	def save_to_checkpoint(self, filename="weights"):
		save_basename = f"{filename}-{self.step}.pt"
		save_name = f"{self.model_dir}/{save_basename}"
		link_name = f"{self.model_dir}/{filename}.pt"
		torch.save(self.state_dict(), save_name)
		if os.name == "nt":
			torch.save(self.state_dict(), link_name)
		else:
			if os.path.islink(link_name):
				os.unlink(link_name)
			os.symlink(save_basename, link_name)

	def restore_from_checkpoint(self, filename="weights"):
		try:
			checkpoint = torch.load(f"{self.model_dir}/{filename}.pt")
			self.load_state_dict(checkpoint)
			return True
		except FileNotFoundError:
			return False

	def _write_summary(self, step, features, loss):
		# Log spectrogram image (convert tensor to numpy and ensure channel-last format)
		# Flip and take first sample, convert to numpy
		spectrogram = torch.flip(features["spectrogram"][:1], [1])
		# Remove batch dimension and convert to numpy
		spectrogram_np = spectrogram.squeeze(0).cpu().numpy()
		# If needed, add a channel dimension (H, W) -> (H, W, 1)
		if spectrogram_np.ndim == 2:
			spectrogram_np = np.expand_dims(spectrogram_np, axis=-1)
		wandb.log({"feature/spectrogram": wandb.Image(spectrogram_np)}, step=step)

		# Log scalars
		wandb.log({"train/loss": loss, "train/grad_norm": self.grad_norm}, step=step)

	def train(self, max_steps=None):
		device = next(self.model.parameters()).device
		while True:
			for features in tqdm(
				self.dataset, desc=f"Epoch {self.step // len(self.dataset)}"
			):  # 데이터셋에서 features를 가져옴
				if max_steps is not None and self.step >= max_steps:
					return
				features = nested_map(
					features,
					lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
				)  # 텐서면 디바이스에 알맞게 변경
				loss = self.train_step(features)  # 실제 train
				if torch.isnan(loss).any():
					raise RuntimeError(f"Detected NaN loss at step {self.step}.")
				if self.step % 50 == 0:
					self._write_summary(self.step, features, loss)
				if self.step % len(self.dataset) == 0:
					self.save_to_checkpoint()
				self.step += 1

	def train_step(self, features):
		for param in self.model.parameters():
			param.grad = None

		lyrics = features["lyrics"]
		spectrogram = features["spectrogram"]

		diff_loss, prior_loss = self.model(
			spectrogram, lyrics, step_idx=None
		)  # random step index inside the model
		loss = diff_loss + (self.prior_lambda * prior_loss)

		self.scaler.scale(loss).backward()
		self.scaler.unscale_(self.optimizer)
		self.grad_norm = nn.utils.clip_grad_norm_(
			self.model.parameters(), self.max_grad_norm or 1e9
		)
		self.scaler.step(self.optimizer)
		self.scaler.update()
		return loss
