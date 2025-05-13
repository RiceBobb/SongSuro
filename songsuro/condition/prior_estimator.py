from torch import nn


class PriorEstimator(nn.Module):
	def __init__(self, condition_embedding_dim: int, output_dim: int):
		"""
		PriorEstimator to predict the start latent from the condition

		:param condition_embedding_dim: The dimension of the condition embedding.
		:param output_dim: The output dimension.
			It must be the same with the latent dimension in the diffusion model.
		"""
		super().__init__()
		self.layer = nn.Linear(condition_embedding_dim, output_dim)

	def forward(self, x):
		# Input x will be (Batch, Embedding_dim)
		x = self.layer(x)  # Output x will be (Batch, output_dim)
		return x
