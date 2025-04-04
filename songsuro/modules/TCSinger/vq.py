import torch
from torch import nn
from torch.nn import functional as F
from scipy.cluster.vq import kmeans2
from einops import rearrange


class VQEmbeddingEMA(nn.Module):
	def __init__(
		self,
		n_embeddings,
		embedding_dim,
		commitment_cost=0.25,
		decay=0.999,
		epsilon=1e-5,
		print_vq_prob=False,
	):
		super(VQEmbeddingEMA, self).__init__()
		self.commitment_cost = commitment_cost
		self.n_embeddings = n_embeddings
		self.decay = decay
		self.epsilon = epsilon
		self.print_vq_prob = print_vq_prob
		self.register_buffer("data_initialized", torch.zeros(1))
		init_bound = 1 / 512
		embedding = torch.Tensor(n_embeddings, embedding_dim)
		embedding.uniform_(-init_bound, init_bound)
		self.register_buffer("embedding", embedding)
		self.register_buffer("ema_count", torch.zeros(n_embeddings))
		self.register_buffer("ema_weight", self.embedding.clone())

	def encode(self, x):
		B, T, _ = x.shape
		M, D = self.embedding.size()

		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(
			torch.sum(self.embedding**2, dim=1)
			+ torch.sum(x_flat**2, dim=1, keepdim=True),
			x_flat,
			self.embedding.t(),
			alpha=-2.0,
			beta=1.0,
		)  # [B*T_mel, N_vq]
		indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
		quantized = F.embedding(indices, self.embedding)
		quantized = quantized.view_as(x)
		return x_flat, quantized, indices

	def encode_indice(self, x):
		x = x.permute(0, 2, 1).contiguous()
		B, T, _ = x.shape
		M, D = self.embedding.size()

		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(
			torch.sum(self.embedding**2, dim=1)
			+ torch.sum(x_flat**2, dim=1, keepdim=True),
			x_flat,
			self.embedding.t(),
			alpha=-2.0,
			beta=1.0,
		)  # [B*T_mel, N_vq]
		indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
		indices = indices.reshape(B, T)
		return indices

	def decode(self, indices):
		quantized = F.embedding(indices, self.embedding)

		return quantized

	def forward(self, x):
		"""

		:param x: [B, T, D]
		:return: [B, T, D]
		"""
		x = x.permute(0, 2, 1).contiguous()
		B, T, _ = x.shape
		M, D = self.embedding.size()
		if self.training and self.data_initialized.item() == 0:
			print(
				"| running kmeans in VQVAE"
			)  # data driven initialization for the embeddings
			x_flat = x.detach().reshape(-1, D)
			rp = torch.randperm(x_flat.size(0))
			kd = kmeans2(
				x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit="points"
			)
			self.embedding.copy_(torch.from_numpy(kd[0]))
			x_flat, quantized, indices = self.encode(x)
			encodings = F.one_hot(indices, M).float()
			self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
			self.ema_count.copy_(torch.sum(encodings, dim=0))

		x_flat, quantized, indices = self.encode(x)
		encodings = F.one_hot(indices, M).float()
		indices = indices.reshape(B, T)

		if self.training and self.data_initialized.item() != 0:
			self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
				encodings, dim=0
			)

			n = torch.sum(self.ema_count)
			self.ema_count = (
				(self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
			)

			dw = torch.matmul(encodings.t(), x_flat)
			self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

			self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
		self.data_initialized.fill_(1)

		e_latent_loss = F.mse_loss(x, quantized.detach(), reduction="none")
		nonpadding = (x.abs().sum(-1) > 0).float()
		e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
		loss = self.commitment_cost * e_latent_loss

		quantized = x + (quantized - x).detach()

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		if self.print_vq_prob:
			print("| VQ code avg_probs: ", avg_probs)
		quantized_ = quantized.view_as(x)
		quantized = quantized_.permute(0, 2, 1).contiguous()

		return quantized, loss, indices, perplexity


class VectorQuantiser(nn.Module):
	"""
	Improved version over vector quantiser, with the dynamic initialisation
	for these unoptimised "dead" points.
	num_embed: number of codebook entry
	embed_dim: dimensionality of codebook entry
	beta: weight for the commitment loss
	distance: distance for looking up the closest code
	anchor: anchor sampled methods
	first_batch: if true, the offline version of our model
	contras_loss: if true, use the contras_loss to further improve the performance
	"""

	def __init__(
		self,
		num_embed,
		embed_dim,
		beta,
		distance="l2",
		anchor="probrandom",
		first_batch=False,
		contras_loss=False,
	):
		super().__init__()

		self.num_embed = num_embed
		self.embed_dim = embed_dim
		self.beta = beta
		self.distance = distance
		self.anchor = anchor
		self.first_batch = first_batch
		self.contras_loss = contras_loss
		self.decay = 0.99
		self.init = False

		self.pool = FeaturePool(self.num_embed, self.embed_dim)
		self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
		self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
		self.register_buffer("embed_prob", torch.zeros(self.num_embed))

	def encode_indice(self, z):
		z = rearrange(z, "b c h -> b h c").contiguous()
		B, H, _ = z.shape
		z_flattened = z.view(-1, self.embed_dim)

		# clculate the distance
		if self.distance == "l2":
			# l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
			d = (
				-torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
				- torch.sum(self.embedding.weight**2, dim=1)
				+ 2
				* torch.einsum(
					"bd, dn-> bn",
					z_flattened.detach(),
					rearrange(self.embedding.weight, "n d-> d n"),
				)
			)
		elif self.distance == "cos":
			# cosine distances from z to embeddings e_j
			normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
			normed_codebook = F.normalize(self.embedding.weight, dim=1)
			d = torch.einsum(
				"bd,dn->bn",
				normed_z_flattened,
				rearrange(normed_codebook, "n d -> d n"),
			)

		# encoding
		sort_distance, indices = d.sort(dim=1)
		# look up the closest point for the indices
		encoding_indices = indices[:, -1]
		encoding_indices = encoding_indices.reshape(B, H)

		return encoding_indices

	def decode(self, encoding_indices):
		encoding_indices = encoding_indices.reshape(-1)
		encodings = torch.zeros(
			encoding_indices.unsqueeze(1).shape[0],
			self.num_embed,
			device=encoding_indices.device,
		)
		encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

		# quantise and unflatten
		z_q = torch.matmul(encodings, self.embedding.weight)
		return z_q

	def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
		assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
		assert not rescale_logits, "Only for interface compatible with Gumbel"
		assert not return_logits, "Only for interface compatible with Gumbel"
		# reshape z -> (batch, height, width, channel) and flatten
		z = rearrange(z, "b c h -> b h c").contiguous()
		B, H, _ = z.shape
		z_flattened = z.view(-1, self.embed_dim)

		# clculate the distance
		if self.distance == "l2":
			# l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
			d = (
				-torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
				- torch.sum(self.embedding.weight**2, dim=1)
				+ 2
				* torch.einsum(
					"bd, dn-> bn",
					z_flattened.detach(),
					rearrange(self.embedding.weight, "n d-> d n"),
				)
			)
		elif self.distance == "cos":
			# cosine distances from z to embeddings e_j
			normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
			normed_codebook = F.normalize(self.embedding.weight, dim=1)
			d = torch.einsum(
				"bd,dn->bn",
				normed_z_flattened,
				rearrange(normed_codebook, "n d -> d n"),
			)

		# encoding
		sort_distance, indices = d.sort(dim=1)
		# look up the closest point for the indices
		encoding_indices = indices[:, -1]
		re_indices = encoding_indices.reshape(B, H)
		encodings = torch.zeros(
			encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device
		)
		encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

		# quantise and unflatten
		z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
		# compute loss for embedding
		loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
			(z_q - z.detach()) ** 2
		)
		# preserve gradients
		z_q = z + (z_q - z).detach()
		# reshape back to match original input shape
		z_q = rearrange(z_q, "b h c -> b c h").contiguous()
		# count
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		# online clustered reinitialisation for unoptimized points
		if self.training:
			# calculate the average usage of code entries
			self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
			# running average updates
			if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
				# closest sampling
				if self.anchor == "closest":
					sort_distance, indices = d.sort(dim=0)
					random_feat = z_flattened.detach()[indices[-1, :]]
				# feature pool based random sampling
				elif self.anchor == "random":
					random_feat = self.pool.query(z_flattened.detach())
				# probabilitical based random sampling
				elif self.anchor == "probrandom":
					norm_distance = F.softmax(d.t(), dim=1)
					prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
					random_feat = z_flattened.detach()[prob]
				# decay parameter based on the average usage
				decay = (
					torch.exp(
						-(self.embed_prob * self.num_embed * 10) / (1 - self.decay)
						- 1e-3
					)
					.unsqueeze(1)
					.repeat(1, self.embed_dim)
				)
				self.embedding.weight.data = (
					self.embedding.weight.data * (1 - decay) + random_feat * decay
				)
				if self.first_batch:
					self.init = True
			# contrastive loss
			if self.contras_loss:
				sort_distance, indices = d.sort(dim=0)
				dis_pos = sort_distance[
					-max(1, int(sort_distance.size(0) / self.num_embed)) :, :
				].mean(dim=0, keepdim=True)
				dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
				dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
				contra_loss = F.cross_entropy(
					dis,
					torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device),
				)
				loss += contra_loss

		return z_q, loss, re_indices, perplexity


class FeaturePool:
	"""
	This class implements a feature buffer that stores previously encoded features

	This buffer enables us to initialize the codebook using a history of generated features
	rather than the ones produced by the latest encoders
	"""

	def __init__(self, pool_size, dim=64):
		"""
		Initialize the FeaturePool class

		Parameters:
			pool_size(int) -- the size of featue buffer
		"""
		self.pool_size = pool_size
		if self.pool_size > 0:
			self.nums_features = 0
			self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

	def query(self, features):
		"""
		return features from the pool
		"""
		self.features = self.features.to(features.device)
		if self.nums_features < self.pool_size:
			if (
				features.size(0) > self.pool_size
			):  # if the batch size is large enough, directly update the whole codebook
				random_feat_id = torch.randint(
					0, features.size(0), (int(self.pool_size),)
				)
				self.features = features[random_feat_id]
				self.nums_features = self.pool_size
			else:
				# if the mini-batch is not large enough, just store it for the next update
				num = self.nums_features + features.size(0)
				self.features[self.nums_features : num] = features
				self.nums_features = num
		else:
			if features.size(0) > int(self.pool_size):
				random_feat_id = torch.randint(
					0, features.size(0), (int(self.pool_size),)
				)
				self.features = features[random_feat_id]
			else:
				random_id = torch.randperm(self.pool_size)
				self.features[random_id[: features.size(0)]] = features

		return self.features
