import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder

class LEDInitializer(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.
		'''
		super(LEDInitializer, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f
		self.d_f = d_f

		# print(t_h)

		self.social_encoder = social_transformer(t_h, d_h) #d_h features per timestep - 6 for 2D, 9 for 3D
		self.ego_var_encoder = st_encoder(d_h)
		self.ego_mean_encoder = st_encoder(d_h)
		self.ego_scale_encoder = st_encoder(d_h)

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())


	
	def forward(self, x, mask=None):
		'''
		x = past_trajectory: (B, past_size, 3*dims)	< 3*dims: absolute, relative, and velocity for all dimensions (incl rotations)
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

		# Social encoding: capture interactions over the past
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256
		
		# Separate encoders for different aspects
		ego_var_embed = self.ego_var_encoder(x) # For variance estimation.
		ego_mean_embed = self.ego_mean_encoder(x) # For mean estimation.
		ego_scale_embed = self.ego_scale_encoder(x) # For scale (used in sample prediction)
		# B, 256

		# Combine embeddings to estimate the mean
		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, self.d_f)

		# Combine embeddings to estimate the scale (and eventually, variance)
		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		# Combine embeddings to estimate the variance and generate K normalized offsets
		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, self.d_f)

		# guess_mean  -> corresponds to μθ
		# guess_var   -> corresponds to the raw prediction that will be transformed into σθ
		# guess_scale -> serves as the normalized offsets S_{bθ,k} after further processing in training

		return guess_var, guess_mean, guess_scale



