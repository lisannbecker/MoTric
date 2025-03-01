import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

from models.layers import PositionalEncoding, ConcatSquashLinear

class st_encoder(nn.Module):
	def __init__(self, d_h):
		super().__init__()
		channel_in = d_h
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x


class social_transformer(nn.Module):
	def __init__(self, t_h: int, d_h:int): # d_h os 6 for 2D, 9 for 3D
		super(social_transformer, self).__init__()
		#self.encode_past = nn.Linear(60, 256, bias=False)
		self.encode_past = nn.Linear(t_h*d_h, 256, bias=False) #LB need to update XXX
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat


class TransformerDenoisingModel(Module):

	def __init__(self, t_h: int, d_f:int, context_dim=256, tf_layer=2): #d_f = dimensions
		super().__init__()

		## Translation
		self.d_f = d_f #should be 6 for SE3
		self.d_h = d_f*3 #18
		self.encoder_context = social_transformer(t_h, self.d_h)
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24) #TODO set to more for kitti
		self.concat1 = ConcatSquashLinear(d_f, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		if d_f in [2, 3]:
			self.linear = ConcatSquashLinear(context_dim//2, d_f, context_dim+3)


		else:
			#self.linear = ConcatSquashLinear(context_dim//2, d_f, context_dim+3)
			# Two separate heads: one for translation (3D) and one for rotation (3D Lie algebra)
			self.linear_trans = ConcatSquashLinear(context_dim // 2, 3, context_dim + 3)
			self.linear_rot = ConcatSquashLinear(context_dim // 2, 3, context_dim + 3)


	def forward(self, x, beta, context, mask):
		batch_size = x.size(0)

		#prepare beta and context (time, agents) embedding
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		#time embedding: 3D embedding using beta, sin(beta) and cos(beta)
		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		# Fuse diffused input with context
		fused = self.concat1(ctx_emb, x) 	# (B, T, 2*context_dim)
		final_emb = fused.permute(1,0,2) 	# rearrange: (T, B, 2*context_dim)
		final_emb = self.pos_emb(final_emb)
		
		# Pass through transformer encoder
		if self.cfg.dimensions in [2,3]:
			trans = self.transformer_encoder(final_emb).permute(1,0,2)
			trans = self.concat3(ctx_emb, trans)
			trans = self.concat4(ctx_emb, trans)
			return self.linear(ctx_emb, trans)

		trans_emb = self.transformer_encoder(final_emb).permute(1,0,2)
		trans_emb = self.concat3(ctx_emb, trans_emb)
		trans_emb = self.concat4(ctx_emb, trans_emb)

		pred_trans = self.linear_trans(ctx_emb, trans_emb) # (B, T, 3) translation
		pred_rot = self.linear_rot(ctx_emb, trans_emb) # (B, T, 3) rotation (Lie algebra)
		pred = torch.cat([pred_trans, pred_rot], dim=-1) # concatenate along pose dimension (last) # (B, T, 6)
		return pred
		#return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask):
		"""
        Accelerated generation method for inference, following a similar pipeline.
        
        Args:
            x: Input tensor with shape (B, num_predictions, T, d_f)
            beta: Diffusion step parameter, shape (B, 1, 1)
            context: Context for conditioning.
            mask: Temporal mask.
        
        Returns:
            pred: Generated predictions of shape (B, num_predictions, T, 6)
        """
		
		batch_size, num_predictions, num_timesteps, dims = x.shape
		# print(x.size())

		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, num_predictions, 1).unsqueeze(2)  # (B, num_predictions, 1, 259)
		# print(ctx_emb.size())

		#Fuse input with context
		#Apply first concat layer
		#x_fused = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, num_timesteps, 512)  # (B*num_predictions, num_timesteps, 512)
		x_fused = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, num_timesteps, 2 * self.pos_emb.pe.size(-1) // 2)  # (B*num_predictions, num_timesteps, 512)

		final_emb = x_fused.permute(1, 0, 2) # Transformer expects (seq_len, batch, feature)
		final_emb = self.pos_emb(final_emb)
		
		# Pass Through Transformer Encoder
		trans_emb = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous() #(B*num_predictions, num_timesteps, 512)
		# print(f"Transformer output shape before reshaping: {trans.shape}")  # Debugging print

		# Reshape to Match Expected Output
		trans_emb = trans_emb.view(batch_size, num_predictions, num_timesteps, -1) 
		# print(trans.size())

		# Apply Final Processing Layers
		trans_emb = self.concat3.batch_generate(ctx_emb, trans_emb)
		trans_emb = self.concat4.batch_generate(ctx_emb, trans_emb)


		if dims in [2, 3]:
			# For 2D or 3D, only predict translation.
			pred = self.linear.batch_generate(ctx_emb, trans_emb)
		else:  # self.cfg.dimensions == 6
			pred_trans = self.linear_trans.batch_generate(ctx_emb, trans_emb)
			pred_rot = self.linear_rot.batch_generate(ctx_emb, trans_emb)
			pred = torch.cat([pred_trans, pred_rot], dim=-1)

		return pred
		#return self.linear.batch_generate(ctx_emb, trans)
