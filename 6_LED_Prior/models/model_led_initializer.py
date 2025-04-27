import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder

class LEDInitializer(nn.Module):
    def __init__(self, t_h: int, t_f: int, d_f: int, k_pred: int):
        """
        Parameters
        ----
		t_h: number of past timesteps.
		t_f: number of future timesteps.
		d_f: feature dimension per future timestep. For 2D, d_f = 2; for 3D, d_f = 3; for TUM (SE(3) raw) d_f = 7.
		k_pred: number of candidate predictions.
		cfg: configuration object that contains, e.g., past_frames, future_frames,and possibly other hyperparameters.
                 
            
        The new input dimension per timestep (d_h_new) is determined as:
          - d_f in [2, 3]:     d_h_new = 2 * d_f   (relative position + velocity)
          - d_f == 7:          d_h_new = 7 + 3     (relative pose (7) + translation velocity (3))
        """
        super(LEDInitializer, self).__init__()
        self.n = k_pred
        
		# determine input dimension per timestep (d_h_new) based on cfg.dimensions / d_f.
        if d_f in [2, 3]:
            self.input_dim_per_timestep = 2 * d_f   # (relative, velocity)
        elif d_f in [6,7,9]:
            self.input_dim_per_timestep = d_f + 3     # 7 (relative pose) + 3 (translation velocity)
        else:
            raise NotImplementedError("LEDInitializer currently handles only dimensions 2, 3, or 7.")
        
        # total input dimension over T/ t_h timesteps.
        self.input_dim = t_h * self.input_dim_per_timestep
        
        # total output dimension = number of future timesteps * dimensions / d_f * k_pred
        self.output_dim = t_f * d_f * k_pred
        self.fut_len = t_f
        self.d_f = d_f  # as provided in config "dimensions"

        # social encoder and ego encoders
        self.social_encoder = social_transformer(t_h, self.input_dim_per_timestep)
        self.ego_var_encoder = st_encoder(self.input_dim_per_timestep)
        self.ego_mean_encoder = st_encoder(self.input_dim_per_timestep)
        self.ego_scale_encoder = st_encoder(self.input_dim_per_timestep)

        # simple MLP to encode the scale value
        self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

        # The decoders: output of the mean decoder should have shape (B, T, D) and the variance decoder should produce a prediction for each of the k candidates:
        self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
        self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
        self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())

    def forward(self, x, mask=None):
        """
        x: past_trajectory of shape (B, T / t_h, input_dim_per_timestep)
        	(B, t_h, 2*d_f) for d_f in [2,3] or 
        	(B, t_h, d_f+3) for d_f==7.
        
        Returns
        -------
		guess_var: variance output for k predictions, shape (B, k_pred, T, D)
		guess_mean: mean output, shape (B, T, D)
		guess_scale: scale (offset) embedding (e.g. for further processing)
        """
        
		# prepare mask
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        # social encoding - capture interactions over the past TODO remove
        social_embed = self.social_encoder(x, mask)  # Expected output shape: (B, 1, F)
        social_embed = social_embed.squeeze(1)         # (B, F)

        # separate encoders for different aspects
        ego_var_embed = self.ego_var_encoder(x)   # (B, F)
        ego_mean_embed = self.ego_mean_encoder(x) # (B, F)
        ego_scale_embed = self.ego_scale_encoder(x) # (B, F)

        # combine embeddings to estimate the mean
        mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
        guess_mean = self.mean_decoder(mean_total)
        guess_mean = guess_mean.contiguous().view(-1, self.fut_len, self.d_f)

        # combine embeddings to estimate the scale  (and eventually, variance)
        scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
        guess_scale = self.scale_decoder(scale_total)

        # process scale features, combine embeddings to estimate the variance and generate K normalised offsets
        guess_scale_feat = self.scale_encoder(guess_scale)
        var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
        guess_var = self.var_decoder(var_total)
        guess_var = guess_var.reshape(x.size(0), self.n, self.fut_len, self.d_f)

		# guess_mean  -> corresponds to μθ
		# guess_var   -> corresponds to the raw prediction that will be transformed into σθ
		# guess_scale -> serves as the normalized offsets S_{bθ,k} after further processing in training
          
        return guess_var, guess_mean, guess_scale
    


class LEDInitializerOld(nn.Module):
	def __init__(self, t_h: int, d_h: int, t_f: int, d_f: int, k_pred: int):
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



