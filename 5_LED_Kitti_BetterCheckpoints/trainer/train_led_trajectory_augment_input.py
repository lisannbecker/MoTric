import sys
import os
import time
import torch
import torch.nn.functional as F

import random
import numpy as np
import torch.nn as nn
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from utils.config import Config
from utils.utils import print_log

from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #LoaderKitti is two levels up
from PoseLoaderCustom import LoadDatasetLeapfrog, seq_collate_custom


from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

import pdb
NUM_Tau = 5

class RegularizedKDE(gaussian_kde):
    def _compute_covariance(self):
        super()._compute_covariance()
        # Add a small value to the diagonal for numerical stability.
        self.covariance += np.eye(self.covariance.shape[0]) * 1e-6

class Trainer:
	def __init__(self, config): 
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)
		self.cfg.dataset = config.dataset #use kitti/oxford spires/newer college dataset if specified in command line

		print("\nConfiguration:")
		for key, value in self.cfg.yml_dict.items():
			print(f"{key}: {value}")
		print()
		
		# ------------------------- prepare train/test data loader -------------------------

		if self.cfg.dataset.lower() == 'nba':
			print("[INFO] NBA dataset (11 agent]s).")
			dataloader_class = NBADataset
			collate_fn = seq_collate

			train_dset = NBADataset(
				obs_len=self.cfg.past_frames,
				pred_len=self.cfg.future_frames,
				training=True
			)
			self.train_loader = DataLoader(
				train_dset,
				batch_size=self.cfg.train_batch_size,
				shuffle=True,
				num_workers=4,
				collate_fn=seq_collate,
				pin_memory=True
			)
			test_dset = NBADataset(
				obs_len=self.cfg.past_frames,
				pred_len=self.cfg.future_frames,
				training=False
			)
			self.test_loader = DataLoader(
				test_dset,
				batch_size=self.cfg.test_batch_size,
				shuffle=False,
				num_workers=4,
				collate_fn=seq_collate,
				pin_memory=True
			)

		elif self.cfg.dataset.lower() != 'nba':
			dataloader_class = LoadDatasetLeapfrog
			collate_fn = seq_collate_custom
			print(f"[INFO] {self.cfg.dataset.upper()} dataset (1 agent).")

			if config.train == 1:
				train_dset = dataloader_class(
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=True,
					final_eval=False,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories
				)
				self.train_loader = DataLoader(
					train_dset,
					batch_size=self.cfg.train_batch_size,
					shuffle=True,
					num_workers=4,
					collate_fn=collate_fn,
					pin_memory=True
				)

				test_dset = dataloader_class(
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=False,
					final_eval=False,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories
				)
				self.test_loader = DataLoader(
					test_dset,
					batch_size=self.cfg.test_batch_size,
					shuffle=False,
					num_workers=4,
					collate_fn=collate_fn,
					pin_memory=True
				)
				print('[INFO] Now using random trajectory shuffling.\n')
		
				### Stats about trajectories
				if self.cfg.dimensions == 2:
					print("Train dataset:")
					self.print_some_stats(train_dset.fut_motion_3D, None, 2)
					print("\nValidation dataset:")
					self.print_some_stats(test_dset.fut_motion_3D, None, 2)

				elif self.cfg.dimensions == 3:
					print("Train dataset:")
					self.print_some_stats(train_dset.fut_motion_3D, None, 3)
					print("\nValidation dataset:")
					self.print_some_stats(test_dset.fut_motion_3D, None, 3)
				
				elif self.cfg.dimensions == 6:
					print("Train dataset:")
					self.print_some_stats(train_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)
					print("\nValidation dataset:")
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)

				elif self.cfg.dimensions == 9:
					print("Train dataset:")
					self.print_some_stats(train_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)
					print("\nValidation dataset:")
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)
			
			elif config.train==0:
				test_dset = dataloader_class(
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=False,
					final_eval=True,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories
				)
				self.test_loader = DataLoader(
					test_dset,
					batch_size=self.cfg.test_batch_size,
					shuffle=False,
					num_workers=4,
					collate_fn=collate_fn,
					pin_memory=True
				)

				### Stats about trajectories
				if self.cfg.dimensions == 2:
					print("\nTest dataset (model evaluation):")
					self.print_some_stats(test_dset.fut_motion_3D, None, 2)

				elif self.cfg.dimensions == 3:
					print("\nTest dataset (model evaluation):")
					self.print_some_stats(test_dset.fut_motion_3D, None, 3)
				
				elif self.cfg.dimensions == 6:
					print("\nTest dataset (model evaluation):")
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)
				
				elif self.cfg.dimensions == 9:
					print("\nTest dataset (model evaluation):")
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)


				#TODO implement for 9D << rotation without last column + translation

			if self.cfg.future_frames < 20:
				print(f"[Warning] Only {self.cfg.future_frames} future timesteps available, "
					f"ADE/FDE will be computed for up to {self.cfg.future_frames // 5} seconds instead of the full 4 seconds.")

		
			# for batch in self.train_loader:
			# 	print(batch.keys())
			# 	print("Batch pre-motion shape:", batch['pre_motion_3D'].shape)  
			# 	print("Batch future motion shape:", batch['fut_motion_3D'].shape)  
			# 	print("Batch pre-motion mask shape:", batch['pre_motion_mask'].shape)  # [batch_size, 1, past_poses, 2]
			# 	print("Batch future motion mask shape:", batch['fut_motion_mask'].shape)  # [batch_size, 1, future_poses, 2]
			# 	print("traj_scale:", batch['traj_scale'])
			# 	print("pred_mask:", batch['pred_mask'])
			# 	print("seq:", batch['seq'], '\n')
			# 	break
			print(f'\n[INFO] {self.cfg.dataset.upper()} dataset - skip subtracting mean from absolute positions.')
			
		
		
		# data normalization parameters
		self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
		self.traj_scale = self.cfg.traj_scale

		# ------------------------- define diffusion parameters -------------------------
		self.n_steps = self.cfg.diffusion.steps # define total diffusion steps

		# make beta schedule and calculate the parameters used in denoising process.
		self.betas = self.make_beta_schedule(
			schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
			start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()
		
		self.alphas = 1 - self.betas
		self.alphas_prod = torch.cumprod(self.alphas, 0)
		self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
		self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)


		# ------------------------- define models -------------------------
		self.model = CoreDenoisingModel(
			t_h=self.cfg.past_frames,
			d_f=self.cfg.dimensions
		).cuda()

		if self.cfg.past_frames == 10 and self.cfg.future_frames == 20 and self.cfg.dataset == 'nba':
			# load pretrained models 
			print('[INFO] Loading pretrained models... (NBA with standard frame configs)\n')
			model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu') #LB expects 60 dimensional input (6 x 10 past poses)
			self.model.load_state_dict(model_cp['model_dict'])

			self.model_initializer = InitializationModel(
				t_h=self.cfg.past_frames, 
				d_h=self.cfg.dimensions*3, 
				t_f=self.cfg.future_frames, 
				d_f=self.cfg.dimensions, 
				k_pred=self.cfg.k_preds
			).cuda()
		
		else:
			print('[INFO] Training from scratch - without pretrained models (Not NBA with frame standard configs)\n')
			# print('Params for model_initialiser: ', self.cfg.past_frames, self.cfg.dimensions*3, self.cfg.future_frames, self.cfg.dimensions, self.cfg.k_preds)
			self.model_initializer = InitializationModel( #DIM update delete
				t_h=self.cfg.past_frames, 
				d_h=self.cfg.dimensions*3, 
				t_f=self.cfg.future_frames, 
				d_f=self.cfg.dimensions, 
				k_pred=self.cfg.k_preds
			).cuda()

		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		

		# ------------------------- prepare logs -------------------------
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.print_model_param(self.model, name='Core Denoising Model')
		self.print_model_param(self.model_initializer, name='Initialization Model')

		# print(self.model)
		#print(self.model_initializer)

		# temporal reweight in the loss, it is not necessary.
		#self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10
		self.temporal_reweight = torch.FloatTensor([self.cfg.future_frames - i for i in range(1, self.cfg.future_frames + 1)]).cuda().unsqueeze(0).unsqueeze(0) / 10

	def save_checkpoint(self, epoch):
		"""
        Save a checkpoint containing both core denoising model (original) and
        the initialization model, along with optimizer and scheduler states
        """
		checkpoint = {
			'epoch': epoch,
			'cfg': self.cfg.yml_dict,
			'model_initializer_state_dict': self.model_initializer.state_dict(),
			'core_denoising_model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.opt.state_dict(),
			'scheduler_state_dict': self.scheduler_model.state_dict(),
		}
		checkpoint_dir = f'./{self.cfg.model_dir}'
		os.makedirs(checkpoint_dir, exist_ok=True)
		ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
		torch.save(checkpoint, ckpt_path)
		print_log(f"[INFO] Checkpoint saved to {ckpt_path}", self.log)

	def load_checkpoint(self, checkpoint_path):
		"""
		Load a checkpoint and restore model, optimizer, and scheduler states.
		"""
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		self.model_initializer.load_state_dict(checkpoint['model_initializer_state_dict'])
		self.model.load_state_dict(checkpoint['core_denoising_model_state_dict'])
		self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler_model.load_state_dict(checkpoint['scheduler_state_dict'])
		print_log(f"[INFO] Checkpoint loaded from {checkpoint_path}", self.log)

	def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
		'''
		Count the trainable/total parameters in `model`.
		'''
		total_num = sum(p.numel() for p in model.parameters())
		trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print_log("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num), self.log)
		return None

	def make_beta_schedule(self, schedule: str = 'linear', 
			n_timesteps: int = 1000, 
			start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
		'''
		Make beta schedule.

		Parameters
		----
		schedule: str, in ['linear', 'quad', 'sigmoid'],
		n_timesteps: int, diffusion steps,
		start: float, beta start, `start<end`,
		end: float, beta end,

		Returns
		----
		betas: Tensor with the shape of (n_timesteps)

		'''
		if schedule == 'linear':
			betas = torch.linspace(start, end, n_timesteps)
		elif schedule == "quad":
			betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
		elif schedule == "sigmoid":
			betas = torch.linspace(-6, 6, n_timesteps)
			betas = torch.sigmoid(betas) * (end - start) + start
		return betas


	### Train denoising network / noise estimation
	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

	def noise_estimation_loss(self, x, y_0, mask):
		"""
		Estimate how much noise has been added during the forward diffusion
		"""
		batch_size = x.shape[0]
		# Select a random step for each example
		t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
		t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
		# x0 multiplier
		a = self.extract(self.alphas_bar_sqrt, t, y_0)
		beta = self.extract(self.betas, t, y_0)
		# eps multiplier
		am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
		e = torch.randn_like(y_0)
		# model input
		y = y_0 * a + e * am1
		output = self.model(y, beta, x, mask)
		# batch_size, 20, 2
		return (e - output).square().mean()


	### Reverse diffusion process
	def p_sample(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)
	
	def p_sample_accelerate(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)

	def p_sample_loop(self, x, mask, shape):
		self.model.eval()
		prediction_total = torch.Tensor().cuda()
		for _ in range(self.cfg.future_frames):
			cur_y = torch.randn(shape).to(x.device)
			for i in reversed(range(self.n_steps)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total
	
	def p_sample_loop_mean(self, x, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate_old(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [Batch size, past steps, feature dimension per timestep = 6 (absolute position, relative position, velocity - all 2D)]
		mask: [Batch size, batch size]
		loc: [Batch size, number of predictions per timestep k_preds = alternative futures, timesteps into the future, dimensionality - x and y]
		cur_y: [11, 10, 20, 2]
		'''
		# print(f"Past Trajectory Shape (x): {x.size()}")  
		# print(f"Trajectory Mask Shape: {mask.size()}")  
		# print(f"Generated Location Shape (loc): {loc.size()}")  

		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
			
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		# shape: B=b*n, K=10, T, 2
		prediction_total = torch.cat((cur_y, cur_y_), dim=1)

		return prediction_total

	def p_sample_loop_accelerate(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [Batch size, past steps, feature dimension per timestep = 6 (absolute position, relative position, velocity - all 2D)]
		mask: [Batch size, batch size]
		loc: [Batch size, number of predictions per timestep k_preds = alternative futures, timesteps into the future, dimensionality - x and y]
		cur_y: [11, 10, 20, 2]
		'''
		# print(f"Past Trajectory Shape (x): {x.size()}")  
		# print(f"Trajectory Mask Shape: {mask.size()}")  
		# print(f"Generated Location Shape (loc): {loc.size()}")  
		cur_y = loc  # use all k predictions
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		return cur_y  # shape: (B, k_pred, T, d)


	def fit(self):
		# Training loop		
		for epoch in range(0, self.cfg.num_epochs):
			loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)

			if self.cfg.dimensions in [2,3]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch, loss_total, loss_distance, loss_uncertainty), self.log)
			
			elif self.cfg.dimensions in [6,9]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Rotation.: {:.6f}\tCombined Loss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch, loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty), self.log)


			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples= self._test_single_epoch() #average_euclidean = average total distance start to finish - to contextualise how good the FDE and ADE are
				for i, time_i in enumerate(range(5,21,5)):
					print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
						time_i, performance['ADE'][i]/samples,
						time_i, performance['FDE'][i]/samples), self.log)
					

				# cp_path = self.cfg.model_path % (epoch + 1)
				# model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
				# torch.save(model_cp, cp_path)

				self.save_checkpoint(epoch + 1)

			self.scheduler_model.step()


	def data_preprocess(self, data): #Updated to handle any number of agents
		"""
			pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
			fut_motion_3D: torch.Size([32, 11, 20, 2])
			fut_motion_mask: torch.Size([32, 11, 20])
			pre_motion_mask: torch.Size([32, 11, 10])
			traj_scale: 1
			pred_mask: None
			seq: nba
		"""
		# data['pre_motion_3D'][..., :3]
		batch_size = data['pre_motion_3D'].shape[0]
		num_agents = data['pre_motion_3D'].shape[1]

		#Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
		traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
		for i in range(batch_size):
			traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.

		# print('traj_mask: ', traj_mask.size())
		# Get last observed pose (for each agent) as initial position << both translation and rotation
		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:] # 2D: [B, num_agents, 1, 2] or 3D: [B, num_agents, 1, 3] or 6D: [B, num_agents, 1, 6]

		# augment input: absolute position, relative position, velocity TODO augmentation is applied to whole input, not only translation
		if self.cfg.dataset != 'nba':
			past_traj_abs = (data['pre_motion_3D'].cuda() / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #single agent: effectively only (B, 1, Past, Dims) > (B*1, Past, Dims) and scaling
		elif self.cfg.dataset == 'nba':
			past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) 
		
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #only relativises if initial pos is not 0 already (relative = True)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1) #(B, 1, Dim)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.future_frames, self.cfg.dimensions) #relativises (if not done already) and (B, 1, Past, Dims) > (B*1, Past, Dims)

		return batch_size, traj_mask, past_traj, fut_traj



	### Rotation conversion helpers
	def skew_symmetric(self,w):
		w0,w1,w2 = w.unbind(dim=-1)
		O = torch.zeros_like(w0)
		wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
							torch.stack([w2,O,-w0],dim=-1),
							torch.stack([-w1,w0,O],dim=-1)],dim=-2)
		return wx

	def taylor_A(self,x,nth=10):
		# Taylor expansion of sin(x)/x
		ans = torch.zeros_like(x)
		denom = 1.
		for i in range(nth+1):
			if i>0: denom *= (2*i)*(2*i+1)
			ans = ans+(-1)**i*x**(2*i)/denom
		return ans
	
	def taylor_B(self,x,nth=10):
		# Taylor expansion of (1-cos(x))/x**2
		ans = torch.zeros_like(x)
		denom = 1.
		for i in range(nth+1):
			denom *= (2*i+1)*(2*i+2)
			ans = ans+(-1)**i*x**(2*i)/denom
		return ans

	def so3_to_SO3(self,w): # [...,3] added from Ma PoseNet paper
		wx = self.skew_symmetric(w)
		theta = w.norm(dim=-1)[...,None,None]
		I = torch.eye(3,device=w.device,dtype=torch.float32)
		A = self.taylor_A(theta)
		B = self.taylor_B(theta)
		R = I+A*wx+B*wx@wx
		return R

	def rot6d_to_rotmat_SO3(self, x):
		"""
		Convert a 6D rotation representation to a 3x3 rotation matrix.
		x: tensor of shape (N, 6)
		Returns a tensor of shape (N, 3, 3)
		"""
		# Split into two 3D vectors
		a1 = x[:, :3]
		a2 = x[:, 3:]
		# Normalize the first vector to get v1
		v1 = a1 / torch.norm(a1, dim=1, keepdim=True)
		# Make a2 orthogonal to v1
		a2_proj = (torch.sum(v1 * a2, dim=1, keepdim=True)) * v1
		v2 = a2 - a2_proj
		v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
		# Compute v3 as cross product of v1 and v2
		v3 = torch.cross(v1, v2, dim=1)
		# Stack into rotation matrix
		R = torch.stack([v1, v2, v3], dim=-1)
		return R




	def print_some_stats(self, future, future_rot=None, translation_dims=3):
		print('Length:', future.size(0))
		future = future.squeeze(dim=1) #torch.Size([16106, 20, 3])

		distance_per_step = future[:, 1:, :] - future[:, :-1, :]
		abs_distance_per_step = torch.abs(distance_per_step)
		total_distance_per_sample = abs_distance_per_step.sum(dim=1) #sum over time steps


		mean_distance_x = total_distance_per_sample[:, 0].mean().item()
		mean_distance_y = total_distance_per_sample[:, 1].mean().item()
		if translation_dims == 3:
			mean_distance_z = total_distance_per_sample[:, 2].mean().item()
		

		step_euclidean = distance_per_step.norm(dim=2)
		total_euclidean_distance = step_euclidean.sum(dim=1)
		mean_euclidean_distance = total_euclidean_distance.mean().item()

		if translation_dims == 2:
			print(f"Total x and y distances travelled: {mean_distance_x:.5f}, {mean_distance_y:.5f}")
		elif translation_dims == 3:
			print(f"Total x, y and z distances travelled: {mean_distance_x:.5f}, {mean_distance_y:.5f}, {mean_distance_z:.5f}")

		print(f"Euclidean dist diff avg: {mean_euclidean_distance:.5f}")

		if future_rot is not None:
			print('Still need to implement rotation statistics')

	def visualise_single_KDE_GT_Past(self, k_preds_at_t, t_kde, all_past, GT_at_t, idx): #TODO make size dynamic
		if hasattr(k_preds_at_t, "cpu"):
			k_preds_at_t = k_preds_at_t.cpu().numpy()
		if hasattr(all_past, "cpu"):
			all_past = all_past.cpu().numpy()
		if hasattr(GT_at_t, "cpu"):
			GT_at_t = GT_at_t.cpu().numpy()
		# k_preds_at_t = k_preds_at_t.cpu().numpy()
		# all_past = all_past.cpu().numpy()
		# GT_at_t = GT_at_t.cpu().numpy()
		
		# Ensure GT_at_t is 2D
		if GT_at_t.ndim == 1:
			GT_at_t = GT_at_t.reshape(1, -1)  # Now shape becomes (1, 2)
		
		# print(k_preds_at_t.size())
		# print(all_past.size())
		# print(GT_at_t.size())

		# Combine all points to compute dynamic grid limits
		all_points = np.concatenate([k_preds_at_t, all_past, GT_at_t], axis=0)
		
		# Compute grid limits with a margin (e.g., 10 units or 10% of the range)
		margin = 1
		min_x, max_x = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
		min_y, max_y = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
		
		# Create a dynamic grid
		grid_res = 100j  # resolution: 100 points in each axis
		x_grid, y_grid = np.mgrid[min_x:max_x:grid_res, min_y:max_y:grid_res]
		grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
		kde_values = t_kde(grid_points).reshape(x_grid.shape)
		
		# Create the plot
		plt.figure(figsize=(10, 8), dpi=300)
		plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
		plt.scatter(k_preds_at_t[:, 0], k_preds_at_t[:, 1], s=3, alpha=1, color="blue", label="Predicted Samples")
		plt.scatter(all_past[:, 0], all_past[:, 1], s=20, color="cyan", alpha=0.8, label="Past Poses")
		plt.scatter(GT_at_t[:, 0], GT_at_t[:, 1], s=50, color="red", marker="*", label="GT Pose")
    
		
		plt.xlabel("X Position")
		plt.ylabel("Y Position")  
		plt.colorbar(label="Density")
		plt.legend()
		#plt.show()
		vis_path = f'./visualization/Sanity_{idx}.jpg'
		plt.savefig(vis_path)
		print(f'[INFO] Visualisation saved to {vis_path}')
		plt.close()

	def compute_kde_and_vis_full_traj(self, k_preds, past_traj, fut_traj): #TODO how is KDE computed here
		"""
		Computes a KDE-based motion prior for each sample (batch) using all predictions across time,
		and visualizes the density along with sample points, past trajectories, and GT future poses.

		Args:
			k_preds (Tensor): Predicted trajectories of shape (B, K, T, Dim).
			past_traj (Tensor): Past trajectories of shape (B, TPast, 2).
			fut_traj (Tensor): Ground truth future trajectories of shape (B, T, 2).
		"""
		# Convert tensors to NumPy arrays
		k_preds_np = k_preds.detach().cpu().numpy()
		past_np = past_traj.detach().cpu().numpy()
		fut_np = fut_traj.detach().cpu().numpy()

		B, K, T, D = k_preds_np.shape

		for b in range(B):
			# Reshape predicted trajectories to combine all time steps (K * T, D)
			all_samples = k_preds_np[b].reshape(K * T, D)
			# Optionally, filter out any outlier samples if needed.
			all_samples = self.filter_k_preds_single_pose(all_samples)
			# Fit the KDE using the (filtered) sample points.
			kde = gaussian_kde(all_samples.T)

			# Combine all points (samples, past, and GT future) for dynamic grid limits.
			points_for_grid = np.concatenate([all_samples, past_np[b], fut_np[b]], axis=0)
			margin = 1  # Adjust this margin if necessary.
			min_x = points_for_grid[:, 0].min() - margin
			max_x = points_for_grid[:, 0].max() + margin
			min_y = points_for_grid[:, 1].min() - margin
			max_y = points_for_grid[:, 1].max() + margin

			# Create a grid dynamically based on computed boundaries.
			x_grid, y_grid = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
			grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
			kde_values = kde(grid_points).reshape(100, 100)

			# Begin plotting
			plt.figure(figsize=(10, 8))
			plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
			# Plot the KDE sample points (e.g., small dots)
			plt.scatter(all_samples[:, 0], all_samples[:, 1], s=3, alpha=1, label="Samples")
			# Plot the past trajectory with light blue dots
			plt.scatter(past_np[b][:, 0], past_np[b][:, 1], color='lightblue', s=30, label="Past Trajectory")
			# Plot the GT future poses with red stars
			plt.scatter(fut_np[b][:, 0], fut_np[b][:, 1], color='red', marker='*', s=100, label="GT Future Poses")
			
			plt.xlabel("X Position")
			plt.ylabel("Y Position")
			plt.colorbar(label="Density")
			plt.legend()
			plt.title(f"KDE Visualization for Batch Sample {b}")
			plt.savefig(f'./visualization/Trajectory{b}_AllTimesteps_dynamic.jpg')
			plt.close()
	
	def compute_batch_motion_priors_kde_temporal_independence(self, k_preds):
		"""
		Computes a KDE-based motion prior for each sample and each future time step.

		Limitation: Using per-timestep KDE densities assumes that the predictions at each timestep are independent, but they aren’t.
		
		Args:
			k_preds_np (np.array): Predicted trajectories of shape (B, K, T, Dimension).
								B: batch size, K: number of predictions, 
								T: future timesteps, 6: pose dimensions.
		
		Returns:
			priors: A list of lists such that priors[b][t] is a KDE object for sample b at time step t.
		"""
		k_preds_np = k_preds.detach().cpu().numpy()

		B, K, T, D = k_preds_np.shape

		priors = {}
		for trajectory_idx in range(B):
			# Option A: all Ks, all future poses
			#all_samples = k_preds_np[b].reshape(K * T, D)  # Merge all timesteps into one
			#kde = gaussian_kde(all_samples.T) # Fit KDE using all K*T samples
			#priors[b] = kde

			
			pose_priors = [] #one pose prior per timestep
			for time_idx in range(T): # All k-preds for a specific timestep
				k_poses = k_preds_np[trajectory_idx, :, time_idx, :]  # shape: (K, Dimension)
				
				# Fit a KDE for these 2D/3D/6D samples.
				kde = gaussian_kde(k_poses.T)  	# gaussian_kde expects shape (D, N), 
												# Kernel density estimation places a smooth "kernel" (Gaussian) at each sample point and sums them to create an overall density estimate
												# Parameter: bandwidth = how smooothly the points are summed. Eg affects whether two close modes merge into one or not
				
				pose_priors.append(kde)
			priors[trajectory_idx] = pose_priors

		return priors

	def filter_k_preds_single_pose(self, single_pose_all_ks):
		med = np.median(single_pose_all_ks, axis=0)
		#Euclidean distances from the median
		distances = np.linalg.norm(single_pose_all_ks - med, axis=1)
		mad = np.median(np.abs(distances - np.median(distances)))

		# threshold: remove samples that are more than 5 MAD away from the median distance
		threshold = np.median(distances) + 5 * mad
		
		# Filter out outliers (if only one crazy outlier exists, this will remove it)
		filtered_k_poses = single_pose_all_ks[distances <= threshold]
		return filtered_k_poses

	def KDE_single_pose_outlier_filtered(self, single_pose_all_ks):

		filtered_k_poses = self.filter_k_preds_single_pose(single_pose_all_ks)
		single_pose_KDE = gaussian_kde(filtered_k_poses.T)
		return single_pose_KDE

	def evaluate_pose_prior(self, pose, kde): #not used yet
		"""
		Evaluate the probability density for a given 6D pose under the provided KDE.
		
		Args:
			pose (np.array): 6D pose, shape (6,).
			kde: A gaussian_kde object.
		
		Returns:
			density (float): Estimated probability density at the pose.
		"""
		# gaussian_kde expects input shape (D, N); here N=1.
		dims = pose.shape[0]
		density = kde(pose.reshape(dims, 1))[0]
		return density

	def GT_KDE_density_histograms(self, all_densities_by_time, out_dir):
		for t in range(self.cfg.future_frames):
			plt.figure(figsize=(8, 6))
			plt.hist(all_densities_by_time[t], bins=30, edgecolor='black')
			plt.xlabel("KDE Density")
			plt.ylabel("Frequency")
			plt.title(f"KDE Density Histogram for Future Timestep {t}")
			vis_path = os.path.join(out_dir, f'KDE_Density_Time_{t}.jpg')
			plt.savefig(vis_path)
			plt.close()
			print(f"[INFO] Saved histogram for future timestep {t} at {vis_path}")


	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, loss_trans, loss_rot, count = 0, 0, 0, 0, 0,0
		#LB 3D addition to reshape tensors 
		
		for data in self.train_loader:
			# print("data['fut_motion_3D'].shape: ", data['fut_motion_3D'].shape)
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data) # past_traj =(past_traj_abs, past_traj_rel, past_traj_vel)
			# print('fut_traj:', fut_traj[0,0,:]) #first fut timestep

			# first_traj_mask = traj_mask.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_traj_mask.npy', first_traj_mask)
			# first_past_traj = past_traj.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_past_traj.npy', first_past_traj)			
			# first_fut_traj= fut_traj.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_fut_traj.npy', first_fut_traj)
			# print('traj_mask:', traj_mask.size()) # [32, 32]
			# print('past_traj:', past_traj.size()) # [32, 15, 9] 
			# print('fut_traj:', fut_traj.size()) # [32, 8, 3] < GT poses for future_frames timesteps


			### 1. Leapfrogging Denoising (LED initializer outputs): instead of full denoising process, predicts intermediate, already denoised trajectory 
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) #offset, mean, stdev
			#uses the past trajectory (and possibly social context) to produce a mean and variance for the future trajectory - sampled from to get candidate future trajectories next
			#sample prediction: provides the normalized offsets Sbθ,k that, when scaled and added to the mean, yield the final candidate trajectories

			# print("sample_prediction shape:", sample_prediction.shape)
			# print("mean_estimation shape:", mean_estimation.shape)
			# print("variance_estimation shape:", variance_estimation.shape)
		

			#TODO could clip variance to constrain k_preds more
			# variance_scale = torch.exp(variance_estimation/2)
			# variance_scale = torch.clamp(variance_scale, max=threshold_value)  # Set an appropriate threshold
			# sample_prediction = variance_scale[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None] 

			# Reparameterisation with uncertainty (original)
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			# Add the mean estimation to the scaled normalized offsets / center predictions around the mean (each candidate trajectory is expressed as a deviation from the mean)
			loc = sample_prediction + mean_estimation[:, None]

			# print('sample_prediction:', sample_prediction.size())
			# print('loc:', loc.size()) #[32, 24, 24, 3]

			### 2. Denoising (Denoising Module): Generate K alternative future trajectories - multi-modal
			k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) #(B, K, T, 2/3/6/9)
			
			#print('k_alternative_preds:', k_alternative_preds.size()) #check if prediction is 9D

			# self.old_compute_kde_and_vis_full_traj(k_alternative_preds)
			# exit()

			### 3D/2D code
			if self.cfg.dimensions in [2,3]:
				#squared distances / Euclidian, equal weight for all timesteps
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)
				loss_distance = ((k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1) * 
								self.temporal_reweight
								).mean(dim=-1).min(dim=1)[0].mean()
				loss_uncertainty = (torch.exp(-variance_estimation)*
									(k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
									variance_estimation
									).mean()
			
			elif self.cfg.dimensions == 6:
				# For loss, unsqueeze the ground truth to have predictions dimension
				"""6D specific code"""
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)

				# Split into translation and rot
				pred_trans = k_alternative_preds[..., :3]    # (B, K, T, 3)
				pred_rot_lie = k_alternative_preds[..., 3:]    # 6D: (B, K, T, 3) or 9D: (B, 1, T, 6)
				gt_trans = fut_traj_wpreds[..., :3]      # (B, 1, T, 3) 
				gt_rot_lie = fut_traj_wpreds[..., 3:]      # 6D: (B, 1, T, 3) or 9D: (B, 1, T, 6)

				### (1) TRANSLATION LOSS
				# L2 Euclidian distance - squared distances, equal weight for timesteps
				trans_diff = pred_trans-gt_trans # (B, K, T, 3)
				trans_error = trans_diff.norm(p=2, dim=-1) # (B, K, T)

				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1)	# (B, K) loss for all k preds
				# print(loss_translation)


				### (2) ROTATION LOSS (Geodesic)
				# convert Lie algebra rotations to classic 3x3 rotation matrices - need to flatten and unflatten into rot matrix
				B, K, T, _ = pred_rot_lie.shape
				pred_rot_flat = pred_rot_lie.view(-1, 3)          # (B*K*T, 3)
				pred_R = self.so3_to_SO3(pred_rot_flat)         # (B*K*T, 3, 3)
				pred_R = pred_R.view(B, K, T, 3, 3)

				# Same for ground truth
				gt_rot_lie_expanded = gt_rot_lie.expand(B, K, T, 3)  # (B, K, T, 3)
				gt_rot_flat = gt_rot_lie_expanded.contiguous().view(-1, 3)  # (B*K*T, 3)
				gt_R = self.so3_to_SO3(gt_rot_flat)              # (B*K*T, 3, 3)
				gt_R = gt_R.view(B, K, T, 3, 3)


			elif self.cfg.dimensions == 9:
				#generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) 

				# print('Predictions batch:', generated_y.size())
				# print('GT batch:', fut_traj.size())

				# print('Prediction 0 shape:', generated_y[0].size())
				# print('GT 0 shape:', fut_traj[0].size())

				# print('Prediction:', generated_y[0])
				# print('GT:', fut_traj[0])

				# For loss, unsqueeze the ground truth to have predictions dimension
				"""9D specific code"""
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)

				# Split into translation and rot
				pred_trans = k_alternative_preds[..., :3]    # (B, K, T, 3)
				pred_rot_6D = k_alternative_preds[..., 3:]    # (B, 1, T, 6)
				gt_trans = fut_traj_wpreds[..., :3]      # (B, 1, T, 3) 
				gt_rot_6D = fut_traj_wpreds[..., 3:]      # (B, 1, T, 6)

				### (1) TRANSLATION LOSS
				# L2 Euclidian distance - squared distances, equal weight for timesteps
				trans_diff = pred_trans-gt_trans # (B, K, T, 3)
				trans_error = trans_diff.norm(p=2, dim=-1) # (B, K, T)

				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1)	# (B, K) loss for all k preds


				### (2) ROTATION LOSS (Geodesic)
				# convert 6D rotations to classic 3x3 rotation matrices: reconstruct column 3 by taking cross product of first two columns
				B, K, T, _ = pred_rot_6D.shape
				# Flatten for conversion
				pred_rot_flat = pred_rot_6D.view(-1, 6)
				gt_rot_flat = gt_rot_6D.expand(B, K, T, 6).contiguous().view(-1, 6) #need to expand to match K dimension


				# Convert to rotation matrices
				pred_R = self.rot6d_to_rotmat_SO3(pred_rot_flat)  # shape: (B*K*T, 3, 3)
				gt_R = self.rot6d_to_rotmat_SO3(gt_rot_flat)      # shape: (B*K*T, 3, 3)

				# Reshape back
				pred_R = pred_R.view(B, K, T, 3, 3)
				gt_R = gt_R.view(B, K, T, 3, 3) # << now we have SO3
				# print(pred_R[0,0,0,...])
				# print(gt_R[0,0,0,...])

			if self.cfg.dimensions in [6,9]:
				# Compute relative rotation: R_diff = R_pred^T * R_gt.
				R_diff = torch.matmul(pred_R.transpose(-2, -1), gt_R)  # (B, K, T, 3, 3)

				#get rotation error angle theta of rot 3x3 rot matrix
				trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # (B, K, T) trace of each individual relative rotation matrix
				# Clamp to avoid numerical issues
				angular_error_theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))  # (B, K, T) take arccosine to get error angle cos(theta) = trace(R)-1 / 2 with clamping -1 + 1e-6, 1 - 1e-6
				loss_rotation = angular_error_theta.mean(dim=-1)  # average over time, so one loss per candidate K, shape (B, K)

				### (1+2) COMBINED DISTANCE LOSS (ROT AND TRANS)
				combined_error = loss_translation + loss_rotation  # (B, K) add translation and rotation error TODO normalise rot and trans loss
				loss_distance = combined_error.min(dim=1)[0].mean()  # some scalar << for whole batch, choose k_pred with lowest error, then average the error into one distance loss scalar

			"""General 2D/3D/6D/9D code continues here"""
			### (3) UNCERTAINTY LOSS (original)
			loss_uncertainty = (
				torch.exp(-variance_estimation) *
				(k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
				variance_estimation
			).mean()
		
			# print(loss_uncertainty)
			
			
			### TOTAL LOSS
			loss = loss_distance * 50 + loss_uncertainty #make distance loss more important than uncertainty loss (?) TODO maybe not this much
			# print(loss)
			# exit()
			loss_total += loss.item()
			loss_dt += loss_distance.item()*50
			loss_dc += loss_uncertainty.item()

			if self.cfg.dimensions in [6,9]:
				loss_trans += loss_translation.min(dim=1)[0].mean().item()
				loss_rot += loss_rotation.min(dim=1)[0].mean().item()

			self.opt.zero_grad()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()

			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total/count, loss_trans/count, loss_rot/count, loss_dt/count, loss_dc/count

	def _test_single_epoch(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)

		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				# LED initializer outputs
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
           		# Generate candidate future trajectories (B, K, T, 6)
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				#fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1) #expand GT to match k_preds dimension (B, 1, T, 6)
				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1) # (B, K, T, D) (?)


				# ---- Translation Error Metrics (already in your code) ----
				pred_traj_trans = pred_traj[..., :3]  # (B, K, T, 3)
				fut_traj_trans = fut_traj[..., :3]      # (B, 1, T, 3)

				distances = torch.norm(fut_traj_trans - pred_traj_trans, dim=-1) * self.traj_scale ## Euclidian translation errors (B, K, T)
				# print('distances: ', distances)


            	# Compute ADE and FDE at different timesteps.
				# Here we compute ADE and FDE for time steps: 5, 10, 15, and 20.
				for time_i in range(1, 5):
					max_index = min(5 * time_i - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps
					"""
					1s: 5 * 1 - 1 = 4 → Requires at least 5 timesteps.
					2s: 5 * 2 - 1 = 9 → Requires at least 10 timesteps.
					3s: 5 * 3 - 1 = 14 → Requires at least 15 timesteps.
					4s: 5 * 4 - 1 = 19 → Requires at least 20 timesteps.
					"""
					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
			
		return performance, samples

	def save_data(self):
		'''
		Save the visualization data.
		'''
		model_path = './results/checkpoints/led_vis.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		root_path = './visualization/data/'
				
		with torch.no_grad():
			for data in self.test_loader:
				_, traj_mask, past_traj, _ = self.data_preprocess(data)

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				torch.save(sample_prediction, root_path+'p_var.pt')
				torch.save(mean_estimation, root_path+'p_mean.pt')
				torch.save(variance_estimation, root_path+'p_sigma.pt')

				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				pred_mean = self.p_sample_loop_mean(past_traj, traj_mask, mean_estimation)

				torch.save(data['pre_motion_3D'], root_path+'past.pt')
				torch.save(data['fut_motion_3D'], root_path+'future.pt')
				torch.save(pred_traj, root_path+'prediction.pt')
				torch.save(pred_mean, root_path+'p_mean_denoise.pt')

				raise ValueError





	def test_single_model(self, checkpoint_path = None):
		# checkpoint_path = './results/5_Experiments/checkpoint_rework/models/checkpoint_epoch_40.pth'
		#checkpoint_path = './results/5_1_Overfitting_VisCheck/5_1_Overfitting_VisCheck/models/checkpoint_epoch_63.pth'
		checkpoint_path = './results/5_1_Overfitting_VisCheck/outlier_exclusion_check/models/checkpoint_epoch_17.pth'

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples, count = 0, 0

		# Ensure reproducibility for testing
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(42)

		hist_out_dir = './visualization/Temporal_Independence_KDE/Overfitting' #to save KDE densities on GT
		all_densities_by_time = {t: [] for t in range(self.cfg.future_frames)}

		# past_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_past_traj.npy')).to(self.device)
		# traj_mask = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_traj_mask.npy')).to(self.device)
		# fut_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_fut_traj.npy')).to(self.device)

		# sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
		# sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
		# loc = sample_prediction + mean_estimation[:, None]
		
		# k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)


		### Regular code
		with torch.no_grad():
			for data in self.test_loader:
				#print(data['fut_motion_3D']) #not scaled doen
				# print(data['fut_motion_3D'][5,:,:])
				# exit()
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				# print(past_traj.size())
				# print(past_traj[0, :, :])


				# Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				

				# print(k_alternative_preds.size())  # (B, K, T, 2)
				# print(fut_traj.size()) # (B, T, 2)
				# print(past_traj.size()) # (32, 10, 2)

				
				
				# ### ============= motion prior + visualisation =============
				# self.compute_kde_and_vis_full_traj(k_alternative_preds)

				priors = self.compute_batch_motion_priors_kde_temporal_independence(k_alternative_preds) #Currently assumes temporal independence
				# 																	#dictionary keys: traj index within batch (0-31); 
				# 	 																#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)
				# # # (B, K, T, 2)
				# # priors = self.compute_batch_motion_priors_kde_joined(k_alternative_preds) # currently ignores time completely/does not work
				# # print(priors)

				for i in range(24):
					print(k_alternative_preds[0,0,i,:])

				exit()
				self.compute_kde_and_vis_full_traj(k_alternative_preds, past_traj[:,:,2:4], fut_traj)
				exit()

				for traj_idx in range(batch_size):

					for time in range(fut_traj.size(1)):
						single_pose_GT = fut_traj[traj_idx, time, :]

						#single_pose_all_ks = k_alternative_preds[traj_idx, :, time, :]
						#single_pose_KDE = self.KDE_single_pose_outlier_filtered(single_pose_all_ks.detach().cpu().numpy()) #outlier filtering
						single_pose_KDE = priors[traj_idx][time] # KDE based on all k_preds for the pose #uncomment for no outlier filtering
						
						#single_pose_past = past_traj[traj_idx, :, 2:4] #2:4 are relative poses - first two channels of past are absolute traj, then relative, then velocities					
						#self.visualise_single_KDE_GT_Past(single_pose_all_ks, single_pose_KDE, single_pose_past, single_pose_GT, time) 
					
						### Probability density
						GT_pose_np = single_pose_GT.detach().cpu().numpy().reshape(-1, 1)
						density = single_pose_KDE(GT_pose_np)[0]

						all_densities_by_time[time].append(density)

						#print(f"Probability Density of GT pose at time {time}: {density}")


				### Regular code continues
				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - k_alternative_preds, dim=-1) * self.traj_scale

				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()

				samples += distances.shape[0]
				count += 1
					# if count==2:
					# 	break

			
			### save KDE density-per-t histograms
			self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)

			avg_densities_per_t = {t: np.array(all_densities_by_time[t]).mean() for t in range(self.cfg.future_frames)}
			print('avg_densities_per_t:\n', avg_densities_per_t)

		for time_i in range(1,20,5):
			print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		