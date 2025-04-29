import sys
import os
import time
import torch
import torch.nn.functional as F

import copy
import random
import numpy as np
import torch.nn as nn
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from utils.config import Config
from utils.utils import print_log

from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate
from _rotation_utils import quaternion_relative, relative_lie_rotation, rotmat_to_rot6d, rot6d_to_rotmat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #LoaderKitti is two levels up
from PoseLoaderCustom import LoadDatasetLeapfrog, seq_collate_custom


#from trainer.kde_utils import find_max


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
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
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
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
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
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
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
				t_f=self.cfg.future_frames, 
				d_f=self.cfg.dimensions, 
				k_pred=self.cfg.k_preds
			).cuda()
		
		else:
			print('[INFO] Training from scratch - without pretrained models (Not NBA with frame standard configs)\n')
			# print('Params for model_initialiser: ', self.cfg.past_frames, self.cfg.dimensions*3, self.cfg.future_frames, self.cfg.dimensions, self.cfg.k_preds)
			self.model_initializer = InitializationModel( #DIM update delete
				t_h=self.cfg.past_frames, 
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
		ckpt_path = os.path.join(checkpoint_dir, f'best_checkpoint_epoch_{epoch}.pth')
		
		# delete previous best checkpoint, if it exists
		if hasattr(self, 'best_checkpoint_path') and self.best_checkpoint_path is not None:
			if os.path.exists(self.best_checkpoint_path):
				os.remove(self.best_checkpoint_path)
		
		torch.save(checkpoint, ckpt_path)
		self.best_checkpoint_path = ckpt_path
		print_log(f"[INFO] New best model (ATE)! Checkpoint saved to {ckpt_path}", self.log)

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
		#sample = mean + sigma_t * z * 0.1

		return (sample)
	
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
					epoch+1, loss_total, loss_distance, loss_uncertainty), self.log)
			
			elif self.cfg.dimensions in [6,7,9]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Rotation.: {:.6f}\tCombined Loss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch+1, loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty), self.log)


			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples= self._test_single_epoch() #average_euclidean = average total distance start to finish - to contextualise how good the FDE and ADE are

				# Print ADE/FDE metrics as before
				timesteps = list(range(5, self.cfg.future_frames, 5))
				if not timesteps or timesteps[-1] != self.cfg.future_frames:
					timesteps.append(self.cfg.future_frames)
				for i, time_i in enumerate(timesteps):
					print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
						time_i, performance['ADE'][i]/samples,
						time_i, performance['FDE'][i]/samples), self.log)
				
				# Print new ATE metrics
				print_log('--ATE translation: {:.4f}'.format(performance['ATE_trans']), self.log)
				
				# Update best model selection to include ATE if desired
				ate_trans = performance['ATE_trans']
				ade_final = performance['ADE'][-1]/samples
				
				if epoch == 0:
					best_ate = ate_trans
					best_ade = ade_final
				elif ate_trans < best_ate:  # Use ATE_trans as primary metric
					best_ate = ate_trans
					self.save_checkpoint(epoch+1)
				# elif ade_final < best_ade:  # Or keep using ADE as fallback
				# 	best_ade = ade_final
				# 	self.save_checkpoint(epoch+1)
					
				#print ADE/FDE of timesteps that are a multiple of 5 and final timestep ADE/FDE
				# timesteps = list(range(5, self.cfg.future_frames, 5))
				# if not timesteps or timesteps[-1] != self.cfg.future_frames:
				# 	timesteps.append(self.cfg.future_frames)
				

				# for i, time_i in enumerate(timesteps): #self.cfg.future_frames
				# 	print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
				# 		time_i, performance['ADE'][i]/samples,
				# 		time_i, performance['FDE'][i]/samples), self.log)

				# #save model if it's the best so far
				# ade_final_pose = performance['ADE'][-1]/samples
				# if epoch == 0:
				# 	best_ade = ade_final_pose
				# elif ade_final_pose < best_ade:
				# 	best_ade = ade_final_pose
				# 	self.save_checkpoint(epoch+1)

			self.scheduler_model.step()


	def data_preprocess_with_abs(self, data): #Updated to handle any number of agents
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

		#print(data['pre_motion_3D'][0,0,:,:])

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
		batch_size = data['pre_motion_3D'].shape[0]
		num_agents = data['pre_motion_3D'].shape[1]
		#print(data['pre_motion_3D'][0,0,:,:])


		### 1.0 Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
		traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
		for i in range(batch_size):
			traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.


		### 2.0 compute relative translation and rotation of past and future trajectory, compute velocities of translation
		if self.cfg.dimensions in [2,3]:
			# Get last observed pose (for each agent) as initial position
			last_observed_pos = data['pre_motion_3D'].cuda()[:, :, -1:] # 2D: [B, num_agents, 1, 2] or 3D: [B, num_agents, 1, 3]

			# relative positions
			past_rel = (data['pre_motion_3D'].to(self.device) - last_observed_pos) / self.traj_scale  # [B, num_agents, past_frames, d]
			# velocities (difference between consecutive relative positions)
			past_vel = torch.cat((past_rel[:, :, 1:] - past_rel[:, :, :-1],
								torch.zeros_like(past_rel[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, d]
			
			
			# concat relative positions and velocities
			past_traj = torch.cat((past_rel, past_vel), dim=-1)  # [B, num_agents, past_frames, 2*d]

			# relative future trajectory
			fut_traj = (data['fut_motion_3D'].to(self.device) - last_observed_pos) / self.traj_scale  # [B, num_agents, future_frames, d]
			
			# reshape to merge batch and agent dimensions:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)
		
		elif self.cfg.dimensions ==6:
			# For 6D: Data is expected to be of shape [B, num_agents, past_frames, 6] for past,
			# and [B, num_agents, future_frames, 6] for future.
			past_abs = data['pre_motion_3D'].to(self.device)  # [B, num_agents, past_frames, 6]
			fut_abs  = data['fut_motion_3D'].to(self.device)  # [B, num_agents, future_frames, 6]
			
			# Split into translation and rotation (Lie algebra) parts:
			past_trans = past_abs[..., :3]  # [B, num_agents, past_frames, 3]
			past_rot   = past_abs[..., 3:]  # [B, num_agents, past_frames, 3]
			
			fut_trans = fut_abs[..., :3]    # [B, num_agents, future_frames, 3]
			fut_rot   = fut_abs[..., 3:]    # [B, num_agents, future_frames, 3]
			
			# Get the last observed pose per agent:
			last_obs = past_abs[:, :, -1:]  # [B, num_agents, 1, 6]
			last_trans = last_obs[..., :3]  # [B, num_agents, 1, 3]
			last_rot = last_obs[..., 3:]    # [B, num_agents, 1, 3]
			
			# Compute relative translation (for past and future):
			rel_trans_past = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			rel_trans_fut  = (fut_trans - last_trans) / self.traj_scale   # [B, num_agents, future_frames, 3]
			
			# Compute relative rotation for past:
			# We need to compute, for each time step, the relative rotation given last_rot.
			# We'll reshape to merge batch, agent, and time dimensions, apply our helper, then reshape back.
			B, N, T, _ = past_rot.shape
			past_rot_flat = past_rot.reshape(-1, 3)              # (B*N*T, 3)
			last_rot_expanded = last_rot.expand(B, N, T, 3).reshape(-1, 3)  # (B*N*T, 3)
			rel_rot_past_flat = relative_lie_rotation(past_rot_flat, last_rot_expanded)
			rel_rot_past = rel_rot_past_flat.view(B, N, T, 3)     # [B, num_agents, past_frames, 3]
			
			# Concatenate to obtain past relative pose (6D)
			past_rel = torch.cat((rel_trans_past, rel_rot_past), dim=-1)  # [B, num_agents, past_frames, 6]
			
			# Compute translation velocity for past (only on translation):
			past_vel = torch.cat((rel_trans_past[:, :, 1:] - rel_trans_past[:, :, :-1],
								torch.zeros_like(rel_trans_past[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, 3]
			
			# Concatenate past relative pose and translation velocity → per-timestep input of 9 dims:
			past_traj = torch.cat((past_rel, past_vel), dim=-1)  # [B, num_agents, past_frames, 9]
			
			# For future: compute relative rotation similarly:
			B_f, N_f, T_f, _ = fut_rot.shape
			fut_rot_flat = fut_rot.reshape(-1, 3)
			last_rot_fut = last_rot.expand(B_f, N_f, T_f, 3).reshape(-1, 3)
			rel_rot_fut_flat = relative_lie_rotation(fut_rot_flat, last_rot_fut)
			rel_rot_fut = rel_rot_fut_flat.view(B_f, N_f, T_f, 3)  # [B, num_agents, future_frames, 3]
			
			fut_rel = torch.cat((rel_trans_fut, rel_rot_fut), dim=-1)  # [B, num_agents, future_frames, 6]
			
			# Reshape: merge the batch and agent dimensions
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_rel.view(-1, self.cfg.future_frames, self.cfg.dimensions)  # here self.cfg.dimensions == 6
    
		elif self.cfg.dimensions == 7: # TUM format: Absolute SE(3) poses: [B, num_agents, T, 7] (translations + quaternions)

			last_obs_pose = data['pre_motion_3D'].to(self.device)[:, :, -1:]  # [B, num_agents, 1, 7]
			
			# translation: subtract last observed translation.
			past_abs = data['pre_motion_3D'].to(self.device)  # [B, num_agents, past_frames, 7]
			past_trans = past_abs[..., :3]               # translations: [B, num_agents, past_frames, 3]
			last_trans  = last_obs_pose[..., :3]           # [B, num_agents, 1, 3]
			rel_trans = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			
			# rotation: extract quaternions and compute relative quaternion.
			past_quat = past_abs[..., 3:7]                 # [B, num_agents, past_frames, 4]
			last_quat = last_obs_pose[..., 3:7]             # [B, num_agents, 1, 4]
			rel_quat = quaternion_relative(last_quat, past_quat)  # [B, num_agents, past_frames, 4]

			# combine past relative pose - concat of relative translation and quaternion
			past_rel = torch.cat((rel_trans, rel_quat), dim=-1)  # [B, num_agents, past_frames, 7]

			# velocities, compute for translation only TODO not for rot?
			past_vel = torch.cat((rel_trans[:, :, 1:] - rel_trans[:, :, :-1],
								torch.zeros_like(rel_trans[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, 3]
			
			# concat past relative pose (7) and translation velocity (3): XXX TODO so velocity of rotation can't be computed?
			past_traj = torch.cat((past_rel, past_vel), dim=-1)  # [B, num_agents, past_frames, 10]   <<< output size


			# relative future trajectory
			fut_abs = data['fut_motion_3D'].to(self.device)  # [B, num_agents, future_frames, 7]
			fut_trans = fut_abs[..., :3]
			fut_rel_trans = (fut_trans - last_trans) / self.traj_scale  # [B, num_agents, future_frames, 3]
			fut_quat = fut_abs[..., 3:7]
			fut_rel_quat = quaternion_relative(last_quat, fut_quat)  # [B, num_agents, future_frames, 4]
			fut_traj = torch.cat((fut_rel_trans, fut_rel_quat), dim=-1)  # [B, num_agents, future_frames, 7]   <<< output size


			# Reshape so that each agent is a separate trajectory:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)

		elif self.cfg.dimensions == 9: 
			# 2. Get absolute poses for past and future
			past_abs = data['pre_motion_3D'].to(self.device)  # shape: [B, num_agents, past_frames, 9]
			fut_abs  = data['fut_motion_3D'].to(self.device)   # shape: [B, num_agents, future_frames, 9]
			
			# 3. Split into translation and rotation components
			past_trans = past_abs[..., :3]      # [B, num_agents, past_frames, 3]
			past_rot6d = past_abs[..., 3:]      # [B, num_agents, past_frames, 6]
			
			fut_trans = fut_abs[..., :3]        # [B, num_agents, future_frames, 3]
			fut_rot6d = fut_abs[..., 3:]        # [B, num_agents, future_frames, 6]
			
			# 4. Last observed absolute pose for each agent (from past)
			last_obs = past_abs[:, :, -1:]      # [B, num_agents, 1, 9]
			last_trans = last_obs[..., :3]       # [B, num_agents, 1, 3]
			last_rot6d = last_obs[..., 3:]       # [B, num_agents, 1, 6]
			
			# 5. Compute relative translation
			rel_trans_past = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			rel_trans_fut  = (fut_trans  - last_trans) / self.traj_scale   # [B, num_agents, future_frames, 3]
			
			# 6. Compute relative rotation for past:
			# Convert past 6d rotations to rotation matrices:
			R_past = rot6d_to_rotmat(past_rot6d.reshape(-1, 6))  # shape: [B*num_agents*past_frames, 3, 3]
			R_past = R_past.view(batch_size, num_agents, past_abs.shape[2], 3, 3)  # [B, num_agents, past_frames, 3, 3]
			# Convert reference (last) rotation to a matrix:
			R_ref = rot6d_to_rotmat(last_rot6d.squeeze(2))       # [B, num_agents, 3, 3]
			# Expand R_ref along time:
			R_ref_exp = R_ref.unsqueeze(2).expand_as(R_past)       # [B, num_agents, past_frames, 3, 3]
			# Compute relative rotation: R_rel = R_ref^T @ R_past.
			R_rel_past = torch.matmul(R_ref_exp.transpose(-2, -1), R_past)   # [B, num_agents, past_frames, 3, 3]
			# Convert relative rotation matrix back to 6d:
			rel_rot6d_past = rotmat_to_rot6d(R_rel_past.view(-1, 3, 3))  # [B*num_agents*past_frames, 6]
			rel_rot6d_past = rel_rot6d_past.view(batch_size, num_agents, past_abs.shape[2], 6)  # [B, num_agents, past_frames, 6]
			
			# For future
			R_fut = rot6d_to_rotmat(fut_rot6d.reshape(-1, 6))  # [B*num_agents*future_frames, 3, 3]
			R_fut = R_fut.view(batch_size, num_agents, fut_abs.shape[2], 3, 3)  # [B, num_agents, future_frames, 3, 3]
			R_ref_fut = R_ref.unsqueeze(2).expand_as(R_fut)    # [B, num_agents, future_frames, 3, 3]
			R_rel_fut = torch.matmul(R_ref_fut.transpose(-2, -1), R_fut)  # [B, num_agents, future_frames, 3, 3]
			rel_rot6d_fut = rotmat_to_rot6d(R_rel_fut.view(-1, 3, 3))  # [B*num_agents*future_frames, 6]
			rel_rot6d_fut = rel_rot6d_fut.view(batch_size, num_agents, fut_abs.shape[2], 6)  # [B, num_agents, future_frames, 6]
			
			# 7. Concatenate translation and rotation for past relative pose.
			past_rel_pose = torch.cat((rel_trans_past, rel_rot6d_past), dim=-1)  # [B, num_agents, past_frames, 3+6 = 9]
			
			# 8. Compute translational velocity for past (from relative translation only).
			past_vel = torch.cat((rel_trans_past[:, :, 1:] - rel_trans_past[:, :, :-1],
								torch.zeros_like(rel_trans_past[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, 3]
			
			# 9. Final past processed trajectory: concatenate relative pose (9) with translation velocity (3) → 12 per timestep.
			past_traj = torch.cat((past_rel_pose, past_vel), dim=-1)  # [B, num_agents, past_frames, 12]
			
			# 10. Final future processed trajectory: simply the relative pose (9)
			fut_traj = torch.cat((rel_trans_fut, rel_rot6d_fut), dim=-1)  # [B, num_agents, future_frames, 9]
			
			# 11. Merge batch and agent dimensions:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)  # here self.cfg.dimensions should be 9
  
		else:
			raise NotImplementedError("data_preprocess for dimensions 6,7,9,2,3 are implemented; others not yet.")

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
		vis_path = f'./visualization/Sanity_Synthetic_Single_Timestep_{idx}.jpg'
		plt.savefig(vis_path)
		print(f'[INFO] Visualisation saved to {vis_path}')
		plt.close()

	def compute_kde_and_vis_full_traj(self, k_preds, past_traj, fut_traj, experiment_name, exclude_last_timestep=True, kpreds=True): #TODO how is KDE computed here
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

		if exclude_last_timestep:
			k_preds_np = k_preds_np[:, :, :-1, :]  # Now T becomes T-1
			fut_np = fut_np[:, :-1, :]
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
			margin = 0.1  # Adjust this margin if necessary.
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
			if kpreds==True:
				plt.savefig(f'./visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_denoised_kpreds.jpg')
				print(f"[INFO] Saved KDE visualization with kpreds at './visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_denoised_kpreds.jpg'")
			else:
				plt.savefig(f'./visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_initialised_only.jpg')

			plt.close()

	
	def compute_batch_motion_priors_kde_temporal_independence(self, k_preds):#consider that we only visualise xy TODO
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
		
		for i, data in enumerate(self.train_loader):
			# torch.set_printoptions(
			# 	precision=4,   # number of digits after the decimal
			# 	sci_mode=False # turn off scientific (e+) notation
			# )
			# print(f"first traj all poses (pre): ", data['pre_motion_3D'][0,0,:,:]) #first traj all poses (pre) (B, A, T, D)
			# print(f"first traj all poses (fut): ", data['fut_motion_3D'][0,0,:,:]) #first traj all poses (fut)
			# exit()
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data) # past_traj =(past_traj_abs, past_traj_rel, past_traj_vel)
			# if i in [0,1]:
			# 	print("past_traj[0,0,:] on bad batch:", past_traj[0,0,:].detach().cpu())
			#print('fut_traj:', fut_traj[0,0,:]) #first fut timestep


			# print('traj_mask:', traj_mask.size()) # [32, 32]
			# print('past_traj processed:', past_traj[0]) # [32, T, D+3] 
			# print('fut_traj processed:', fut_traj[0]) # [32, T, D+3] < GT poses for future_frames timesteps
			# exit()
			
			### 1. Leapfrogging Denoising (LED initializer outputs): instead of full denoising process, predicts intermediate, already denoised trajectory 
			### the 9d nan issue happens here on the second batch in the train loop
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) #offset, mean, stdev
			# print("— raw_pred[0,0,0,:] pre‐reparam:", sample_prediction[0,0,0,:].detach().cpu().numpy())
			# print("— mean_est[0,:]           :", mean_estimation[0,:].detach().cpu().numpy())
			# print("— var_est[0,:]            :", variance_estimation[0,:]e.dtach().cpu().numpy())
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
			# denom = sample_prediction.std(dim=1).mean(dim=(1,2))  # a single scalar per batch element
			# with torch.no_grad():
			# 	denom = sample_prediction.std(dim=1).mean(dim=(1,2))  # shape [B]
			# 	print("variance head:", variance_estimation.min().item(), variance_estimation.max().item())
			# 	print("denom min/mean/max:", denom.min().item(), denom.mean().item(), denom.max().item())


			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]

			# print('sample_prediction')
			# print(sample_prediction[0,0,0,:]) #check if prediction is 9D
			# print('This shouldnt be super small - if it is, this explains outliers:', sample_prediction.std(dim=1).mean(dim=(1, 2)))
			
			# Add the mean estimation to the scaled normalized offsets / center predictions around the mean (each candidate trajectory is expressed as a deviation from the mean)

			loc = sample_prediction + mean_estimation[:, None] #prediction before denoising
			# print("— loc min/max:", loc.min().item(), loc.max().item())


			# print('sample_prediction:', sample_prediction.size())
			# print('loc:', loc.size()) #[32, 24, 24, 3]

			### 2. Denoising (Denoising Module): Generate K alternative future trajectories - multi-modal
			k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) #(B, K, T, 2/3/6/7/9)
			# print('k_alternative_preds:',k_alternative_preds[0,0,0,:])

			
			# print('k_alternative_preds:', k_alternative_preds[0,:,0,:2]) #check if prediction is 9D

			# scale = torch.exp(variance_estimation/2)
			# Log statistics for each batch element and each prediction (over K)
			# print("Variance stats per batch element:")
			# for i in range(scale.shape[0]):
			# 	print("Batch {}: min={:.3f}, max={:.3f}, mean={:.3f}".format(
			# 		i, scale[i].min().item(), scale[i].max().item(), scale[i].mean().item()))

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
			
			elif self.cfg.dimensions == 7:
				# For 7D, we need to split the 7 dimensions: translation (first 3) and quaternion (last 4)
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)  # shape: (B, 1, T, 7)
				# Split predictions and ground truth:
				pred_trans = k_alternative_preds[..., :3]  # shape: (B, K, T, 3)
				pred_quat  = k_alternative_preds[..., 3:]  # shape: (B, K, T, 4)
				gt_trans   = fut_traj_wpreds[..., :3]        # shape: (B, 1, T, 3)
				gt_quat    = fut_traj_wpreds[..., 3:]        # shape: (B, 1, T, 4)

				# Compute translation error (L2 norm)
				trans_error = (pred_trans - gt_trans).norm(p=2, dim=-1)  # (B, K, T)
				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1).min(dim=1)[0].mean()

				# Normalize quaternions (add a small epsilon to avoid div-by-zero)
				pred_quat_norm = pred_quat / (pred_quat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
				gt_quat_norm   = gt_quat   / (gt_quat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
				
				# Compute dot product between normalized quaternions and clamp to valid range [-1,1]
				dot = (pred_quat_norm * gt_quat_norm).sum(dim=-1).abs().clamp(max=1.0 - 1e-6)  # (B, K, T)
				# Angular error (in radians)
				rot_error = torch.acos(dot)  # (B, K, T)
				loss_rotation = rot_error.mean(dim=-1).min(dim=1)[0].mean()

				# Combined loss over translation and rotation.
				loss_distance = loss_translation + loss_rotation


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

			"""General 2D/3D/6D/7D/9D code continues here"""
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

			if self.cfg.dimensions in [7]:
				loss_trans += loss_translation.item()
				loss_rot += loss_rotation.item()
				
			self.opt.zero_grad()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()

			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total/count, loss_trans/count, loss_rot/count, loss_dt/count, loss_dc/count

	def _test_single_epoch(self):
		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE': [0] * len(timesteps),
			'FDE': [0] * len(timesteps),
			'ATE_trans': 0
		}
		# performance = { 'FDE': [0, 0, 0, 0],
		# 				'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)

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
				fut_traj_expanded = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)  # (B, K, T, D)
				
				# ATE metric (translation only)
				ate_results = self.compute_ate(
					pred_traj, 
					fut_traj.unsqueeze(1),  # Add K dimension to ground truth
					self.cfg.dimensions,
					self.traj_scale
				)

				# Add ATE results to performance metrics
				performance['ATE_trans'] += ate_results['ate_trans'] * batch_size

				# ---- Translation Error Metrics (already in your code) ----
				# Calculate traditional ADE/FDE metrics (from your original code)
				# Extract translational components for ADE/FDE calculation
				if self.cfg.dimensions in [2, 3]:
					pred_traj_trans = pred_traj
					fut_traj_trans = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)
				else:
					pred_traj_trans = pred_traj[..., :3]  # (B, K, T, 3)
					fut_traj_trans = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)[..., :3]  # (B, K, T, 3)
				
				distances = torch.norm(fut_traj_trans - pred_traj_trans, dim=-1) * self.traj_scale  ## Euclidian translation errors (B, K, T)
				# print('distances: ', distances)


            	# Compute ADE and FDE at different timesteps.
				# Here we compute ADE and FDE for time steps: every 5th and final
				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps
					"""
					1s: 5 * 1 - 1 = 4 → Requires at least 5 timesteps.
					2s: 5 * 2 - 1 = 9 → Requires at least 10 timesteps.
					3s: 5 * 3 - 1 = 14 → Requires at least 15 timesteps.
					4s: 5 * 4 - 1 = 19 → Requires at least 20 timesteps.
					"""
					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()
				samples += distances.shape[0]
			
		# Normalize ATE metrics by number of samples
		performance['ATE_trans'] /= samples

		return performance, samples

	def compute_ate(self, pred_trajectories, gt_trajectory, dimensions, traj_scale=1.0):
		"""
		Compute Absolute Trajectory Error (ATE) between predicted trajectories and ground truth.
		Focus on translational error only, regardless of pose representation.
		
		Args:
			pred_trajectories: Predicted trajectories of shape (B, K, T, D) where:
				B = batch size
				K = number of predictions per sample
				T = number of timesteps
				D = dimensions (2/3/6/7/9)
			gt_trajectory: Ground truth trajectory of shape (B, 1, T, D)
			dimensions: Dimensionality of the pose (2/3/6/7/9)
			traj_scale: Scale factor used during preprocessing
		
		Returns:
			Dictionary containing ATE metric:
			- 'ate_trans': Translational ATE (for all dimension types)
		"""
		# Initialize variables to store results
		B, K, T, D = pred_trajectories.shape
		results = {}
		
		# Extract translational component based on pose representation
		if dimensions in [2, 3]:
			# For 2D/3D trajectories, the entire representation is positional
			pred_trans = pred_trajectories * traj_scale  # (B, K, T, 2/3)
			gt_trans = gt_trajectory * traj_scale  # (B, 1, T, 2/3)
		else:
			# For 6D/7D/9D poses, extract just the translation part (first 3 dimensions)
			pred_trans = pred_trajectories[..., :3] * traj_scale  # (B, K, T, 3)
			gt_trans = gt_trajectory[..., :3] * traj_scale  # (B, 1, T, 3)
		
		# Compute translational errors for each candidate trajectory
		trans_errors = torch.norm(gt_trans - pred_trans, dim=-1)  # (B, K, T)
		
		# Select the best prediction among K candidates based on translation error only
		k_indices = trans_errors.mean(dim=-1).argmin(dim=-1)  # (B,)
		
		# Extract best trajectory for each sample
		batch_indices = torch.arange(B).to(pred_trajectories.device)
		best_trans = pred_trans[batch_indices, k_indices]  # (B, T, D_trans)
		
		# Compute RMSE of the best trajectory compared to GT
		# This is the true ATE - square root of the mean squared error across entire trajectory
		ate_trans = torch.sqrt(((best_trans - gt_trans.squeeze(1))**2).sum(dim=-1).mean(dim=-1))  # (B,)
		
		# Average across the batch
		results['ate_trans'] = ate_trans.mean().item()
		
		return results


	def test_single_model(self, checkpoint_path = None):
		# checkpoint_path = './results/5_Experiments/checkpoint_rework/models/checkpoint_epoch_40.pth'
		#checkpoint_path = './results/5_1_Overfitting_VisCheck/5_1_Overfitting_VisCheck/models/checkpoint_epoch_63.pth'
		# checkpoint_path = './results/6_Testing/9D_kitti/models/best_checkpoint_epoch_40.pth' #should respect rotation
		#checkpoint_path = './results/6_Testing/2D_kitti/models/best_checkpoint_epoch_37.pth' #does not respect rotation
		# checkpoint_path = './results/6_Testing_9D_newer/9D_newer/models/best_checkpoint_epoch_51.pth'
		# checkpoint_path = './results/6_Integration_7D/6_Integration_7D/models/best_checkpoint_epoch_27.pth'
		
		# checkpoint_path = './results/6_Developing_Synthetic/6_Developing_Synthetic/models/best_checkpoint_epoch_61.pth'
		# checkpoint_path = './results/6_Developing_Synthetic_Overfitting/6_Developing_Synthetic_Overfitting/models/best_checkpoint_epoch_57.pth'
		#checkpoint_path = './results/6_HighData_Synthetic/6_HighData_Synthetic/models/best_checkpoint_epoch_38.pth'
		# checkpoint_path = './results/6_HighData_Synthetic/6_500kHighData_Synthetic/models/best_checkpoint_epoch_15.pth'
		# checkpoint_path = './results/6_2_Testing_GT-N-Synthetic/FirstTry/models/best_checkpoint_epoch_9.pth'
		# checkpoint_path = './results/6_2_Testing_Synthetic/6_2_Testing_Synthetic/models/best_checkpoint_epoch_19.pth'
		checkpoint_path = './results/6_2_Testing_Synthetic/6_2_Testing_Synthetic_6D/models/best_checkpoint_epoch_25.pth'
		experiment_name = checkpoint_path.split('/')[3]

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()

		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE': [0] * len(timesteps),
			'FDE': [0] * len(timesteps)
		}

		# performance = { 'FDE': [0, 0, 0, 0],
		# 				'ADE': [0, 0, 0, 0]}
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
				# print(past_traj[0, :, :]) 
				# print(past_traj.size())
				# first position past traj - should be relativised to [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00, 0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00]
				# exit()

				# Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			
				initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions


				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
				

				# print(k_alternative_preds.size())  # (B, K, T, D)
				# print(fut_traj.size()) # (B, T, D)
				# print(past_traj.size()) # (32, 10, D)

				
				
				# ### ============= motion prior + visualisation =============
				# self.compute_kde_and_vis_full_traj(k_alternative_preds)
				initializer_preds_xy = initializer_preds[:,:,:,:2]
				k_alternative_preds_xy = k_alternative_preds[:,:,:,:2]
				#print(k_alternative_preds_xy.size())


				priors = self.compute_batch_motion_priors_kde_temporal_independence(k_alternative_preds_xy) #Currently assumes temporal independence
				# 																	#dictionary keys: traj index within batch (0-31); 
				# 	 																#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)
				# # # (B, K, T, 2)
				# # priors = self.compute_batch_motion_priors_kde_joined(k_alternative_preds) # currently ignores time completely/does not work

				# for i in range(24):
				# 	print(k_alternative_preds[0,0,i,:])

				### not used anymore - this actually gets the velocity, use raw traj instead
				# past_traj_relative = past_traj[:, :, 1*self.cfg.dimensions:2*self.cfg.dimensions]
				# past_traj_xy = past_traj_relative[:,:,:2]
				# fut_traj_xy = fut_traj[:,:,:2]

				###raw traj
				raw_past = data['pre_motion_3D'][:, 0, :, :2]   # [B, past_frames, 2]
				raw_fut  = data['fut_motion_3D'][:, 0, :, :2]   # [B, fut_frames,   2]
				# relative to the last observed point:
				last_obs  = raw_past[:, -1:, :]                # [B, 1, 2]
				past_traj_xy = (raw_past - last_obs) / self.traj_scale
				fut_traj_xy  = (raw_fut  - last_obs) / self.traj_scale

				# print(k_alternative_preds_xy.size(), past_traj_xy.size(), fut_traj_xy.size())
				

				# print(k_alternative_preds[0,:,0,:])
				# exit()
				
				# self.compute_kde_and_vis_full_traj(initializer_preds_xy, past_traj_xy, fut_traj_xy, True, False)
				self.compute_kde_and_vis_full_traj(k_alternative_preds_xy, past_traj_xy, fut_traj_xy, experiment_name, True, True)
				exit()

				
				for traj_idx in range(batch_size):

					for time in range(fut_traj.size(1)):
						single_pose_GT = fut_traj[traj_idx, time, :]

						single_pose_all_ks = k_alternative_preds[traj_idx, :, time, :]
						single_pose_all_ks_xy = single_pose_all_ks[:,:2]
						# print(single_pose_all_ks_xy.size())
						# single_pose_KDE = self.KDE_single_pose_outlier_filtered(single_pose_all_ks.detach().cpu().numpy()) #outlier filtering
						single_pose_KDE = priors[traj_idx][time] # KDE based on all k_preds for the pose #uncomment for no outlier filtering
						# print(single_pose_KDE)

						single_absolute_pose_past = past_traj[traj_idx, :, 1*self.cfg.dimensions:2*self.cfg.dimensions] #there are 3*dims poses - the first dim=n ones are absolute, the middle ones relative, then velocities. we need the middles ones-velocities					
						single_absolute_pose_past_xy = single_absolute_pose_past[:,:2] #9D starts with tx,ty,tz,r....
						# print(single_absolute_pose_past_xy.size())

						single_pose_GT_xy = single_pose_GT[:2]
						# print(single_pose_GT_xy.size())
						# print(single_pose_all_ks_xy)

						self.visualise_single_KDE_GT_Past(single_pose_all_ks_xy, single_pose_KDE, single_absolute_pose_past_xy, single_pose_GT_xy, time) 


						### Probability density
						GT_pose_np = single_pose_GT_xy.detach().cpu().numpy().reshape(-1, 1)
						density = single_pose_KDE(GT_pose_np)[0]

						all_densities_by_time[time].append(density)

						#print(f"Probability Density of GT pose at time {time}: {density}")
					exit()
				
				### Regular code continues
				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - k_alternative_preds, dim=-1) * self.traj_scale


				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()

				# for time_i in range(1, 5):
				# 	ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
				# 	fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
				# 	performance['ADE'][time_i-1] += ade.item()
				# 	performance['FDE'][time_i-1] += fde.item()

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
		





# ----------- not used ---------------
"""
def save_vis_data(self):
	
	### Save the visualization data.
	
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

def p_sample_loop(self, x, mask, shape):
	self.model.eval()
	prediction_total = torch.Tensor().cuda()
	for _ in range(20):
		cur_y = torch.randn(shape).to(x.device)
		for i in reversed(range(self.n_steps)):
			cur_y = self.p_sample(x, mask, cur_y, i)
		prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
	return prediction_total


def noise_estimation_loss(self, x, y_0, mask):
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
"""