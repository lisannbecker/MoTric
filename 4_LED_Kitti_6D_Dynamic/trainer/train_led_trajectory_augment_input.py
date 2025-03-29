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
from PoseLoaderCustom import KITTIDatasetLeapfrog, seq_collate_kitti


from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

import pdb
NUM_Tau = 5

class Trainer:
	def __init__(self, config): 
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)
		self.cfg.dataset = config.dataset #use kitti dataset if specified in command line

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

		elif self.cfg.dataset.lower() == 'kitti':
			dataloader_class = KITTIDatasetLeapfrog
			collate_fn = seq_collate_kitti
			print("[INFO] KITTI dataset (1 agent).")

			train_dset = dataloader_class(
				dims=self.cfg.dimensions,
				input_size=self.cfg.past_frames,
				preds_size=self.cfg.future_frames,
				training=True,
				final_eval=False,
				relative=self.cfg.relative, 
				normalised=self.cfg.normalised, 
				train_ratio=0.85,
				seed=42,
				overlapping = False
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
				dims=self.cfg.dimensions,
				input_size=self.cfg.past_frames,
				preds_size=self.cfg.future_frames,
				training=False,
				final_eval=False,
				relative=self.cfg.relative, 
				normalised=self.cfg.normalised, 
				train_ratio=0.85,
				seed=42,
				overlapping = False
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
				print("\nTest dataset:")
				self.print_some_stats(test_dset.fut_motion_3D, None, 2)

			elif self.cfg.dimensions == 3:
				print("Train dataset:")
				self.print_some_stats(train_dset.fut_motion_3D, None, 3)
				print("\nTest dataset:")
				self.print_some_stats(test_dset.fut_motion_3D, None, 3)
			
			elif self.cfg.dimensions == 6:
				print("Train dataset:")
				self.print_some_stats(train_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)
				print("\nTest dataset:")
				self.print_some_stats(test_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)


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
			print('\n[INFO] Kitti dataset - skip subtracting mean from absolute positions.')
			
		
		
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
		self.model = CoreDenoisingModel(t_h=self.cfg.past_frames,d_f=self.cfg.dimensions).cuda()

		if self.cfg.past_frames == 10 and self.cfg.future_frames == 20 and self.cfg.dataset == 'nba':
			# load pretrained models 
			print('[INFO] Loading pretrained models... (NBA with standard frame configs)\n')
			model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu') #LB expects 60 dimensional input (6 x 10 past poses)
			self.model.load_state_dict(model_cp['model_dict'])

			self.model_initializer = InitializationModel(t_h=self.cfg.past_frames, d_h=self.cfg.dimensions*3, t_f=self.cfg.future_frames, d_f=self.cfg.dimensions, k_pred=self.cfg.k_preds).cuda()
		
		else:
			print('[INFO] Training from scratch - without pretrained models (Not NBA with frame standard configs)\n')
			# print('Params for model_initialiser: ', self.cfg.past_frames, self.cfg.dimensions*3, self.cfg.future_frames, self.cfg.dimensions, self.cfg.k_preds)
			self.model_initializer = InitializationModel(t_h=self.cfg.past_frames, d_h=self.cfg.dimensions*3, t_f=self.cfg.future_frames, d_f=self.cfg.dimensions, k_pred=self.cfg.k_preds).cuda()

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

		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
			
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		# shape: B=b*n, K=10, T, 2
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)

		return prediction_total


	def fit(self):
		# Training loop
		for epoch in range(0, self.cfg.num_epochs):
			loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)

			if self.cfg.dimensions in [2,3]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch, loss_total, loss_distance, loss_uncertainty), self.log)
			
			elif self.cfg.dimensions == 6:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Rotation.: {:.6f}\tCombined Loss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch, loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty), self.log)


			if (epoch + 1) % self.cfg.test_interval == 0: #TODO have a look here
				performance, samples= self._test_single_epoch() #average_euclidean = average total distance start to finish - to contextualise how good the FDE and ADE are
				for i, time_i in enumerate(range(5,21,5)):
					print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
						time_i, performance['ADE'][i]/samples,
						time_i, performance['FDE'][i]/samples), self.log)
				cp_path = self.cfg.model_path % (epoch + 1)
				model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
				torch.save(model_cp, cp_path)

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

		# augment input: absolute position, relative position, velocity
		if self.cfg.dataset == 'kitti':
			past_traj_abs = (data['pre_motion_3D'].cuda() / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #single agent: effectively only (B, 1, Past, Dims) > (B*1, Past, Dims)
		elif self.cfg.dataset == 'nba':
			past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) 
		
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #only relativises if initial pos is not 0 already (relative = True)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1) #(B, 1, Dim)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.future_frames, self.cfg.dimensions) #relativises (if not done already) and (B, 1, Past, Dims) > (B*1, Past, Dims)

		return batch_size, traj_mask, past_traj, fut_traj

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
	
	def old_vis_prior_compute_motion_prior_kde(self, k_preds):
		"""
		Computes a KDE-based motion prior for each sample and each future time step.
		
		Args:
			k_preds_np (np.array): Predicted trajectories of shape (B, K, T, Dim).
								B: batch size, K: number of predictions, 
								T: future timesteps, 6: pose dimensions.
		
		Returns:
			priors: A list of lists such that priors[b][t] is a KDE object for sample b at time step t.
		"""
		k_preds_np = k_preds.detach().cpu().numpy()


		B, K, T, D = k_preds_np.shape
		# print(k_preds_np.shape)
		priors = []
		for b in range(B):
			np.save('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PostTraining/first_k_preds.npy', k_preds_np)
			# Option A: all Ks, all future poses
			# print(k_preds_np[b, :, 23, :])
			all_samples = k_preds_np[b].reshape(K * T, D)  # Merge all timesteps into one

			# Fit KDE using all K*T samples
			kde = gaussian_kde(all_samples.T)  

			### VIS
			# Create evaluation grid
			x_grid, y_grid = np.mgrid[-4:20:100j, -4:7:100j]
			grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Shape (2, N)
			kde_values = kde(grid_points).reshape(100, 100)  # Reshape into 2D

			# Plot KDE as a contour plot
			plt.figure(figsize=(10, 8))  # Larger canvas, less distortion
			plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
			plt.scatter(all_samples[:, 0], all_samples[:, 1], s=3, alpha=1, label="Samples")
			
			# Add labels and colorbar
			plt.xlabel("X Position")
			plt.ylabel("Y Position")
			plt.colorbar(label="Density")
			plt.legend()

			# Save the visualization
			plt.savefig(f'/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PostTraining/Trajectory{b}_AllTimesteps.jpg')
			plt.close()
			# exit()


			#All Ks, only one future pose
			sample_priors = []
			for t in range(T): # All k-preds for a specific timestep
				samples = k_preds_np[b, :, t, :]  # shape: (K, Dimension)
				# print(samples)
				# Fit a KDE for these 2D/3D/6D samples.
				kde = gaussian_kde(samples.T)  # gaussian_kde expects shape (D, N), 
				# Kernel density estimation places a smooth "kernel" (Gaussian) at each sample point and sums them to create an overall density estimate
				# Parameter: bandwidth = how smooothly the points are summed. Eg affects whether two close modes merge into one or not
				sample_priors.append(kde)

				### VIS
				if D ==2:
					# Create grid to evaluate KDE 
					#x_grid, y_grid = np.mgrid[-4:4:100j, -4:4:100j]
					x_grid, y_grid = np.mgrid[-4:20:100j, -4:7:100j]
					grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Shape (2, N)
					kde_values = kde(grid_points).reshape(100, 100)  # Reshape into 2D

					# Plot KDE as a contour plot
					plt.figure(figsize=(10, 8)) 
					plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
					plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=1, label="Samples")
					
					plt.xlabel("X Position")
					plt.ylabel("Y Position")  
					plt.colorbar(label="Density")
					plt.legend()
					#plt.show()
					plt.savefig(f'/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PostTraining/Time{t}.jpg')
					plt.close()
					
			priors.append(sample_priors)
			exit()
		return priors
	

	def visualise_single_KDE_GT_Past(self, k_preds_at_t, t_kde, all_past, GT_at_t): #TODO make size dynamic
		k_preds_at_t = k_preds_at_t.cpu().numpy()
		all_past = all_past.cpu().numpy()
		GT_at_t = GT_at_t.cpu().numpy()
		
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
		plt.savefig(f'/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/Sanity.jpg')
		plt.close()

	def visualise_kpreds_KDE(self, k_preds_at_t, t_kde):
		"""
		Visualizes KDE and predicted samples without past trajectory and GT pose.
		
		Args:
			k_preds_at_t (torch.Tensor): Shape (K, 2), predicted future poses at a specific timestep.
			t_kde (scipy.stats.gaussian_kde): KDE object representing motion prior.
		"""
		k_preds_at_t = k_preds_at_t.cpu().numpy()
		
		# Compute dynamic grid limits from predicted samples only
		margin = 1
		min_x, max_x = k_preds_at_t[:, 0].min() - margin, k_preds_at_t[:, 0].max() + margin
		min_y, max_y = k_preds_at_t[:, 1].min() - margin, k_preds_at_t[:, 1].max() + margin
		
		# Create a dynamic grid
		grid_res = 100j  # resolution: 100 points in each axis
		x_grid, y_grid = np.mgrid[min_x:max_x:grid_res, min_y:max_y:grid_res]
		grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
		kde_values = t_kde(grid_points).reshape(x_grid.shape)
		
		# Create the plot
		plt.figure(figsize=(10, 8), dpi=300)
		plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
		plt.scatter(k_preds_at_t[:, 0], k_preds_at_t[:, 1], s=3, alpha=1, color="blue", label="Predicted Samples")
		
		plt.xlabel("X Position")
		plt.ylabel("Y Position")  
		plt.colorbar(label="Density")
		plt.legend()
		
		# Save the visualization
		plt.savefig(f'/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/Sanity_NoGT_NoPast.jpg')
		plt.close()


	def compute_batch_motion_priors_kde(self, k_preds):
		"""
		Computes a KDE-based motion prior for each sample and each future time step.
		
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

	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, loss_trans, loss_rot, count = 0, 0, 0, 0, 0,0
		#LB 3D addition to reshape tensors 
		
		for data in self.train_loader:
			# print("data['fut_motion_3D'].shape: ", data['fut_motion_3D'].shape)
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data) # past_traj =(past_traj_abs, past_traj_rel, past_traj_vel)

			# first_traj_mask = traj_mask.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_traj_mask.npy', first_traj_mask)
			# first_past_traj = past_traj.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_past_traj.npy', first_past_traj)			
			# first_fut_traj= fut_traj.detach().cpu().numpy()
			# np.save('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_fut_traj.npy', first_fut_traj)
			
			# print('traj_mask:', traj_mask.size()) # [32, 32]
			# print('past_traj:', past_traj.size()) # [32, 15, 9] 
			# print('fut_traj:', fut_traj.size()) # [32, 8, 3] < GT poses for future_frames timesteps

			### LED initializer outputs (original)
			#uses the past trajectory (and possibly social context) to produce a mean and variance for the future trajectory - sampled from to get candidate future trajectories next
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)

			# print("sample_prediction shape:", sample_prediction.shape)
			# print("mean_estimation shape:", mean_estimation.shape)
			# print("variance_estimation shape:", variance_estimation.shape)
		

			# Reparameterisation with uncertainty (original)
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			# print('sample_prediction:', sample_prediction.size())


			loc = sample_prediction + mean_estimation[:, None]
			# print('loc:', loc.size()) #[32, 24, 24, 3]
			
			

			# Generate K alternative future trajectories - multi-modal
			k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) #(B, K, T, 6)

			priors = self.compute_batch_motion_priors_kde(k_alternative_preds) 	#dictionary keys: traj index within batch (0-31); 
																				#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)
			
			### motion prior during training (bad first, should get better)
			# last_GT_pose = fut_traj[0, -1]
			# last_pose_KDE = priors[0][-1]
			# print(last_GT_pose)
			# print(last_pose_KDE)

			# last_GT_pose_np = last_GT_pose.detach().cpu().numpy().reshape(-1, 1)
			# density = last_pose_KDE(last_GT_pose_np)[0]
			# print(f"Probability Density of GT final pose: {density}")
			# exit()
			###



			### 3D/2D code
			if self.cfg.dimensions in [2,3]:
				#squared distances / Euclidian, equal weight for all timesteps
				loss_distance = ((k_alternative_preds - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) * 
								self.temporal_reweight
								).mean(dim=-1).min(dim=1)[0].mean()
				loss_uncertainty = (torch.exp(-variance_estimation)*
									(k_alternative_preds - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
									variance_estimation
									).mean()
			
			elif self.cfg.dimensions == 6:
				#generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) 

				# print('Predictions batch:', generated_y.size())
				# print('GT batch:', fut_traj.size())

				# print('Prediction 0 shape:', generated_y[0].size())
				# print('GT 0 shape:', fut_traj[0].size())

				# print('Prediction:', generated_y[0])
				# print('GT:', fut_traj[0])

				# For loss, unsqueeze the ground truth to have predictions dimension
				"""6D specific code"""
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)

				# Split into translation and rot
				pred_trans = k_alternative_preds[..., :3]    # (B, K, T, 3)
				pred_rot_lie = k_alternative_preds[..., 3:]    # (B, K, T, 3)
				gt_trans = fut_traj_wpreds[..., :3]      # (B, 1, T, 3)
				gt_rot_lie = fut_traj_wpreds[..., 3:]      # (B, 1, T, 3)

				### (1) TRANSLATION LOSS
				# L2 Euclidian distance - squared distances, equal weight for timesteps
				trans_diff = pred_trans-gt_trans # (B, K, T, 3)
				trans_error = trans_diff.norm(p=2, dim=-1) # (B, K, T)

				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1)	# (B, K) loss for all k preds
				
				# print(loss_translation)
				### (2) ROTATION LOSS (Geodesic)
				#(original) squared distances / Euclidian, equal weight for all timesteps
				# loss_dist = (	(generated_y - fut_traj_wpreds).norm(p=2, dim=-1) 
				# 					* 
				# 				 self.temporal_reweight
				# 			).mean(dim=-1).min(dim=1)[0].mean()
				# loss_uncertainty = (torch.exp(-variance_estimation)
				#    						*
				# 					(generated_y - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) 
				# 						+ 
				# 					variance_estimation
				# 					).mean()
				
				# print(loss_dist)
				# print(loss_uncertainty)

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

				# Compute relative rotation: R_diff = R_pred^T * R_gt.
				R_diff = torch.matmul(pred_R.transpose(-2, -1), gt_R)  # (B, K, T, 3, 3)

				#get rotation error angle theta of rot 3x3 rot matrix
				trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # (B, K, T) trace of each individual relative rotation matrix
				# Clamp to avoid numerical issues
				angular_error_theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))  # (B, K, T) take arccosine to get error angle cos(theta) = trace(R)-1 / 2 with clamping -1 + 1e-6, 1 - 1e-6
				loss_rotation = angular_error_theta.mean(dim=-1)  # average over time, so one loss per candidate K, shape (B, K)
				# print(loss_rotation)
				### (1+2) COMBINED DISTANCE LOSS (ROT AND TRANS)
				combined_error = loss_translation + loss_rotation  # (B, K) add translation and rotation error
				loss_distance = combined_error.min(dim=1)[0].mean()  # some scalar << for whole batch, choose k_pred with lowest error, then average the error into one distance loss scalar


				### (3) UNCERTAINTY LOSS (original)
				loss_uncertainty = (
					torch.exp(-variance_estimation) *
					(k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
					variance_estimation
				).mean()
				# print(loss_uncertainty)
			
			
			### TOTAL LOSS
			"""2D/3D/6D code continues here"""
			loss = loss_distance * 50 + loss_uncertainty #make distance loss more important than uncertainty loss (?) TODO maybe not this much
			# print(loss)
			# exit()
			loss_total += loss.item()
			loss_dt += loss_distance.item()*50
			loss_dc += loss_uncertainty.item()

			if self.cfg.dimensions == 6:
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


	def _test_single_epoch(self): #for 6D still want to evaluate the trajectory error on the translation part only TODO
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

				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1) #expand GT to match k_preds dimension (B, 1, T, 6)

				pred_traj_trans = pred_traj[..., :3]  # (B, K, T, 3)
				fut_traj_trans = fut_traj[..., :3]      # (B, 1, T, 3)

				distances = torch.norm(fut_traj_trans - pred_traj_trans, dim=-1) * self.traj_scale ## Euclidian translation errors (B, K, T)
				# print('distances: ', distances)

            	# Compute ADE and FDE at different timesteps. TODO improve
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

	def test_single_model(self):
		model_path = './results/checkpoints/2D_model_0069_1040.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		print_log(model_path, log=self.log)

		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)

		### LB Prediction for single trajecotory
		with torch.no_grad():
			past_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_past_traj.npy')).to(self.device)
			traj_mask = torch.from_numpy(np.load('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_traj_mask.npy')).to(self.device)
			fut_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/4_LED_Kitti_6D_Dynamic/visualization/2D_Kitti_KDE_PreTraining/first_fut_traj.npy')).to(self.device)

			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]
			
			k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
			self.old_vis_prior_compute_motion_prior_kde(k_alternative_preds)
			exit()
		# 	priors = self.compute_batch_motion_priors_kde(k_alternative_preds)

		# 	#Visualise 1st traj of first batch first pose
		# 	print(fut_traj.size())
		# 	first_GT_pose = fut_traj[0, 0, :]
		# 	print(first_GT_pose.size())

		# 	first_pose_KDE = priors[0][0] # KDE based on all k_preds for the pose
		# 	first_pose_all_ks = k_alternative_preds[0, :, 0, :]
		# 	# (B, K, T, 2)
		# 	print(first_GT_pose)
		# 	print(first_pose_KDE)
		# 	print(first_pose_all_ks)
		# 	print(first_pose_all_ks.size())

		# 	self.visualise_kpreds_KDE(first_pose_all_ks, first_pose_KDE)
		# 	self.visualise_single_KDE_GT_Past(first_pose_all_ks, first_pose_KDE, past_traj[0, :, :2], first_GT_pose) #first two channels of past are absolute traj, then relative, then velocities
			
		# 	#Probability density
		# 	first_GT_pose_np = first_GT_pose.detach().cpu().numpy().reshape(-1, 1)
		# 	density = first_pose_KDE(first_GT_pose_np)[0]
		# 	print(f"Probability Density of GT final pose: {density}")
		# 	exit()



		### Regular code
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				# print(past_traj.size())
				# print(past_traj[0, :, :])
				# exit()
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				# print(past_traj[0])
				# print(k_alternative_preds[0,0])
				# print(fut_traj[0])
				# exit()
				
				### motion prior
				# priors = self.compute_batch_motion_priors_kde(k_alternative_preds) 	#dictionary keys: traj index within batch (0-31); 
				# 																#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)
				# # (B, K, T, 2)
				# first_GT_pose = fut_traj[10, 0, :]
				# first_pose_KDE = priors[10][0] # KDE based on all k_preds for the pose
				# first_pose_all_ks = k_alternative_preds[10, :, 0, :]
				
				# print(first_GT_pose)
				# print(first_pose_KDE)
				# print(first_pose_all_ks)
				# print(first_pose_all_ks.size())
				
				# self.visualise_single_KDE_GT_Past(first_pose_all_ks, first_pose_KDE, past_traj[10, :, :2], first_GT_pose) #first two channels of past are absolute traj, then relative, then velocities
				
				# #Probability density
				# first_GT_pose_np = first_GT_pose.detach().cpu().numpy().reshape(-1, 1)
				# density = first_pose_KDE(first_GT_pose_np)[0]
				# print(f"Probability Density of GT final pose: {density}")
				# exit()
				###

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
		for time_i in range(4):
			print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		