import sys
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn

from utils.config import Config
from utils.utils import print_log


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #LoaderKitti is two levels up
from LoaderKitti import KITTIDatasetLeapfrog2D, KITTIDatasetLeapfrog3D, seq_collate_kitti


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
			print("NBA dataset (11 agents).")
			dataloader_class = NBADataset
			collate_fn = seq_collate
		elif self.cfg.dataset.lower() == 'kitti':
			dataloader_class = KITTIDatasetLeapfrog3D if self.cfg.dimensions == 3 else KITTIDatasetLeapfrog2D
			collate_fn = seq_collate_kitti
			print("KITTI dataset (1 agent).")


		train_dset = dataloader_class(
			input_size=self.cfg.past_frames,
			preds_size=self.cfg.future_frames,
			training=True
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
			input_size=self.cfg.past_frames,
			preds_size=self.cfg.future_frames,
			training=False
		)
		self.test_loader = DataLoader(
			test_dset,
			batch_size=self.cfg.test_batch_size,
			shuffle=False,
			num_workers=4,
			collate_fn=collate_fn,
			pin_memory=True
		)


		if self.cfg.future_frames < 20:
			print(f"[Warning] Only {self.cfg.future_frames} future timesteps available, "
				f"ADE/FDE will be computed for up to {self.cfg.future_frames // 5} seconds instead of the full 4 seconds.")

		
		if self.cfg.dataset.lower()=='kitti':
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
			print('[INFO] Kitti dataset - skip subtracting mean from absolute positions.')
			
		
		
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
			self.model_initializer = InitializationModel(t_h=self.cfg.past_frames, d_h=self.cfg.dimensions*3, t_f=self.cfg.future_frames, d_f=self.cfg.dimensions, k_pred=self.cfg.k_preds).cuda()

		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		
		# ------------------------- prepare logs -------------------------
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.print_model_param(self.model, name='Core Denoising Model')
		self.print_model_param(self.model_initializer, name='Initialization Model')

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


	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

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
			loss_total, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)
			print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
				epoch, loss_total, loss_distance, loss_uncertainty), self.log)
			

			if (epoch + 1) % self.cfg.test_interval == 0: #TODO have a look here
				performance, samples = self._test_single_epoch()
				for time_i in range(4):
					print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
						time_i+1, performance['ADE'][time_i]/samples,
						time_i+1, performance['FDE'][time_i]/samples), self.log)
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
		batch_size = data['pre_motion_3D'].shape[0]
		num_agents = data['pre_motion_3D'].shape[1]

		#Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
		traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
		for i in range(batch_size):
			traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.

		# print('traj_mask: ', traj_mask.size())
		# Get last observed pose (for each agent) as initial position
		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:] # [B, num_agents, 1, 2] or 3D: [B, num_agents, 1, 3]
		# print('initial_pos:', initial_pos.size()) 

		# augment input: absolute position, relative position, velocity
		if self.cfg.dataset == 'kitti':
			past_traj_abs = (data['pre_motion_3D'].cuda() / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #LB subtracting mean from absolute positions is not informative for kitti dataset
		elif self.cfg.dataset == 'nba':
			past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) 
		
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.future_frames, self.cfg.dimensions)
		#print('fut_traj: ', fut_traj.size())
		return batch_size, traj_mask, past_traj, fut_traj


	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
		#LB 3D addition to reshape tensors 
		
		for data in self.train_loader:
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

			# print('traj_mask:', traj_mask.size()) # [32, 32]
			# print('past_traj:', past_traj.size()) # [32, 15, 9] < [B, Past poses, dimension * 3]
			# print('fut_traj:', fut_traj.size()) # [32, 24, 3] < GT poses for future_frames timesteps


			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
			# print("sample_prediction shape:", sample_prediction.shape)
			# print("mean_estimation shape:", mean_estimation.shape)
			# print("variance_estimation shape:", variance_estimation.shape)
			
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]
			# print('sample_prediction:', sample_prediction.size())
			# print('loc:', loc.size()) #torch.Size([32, 24, 24, 3])
			# exit()
			generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) #k_preds alternative future poses for future_frames timesteps
			
			# print('Predictions batch:', generated_y.size())
			# print('GT batch:', fut_traj.size())

			# print('Prediction 0 shape:', generated_y[0].size())
			# print('GT 0 shape:', fut_traj[0].size())

			# print('Prediction:', generated_y[0])
			# print('GT:', fut_traj[0])


			
			#squared distances / Euclidian, equal weight for all timesteps
			loss_dist = (	(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) 
								* 
							 self.temporal_reweight
						).mean(dim=-1).min(dim=1)[0].mean()
			loss_uncertainty = (torch.exp(-variance_estimation)
		       						*
								(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
									+ 
								variance_estimation
								).mean()
			
			# print(loss_dist)
			# print(loss_uncertainty)
			
			loss = loss_dist*50 + loss_uncertainty
			loss_total += loss.item()
			loss_dt += loss_dist.item()*50
			loss_dc += loss_uncertainty.item()

			self.opt.zero_grad()
			loss.backward()


			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()
			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total/count, loss_dt/count, loss_dc/count


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

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				#print('pred_traj: ', pred_traj.size())


				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				# print('distances: ', distances)
				for time_i in range(1, 5):
					# ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					# fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					max_index = min(5 * time_i - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps
					"""
					1s: 5 * 1 - 1 = 4 → Requires at least 5 timesteps.
					2s: 5 * 2 - 1 = 9 → Requires at least 10 timesteps.
					3s: 5 * 3 - 1 = 14 → Requires at least 15 timesteps.
					4s: 5 * 4 - 1 = 19 → Requires at least 20 timesteps.
					"""
					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum()

					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
				# if count==100:
				# 	break
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
		model_path = './results/checkpoints/led_new.p'
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
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
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
		
	