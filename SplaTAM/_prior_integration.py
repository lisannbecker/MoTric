import torch
from scipy.stats import gaussian_kde

from scipy.optimize import minimize
import numpy as np
import random
# from types import SimpleNamespace

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from _rotation_utils import quaternion_relative

NUM_Tau = 5 #TODO number of iterative steps in reverse diffusion process - higher for more accuracy, lower for more computational efficiency
np.set_printoptions(precision=6, suppress=True)


class PriorIntegration():
    def __init__(self, config):
        """
        Initialise the integration module:
         - Sets device (GPU/CPU)
         - Instantiates the core denoising model and the initialisation model
        """

        print('[INFO] Initialised Motion Prior Integration')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cfg = config

        self.traj_scale = self.cfg.traj_scale

		# ------------------------- define diffusion parameters -------------------------
        self.n_steps = self.cfg.diffusion.steps # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
            start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)


        # ------------------------- initialise models -------------------------
        self.model = CoreDenoisingModel(
            t_h=self.cfg.past_frames,
            d_f=self.cfg.dimensions
        ).to(self.device)
        
        self.model_initializer = InitializationModel(
            t_h=self.cfg.past_frames,
            t_f=self.cfg.future_frames,
            d_f=self.cfg.dimensions,
            k_pred=self.cfg.k_preds
        ).to(self.device)
        

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

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint and restore model.
        """

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_initializer.load_state_dict(checkpoint['model_initializer_state_dict'])
        self.model.load_state_dict(checkpoint['core_denoising_model_state_dict'])

        print(f"[INFO] Checkpoint loaded from {checkpoint_path}")



    def data_preprocess_past_poses(self, prior_config, past_poses, device):
        """
        # self.cfg.dimensions==7 code from LED
        # TUM format: Absolute SE(3) poses: [B, T, 7] (translations + quaternions)

        pre_motion_3D: torch.Size([1, Tpast, 7]), [batch_size, num_agent, past_frame, dimension]
        fut_motion_3D: torch.Size([1, Tfuture, 7])
        fut_motion_mask: torch.Size([1, Tfuture])
        pre_motion_mask: torch.Size([1, Tpast])
        """
        past_frames = prior_config.past_frames
        dimensions = prior_config.dimensions
        traj_scale = prior_config.traj_scale

        batch_size = past_poses.shape[0]
        num_agents = past_poses.shape[1]
        #print(data['pre_motion_3D'][0,0,:,:])


        ### 1.0 Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
        traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
        for i in range(batch_size):
            traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.


        last_obs_pose = past_poses.to(device)[:, :, -1:]  # [B, num_agents, 1, 7]
        
        # translation: subtract last observed translation.
        past_abs = past_poses.to(device)  # [B, num_agents, past_frames, 7]
        past_trans = past_abs[..., :3]               # translations: [B, num_agents, past_frames, 3]
        last_trans  = last_obs_pose[..., :3]           # [B, num_agents, 1, 3]
        rel_trans = (past_trans - last_trans) / traj_scale  # [B, num_agents, past_frames, 3]
        
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


        # relative future trajectory - removed, will be predicted by the model
        # Reshape so that each agent is a separate trajectory - remove agent dimension TODO
        past_traj = past_traj.view(-1, past_frames, past_traj.shape[-1])

        return batch_size, traj_mask, past_traj

    def find_max(self, kde, k_poses):
        # Negate the KDE to turn the maximization into minimization
        
        neg_kde = lambda x: -kde(x)

        # Use the mean of the samples as a starting point
        x0 = np.mean(k_poses.numpy(), axis=0)

        # Minimize the negative density
        res = minimize(neg_kde, x0)

        mode_location = res.x
        mode_density = kde(mode_location)

        print("Mode at:", mode_location)
        print("Max KDE density:", mode_density)
        #print("Some other density:", kde([0.00, 0.00, 0.00]))
        return mode_location

    def kde_metrics(self,kde):
        # gaussian_kde expects shape (D, N), 
        # Kernel density estimation places a smooth "kernel" (Gaussian) at each sample point and sums them to create an overall density estimate
        # Parameter: bandwidth = how smooothly the points are summed. Eg affects whether two close modes merge into one or not
        
        print("Covariance matrix:\n", kde.covariance)
        print("Standard deviation (per dimension):", np.sqrt(np.diag(kde.covariance)))
        # print("Density at point:", kde(point_to_evaluate))


    def extract(self, input, i, past_traj):
        shape = past_traj.shape
        out = torch.gather(input, 0, i.to(input.device))
        reshape = [i.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def p_sample_accelerate(self, past_traj, traj_mask, cur_y, i):
        i = torch.tensor([i]).to(self.device)

        # Factor to the model output
        eps_factor = ((1 - self.extract(self.alphas, i, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, i, cur_y))
       
        # Model output
        beta = self.extract(self.betas, i.repeat(past_traj.shape[0]), cur_y)
        eps_theta = self.model.generate_accelerate(cur_y, beta, past_traj, traj_mask)
        mean = (1 / self.extract(self.alphas, i, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        
        # Generate z
        z = torch.randn_like(cur_y).to(past_traj.device)

        # Fixed sigma
        sigma_t = self.extract(self.betas, i, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001

        return (sample)
    
    def p_sample_loop_accelerate(self, past_traj, traj_mask, initializer_preds): #past_traj, traj_mask, initializer_preds
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
        cur_y = initializer_preds  # use all k predictions
        for i_diffusion_iteration in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(past_traj, traj_mask, cur_y, i_diffusion_iteration)
        return cur_y  # shape: (B, k_pred, T, d)