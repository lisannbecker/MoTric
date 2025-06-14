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
        

    #XXX need?
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


    # ----- Approach 1: Gradient Ascent Correction -----
    def prior_gradient_logl(self, kde, pose, epsilon=1e-5):
        """
        Numerically approximate the gradient of the log likelihood at the given pose.
        pose: numpy array of shape (d,)
        Returns: numpy array of shape (d,) representing the gradient.
        """
        pose = np.array(pose, dtype=float)
        grad = np.zeros_like(pose)
        for i in range(len(pose)):
            delta = np.zeros_like(pose)
            delta[i] = epsilon
            # Evaluate log likelihood at pose + delta and pose - delta.
            logl_plus = np.log(kde(pose + delta))[0]
            logl_minus = np.log(kde(pose - delta))[0]
            grad[i] = (logl_plus - logl_minus) / (2 * epsilon)
        return grad







    def create_KDE(self, checkpoint_path): # <<< adapted version is within the splatam pipeline

        #######  XXX move this
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        self.model_initializer.eval()

        # Ensure reproducibility for testing
        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
        prepare_seed(42)
        #######



        slam_predicted_pose = 'placeholder'
        with torch.no_grad():
            for past_poses in self.XXSLAMPast: ### XXX replace this by whichever method to get past poses
                #batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
                batch_size, traj_mask, past_traj = self.data_preprocess_past_poses(past_poses)

                # Generate initial predictions using the initializer model
                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        
                initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions
                k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds) # Generate the refined trajectory via the diffusion process
                
                # print(k_alternative_preds.size())  # (B, K, T, D)
                # print(past_traj.size()) # (B, 10, D)
                
                # ### ============= motion prior =============
                k_preds_for_kde = k_alternative_preds[0, :, 0, :] # B is one anyway and and T should be the next position
                prior = gaussian_kde(k_preds_for_kde.T)
                #priors = self.compute_batch_motion_priors_kde(k_alternative_preds) #dictionary keys: traj index within batch (0-31); 
                 	 																#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)

                prior_corrected_pose = self.correct_pose_gradient(prior, slam_predicted_pose, lambda_step=0.1)
                print("Corrected SLAM pose (Gradient Ascent):", prior_corrected_pose)









# config_dict = {
#     "traj_scale": 5.0,
#     "past_frames": 10,
#     "future_frames": 20,
#     "dimensions": 9,
#     "k_preds": 50,
#     "diffusion": {
#         "steps": 100,
#         "beta_schedule": "linear",
#         "beta_start": 1e-5,
#         "beta_end": 1e-2,
#     },
#     "relative": True,
#     "normalised": False,
#     "overfitting": False,
#     "selected_trajectories": None,
# }

# def dict_to_namespace(d):
#     return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})



# config = dict_to_namespace(config_dict)
# base_prior_model = './results/6_Testing_9D_newer/9D_newer/models/best_checkpoint_epoch_51.pth'


# SLAMwithPrior = PriorIntegration(config)
# SLAMwithPrior.create_KDE(base_prior_model)





"""
def data_preprocess_single_agent(self, data): #Updated to handle any number of agents TODO adapt

    # pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
    # fut_motion_3D: torch.Size([32, 11, 20, 2])
    # fut_motion_mask: torch.Size([32, 11, 20])
    # pre_motion_mask: torch.Size([32, 11, 10])
    # traj_scale: 1
    # pred_mask: None
    # seq: nba

    pre_motion = data['pre_motion_3D'].to(self.device)
    fut_motion = data['fut_motion_3D'].to(self.device)

    batch_size = pre_motion.shape[0]

    #Create trajectory mask 
    traj_mask = torch.eye(batch_size).to(self.device)

    # Get last observed pose (for each agent) as initial position << both translation and rotation
    # Shape: [B, 1, D]
    initial_pos = pre_motion[:, -1, :]

    # augment input: absolute position, relative position, velocity TODO augmentation is applied to whole input, not only translation
    past_traj_abs = (pre_motion/ self.traj_scale).contiguous() # [B, past_frames, D]
    past_traj_rel = ((pre_motion - initial_pos)/self.traj_scale).contiguous()# only relativises if initial pos is not 0 already (relative = True) [B, past_frames, D]
    past_traj_vel = torch.cat(
                (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], 
                torch.zeros_like(past_traj_rel[:, -1:]))
                , dim=1) #[B, past_frames, D]

    past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1) # [B, past_frames, D * 3]

    fut_traj = ((fut_motion - initial_pos)/self.traj_scale).contiguous() #relativises (if not done already) [B, future_frames, D]

    return batch_size, traj_mask, past_traj, fut_traj






def compute_batch_motion_priors_kde(self, k_preds):#consider that we only visualise xy TODO
    
    # Computes a KDE-based motion prior for each sample and each future time step.

    # Limitation: Using per-timestep KDE densities assumes that the predictions at each timestep are independent, but they aren’t.
    
    # Args:
    #     k_preds_np (np.array): Predicted trajectories of shape (B, K, T, Dimension).
    #                         B: batch size, K: number of predictions, 
    #                         T: future timesteps, 6: pose dimensions.
    
    # Returns:
    #     priors: A list of lists such that priors[b][t] is a KDE object for sample b at time step t.
    
    k_preds_np = k_preds.detach().cpu().numpy()

    B, K, T, D = k_preds_np.shape

    priors = {}
    for trajectory_idx in range(B):
        
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

"""