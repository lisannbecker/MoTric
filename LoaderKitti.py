import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import os
import sys
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

posenet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Continuous-Pose-in-NeRF"))
sys.path.append(posenet_dir)
from PoseNet import PoseNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#TODO I think kitti translation is
    # [... x] <<left-right
    # [... z] <<up/down
    # [... y] <<forward
    
    # do some analyses ie calculate variance of each dim

### Only one can be True
Danda = False
Fully_Custom = False
LED2D = True
LED3D = False
LED6D = False




class KITTIDatasetLeapfrog(Dataset): #3D translation and 3D Lie algebra for rotation

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline.
    NOTE: Loads x z y translation (original LED) and Dual Quaternions: full SE3 pose.
    """

    def __init__(self, dims: int, input_size: int, preds_size: int, training: bool, final_eval: bool, relative=False, normalised=False, train_ratio=0.80, eval_ratio=0.10, seed=42, overlapping=False):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """            
        window_size = input_size+preds_size

        print('Overlapping:', overlapping)

        traj_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] #uncomment

        ### Load and split data from all KITTI trajectories
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size, relative, normalised, overlapping) 
        # p_inputs: [num_windows, input_size, 3, 4] p_targets: [num_windows, output_size, 3, 4]

        # print(f'Time diff avg (fut): {(t_targets[:,-1]-t_targets[:,0]).mean(dim=-1).item():.5f}')


        ### 1.1. 2D: Extract x, y translation
        if dims == 2:
            trans_pre = p_inputs[:, :, [0,2], 3]  # [num_windows, input_size, 2] << 2D 
            trans_fut = p_targets[:, :, [0,2], 3]  # [num_windows, output_size, 2] << 2D 
        ### 1.1. 3D/6D: Extract x, z, y translation
        else:
            trans_pre = p_inputs[:, :, :3, 3]  # [num_windows, input_size, 3] << 3D adjustment
            trans_fut = p_targets[:, :, :3, 3]  # [num_windows, output_size, 3] << 3D adjustment
            # print(trans_pre[0], '\n', trans_fut[0])


        ### 1.2. 6D: Extract rotation and convert to Lie algebra (and make relative)
        if dims ==6:
            self.lie = Lie()

            if relative:
                rotations_pre, rotations_fut = to_relative_rotations(p_inputs, p_targets, input_size, self.lie)

                # Now convert the relative rotation matrices into Lie algebra vectors
                num_windows, seq_len_pre, _, _ = rotations_pre.shape
                num_windows_fut, seq_len_fut, _, _ = rotations_fut.shape
                # lie_pre:  [num_windows, input_size, 3]
                # lie_fut:  [num_windows, output_size, 3]
            else:
                #Convert absolute rotations (for each time step) to Lie algebra
                rotations_pre = p_inputs[:, :, :3, :3]  # [num_windows, input_size, 3, 3]
                rotations_fut = p_targets[:, :, :3, :3]  # [num_windows, output_size, 3, 3]

                num_windows, seq_len_pre = rotations_pre.shape[:2]
                num_windows_fut, seq_len_fut = rotations_fut.shape[:2]


            rotations_pre_flat = rotations_pre.reshape(-1, 3, 3)
            lie_pre_flat = self.lie.SO3_to_so3(rotations_pre_flat)  # [N, 3]
            lie_pre = lie_pre_flat.reshape(num_windows, seq_len_pre, 3)

            rotations_fut_flat = rotations_fut.reshape(-1, 3, 3)
            lie_fut_flat = self.lie.SO3_to_so3(rotations_fut_flat)
            lie_fut = lie_fut_flat.reshape(num_windows_fut, seq_len_fut, 3)


            ### 2. Concatenate translations and rotation and add agent dimension (1 for single-agent Kitti)
            SE3_pre = torch.cat([trans_pre, lie_pre], dim=-1)
            SE3_fut = torch.cat([trans_fut, lie_fut], dim=-1)

        else: #2D or 3D
            SE3_pre = trans_pre
            SE3_fut = trans_fut

        ### 3. Add agent dimension
        all_pre = SE3_pre.unsqueeze(1)  # [num_windows, 1, input_size, 6]
        all_fut = SE3_fut.unsqueeze(1)  # [num_windows, 1, output_size, 6]


        ### 4. Create masks (all 1s) matching the temporal dimensions
        all_pre_mask = torch.ones(all_pre.shape[0], 1, all_pre.shape[2])
        all_fut_mask = torch.ones(all_fut.shape[0], 1, all_fut.shape[2])


        # 5. Randomly split the windows using a fixed seed 
        total_windows = all_pre.shape[0]
        indices = list(range(total_windows))
        random.seed(seed)
        random.shuffle(indices)


        split_idx_train = int(total_windows * train_ratio)
        split_idx_validation = int(total_windows * (train_ratio+eval_ratio))

        if training:
            selected_indices = indices[:split_idx_train]
        elif final_eval: #final testing
            selected_indices = indices[split_idx_validation:]
        else: #validation
            selected_indices = indices[split_idx_train:split_idx_validation]

        # Subset data
        self.pre_motion_3D = all_pre[selected_indices]
        self.fut_motion_3D = all_fut[selected_indices]
        self.pre_motion_mask = all_pre_mask[selected_indices]
        self.fut_motion_mask = all_fut_mask[selected_indices]



    def __len__(self):
        return self.pre_motion_3D.shape[0]

    def __getitem__(self, idx):
        sample = {
            'pre_motion_3D': self.pre_motion_3D[idx],  # [1, input_size, 3, 4]
            'fut_motion_3D': self.fut_motion_3D[idx],    # [1, target_length, 3, 4]
            # 'pre_motion_lie': self.pre_motion_lie[idx],    # [1, input_size, 3] Lie algebra representation of rots
            # 'fut_motion_lie': self.fut_motion_lie[idx],    # [1, output_size, 3]
            'pre_motion_mask': self.pre_motion_mask[idx],  # [1, input_size]
            'fut_motion_mask': self.fut_motion_mask[idx],  # [1, target_length]
        }

        return sample



def seq_collate_kitti(data): #allows to have None as pred_mask (as opposed to default collate)
    pre_motion_3D = torch.stack([d['pre_motion_3D'] for d in data], dim=0)
    fut_motion_3D = torch.stack([d['fut_motion_3D'] for d in data], dim=0)
    pre_motion_mask = torch.stack([d['pre_motion_mask'] for d in data], dim=0)
    fut_motion_mask = torch.stack([d['fut_motion_mask'] for d in data], dim=0)

    # additional keys to match LED's output.
    return {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'pre_motion_mask': pre_motion_mask,
        'fut_motion_mask': fut_motion_mask,
        'traj_scale': 1, # no scaling
        'pred_mask': None,
        'seq': 'kitti'
    }

class KITTIDatasetCustom(Dataset): # corresponds to class NBADataset in dataloader_nba.py

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline

    NOTE Loads full SE3 pose
    """

    def __init__(self, traj_list, input_size, preds_size, use_relative = False, use_normalised = False):
        """
        traj_list (list of str): indices of kitti trajectories to be loaded
        window size (int): total window size to split the kitti trajectories into
        input_size (int): number of poses used as input (past trajectory)
        """


        window_size = input_size+preds_size

        ### Load and split data from all KITTI trajectories
        self.p_inputs, self.p_targets, self.t_inputs, self.t_targets = load_all_and_split(traj_list, window_size, input_size, use_relative, use_normalised)
            # p_inputs: [num_windows, input_size, 3, 4]
            # p_targets: [num_windows, window_size - input_size, 3, 4]


        ### Create masks (all 1s) matching the temporal dimensions
        # self.pre_motion_mask = torch.ones(self.pre_motion_3D.shape[0], 1, self.pre_motion_3D.shape[2])
        # self.fut_motion_mask = torch.ones(self.fut_motion_3D.shape[0], 1, self.fut_motion_3D.shape[2])


    def __len__(self):
        return self.p_inputs.size()[0]

    def __getitem__(self, idx):
        sample = {
            'p_inputs_SE3': self.p_inputs[idx],  # [1, input_size, 3, 4]
            'p_targets_SE3': self.p_targets[idx],    # [1, target_length, 3, 4]
            't_inputs_SE3': self.t_inputs[idx],  # [1, input_size]
            't_targets_SE3': self.t_targets[idx],    # [1, target_length]
            
            # TODO need this?
            # 'pre_motion_mask': self.pre_motion_mask[idx],  # [1, input_size]
            # 'fut_motion_mask': self.fut_motion_mask[idx],  # [1, target_length]
            # 'seq': 'kitti'
        }

        return sample


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

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
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans



###Not using this rn as I'm unsure how to retain the original LED translation pipeline then
def quaternion_from_matrix(R_mat):
    """
    Convert 3x3 rotation matrix to a quaternion in [w, x, y, z] format.
    SciPy’s Rotation.as_quat() returns [x, y, z, w], so we roll it.
    """
    rot = R.from_matrix(R_mat)
    q = rot.as_quat()  # [x, y, z, w]
    q = np.roll(q, 1)  # convert to [w, x, y, z] = move w to front
    return q

def quaternion_multiply(q, r):
    """
    Multiply two quaternions.
    Assumes quaternions are in [w, x, y, z] order.
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def rigid_to_dual_quaternion(R_mat, t):
    """
    Convert a rigid-body transform (R_mat: 3x3 rotation, t: 3D translation) 
    to an 8D dual quaternion representation.
    
    The dual quaternion is represented as [q_r, q_d]:
      - q_r: rotation quaternion (unit, [w,x,y,z])
      - q_d: 0.5 * (0,t) * q_r
    """
    # Get the rotation quaternion (make sure it’s normalized)
    q_r = quaternion_from_matrix(R_mat)
    q_r = q_r / np.linalg.norm(q_r)
    
    # Represent translation as a pure quaternion: [0, t_x, t_y, t_z]
    t_quat = np.concatenate(([0.0], t))
    # Compute the dual part: q_d = 0.5 * (t_quat * q_r)
    q_d = 0.5 * quaternion_multiply(t_quat, q_r)
    
    # Concatenate to get an 8D dual quaternion: [q_r, q_d]
    dq = np.concatenate((q_r, q_d))
    return dq



def to_relative(pose_windows, time_windows, input_size):
    rotations = pose_windows[..., :3]
    translations = pose_windows[..., 3]
    
    last_idx = input_size-1
    last_translation = translations[:, last_idx, :].unsqueeze(1) #[num_windows, 1, 3] Get translation at the last input time step

    translations_relative = translations-last_translation   # subtract last pose of past trajectory / current pose from all poses
                                                            # future will be pos, past neg

    #same for times
    last_times = time_windows[:, last_idx].unsqueeze(1) #[num_windows, 1]
    time_windows_relative = time_windows-last_times

    #concat into one matrix again
    translations_relative = translations_relative.unsqueeze(-1)
    pose_windows_relative = torch.cat([rotations, translations_relative], dim=-1)
    return pose_windows_relative, time_windows_relative

def to_relative_rotations(past_pose_windows, fut_pose_windows, input_size, lie_converter):
    """
    Computes relative rotations for both past and future pose windows relative to the 
    most recent rotation in the past (current rotation).

    Args:
        past_pose_windows: Tensor of shape [num_windows, input_size, 3, 4] 
                           (each 3x4 contains a 3x3 rotation and a translation)
        fut_pose_windows:  Tensor of shape [num_windows, output_size, 3, 4]
        input_size: Number of past poses.
        lie_converter: An instance (or function) that converts a 3x3 rotation matrix to a 3D Lie algebra vector.
    
    Returns:
        lie_past: Relative rotations for the past, as Lie algebra vectors, 
                  shape [num_windows, input_size, 3]
        lie_fut:  Relative rotations for the future, as Lie algebra vectors,
                  shape [num_windows, output_size, 3]
    """
    # Extract rotations (first 3x3 block) from the 3x4 pose matrices
    rotations_past = past_pose_windows[..., :3, :3]  # [num_windows, input_size, 3, 3]
    rotations_fut = fut_pose_windows[..., :3, :3]    # [num_windows, output_size, 3, 3]
    
    # Get the reference rotation: last rotation from the past sequence
    # Shape: [num_windows, 3, 3]
    R_ref = rotations_past[:, input_size - 1]
    
    # Compute the transpose of R_ref (for each window)
    R_ref_T = R_ref.transpose(-2, -1)  # [num_windows, 3, 3]
    
    # For past: compute relative rotations: R_rel = R_ref^T @ R
    # Using broadcasting: (R_ref_T is [num_windows, 3, 3]) and rotations_past is [num_windows, input_size, 3, 3]
    # We add an extra dimension so that multiplication works along the time dimension.
    rel_rotations_past = torch.matmul(R_ref_T.unsqueeze(1), rotations_past)
    # For future: similarly, compute relative rotations for future poses.
    rel_rotations_fut = torch.matmul(R_ref_T.unsqueeze(1), rotations_fut)

    # print(rel_rotations_past[0])
    # print(rel_rotations_fut[0])

    return rel_rotations_past, rel_rotations_fut

def to_normalised(pose_windows, time_windows, input_size):
    """Normalise times and translations but not rotations"""
    t_min = time_windows.min(dim=1, keepdim=True)[0] #(minimum_value, minimum_value_index)
    t_max = time_windows.max(dim=1, keepdim=True)[0]

    time_windows = 2 * (time_windows - t_min) / (t_max - t_min) - 1


    translations_only = pose_windows[:,:,:3,3]

    # print(translations_only.size()) #[4529, 13, 3]
    trans_min = translations_only.min(dim=1, keepdim=True)[0] #takes minimum independently for each coordinate - pose invariant
    trans_max = translations_only.max(dim=1, keepdim=True)[0]

    translations_normalised = 2 * (translations_only - trans_min) / (trans_max - trans_min) - 1
    pose_windows[:,:,:3,3] = translations_normalised
    
    return pose_windows, time_windows



def load_raw_traj_poses(traj_idx):
    file_path = f'/home/scur2440/MoTric/KITTI_odometry/dataset/poses/{traj_idx}.txt'
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split())) #list for line
            #Full SE3 matrix
            # SE3 = np.zeros((4, 4))
            # SE3[:3, :] = np.array(values).reshape(3,4)
            # SE3[3, 3] = 1

            # Danda SE3 implementation
            SE3 = torch.tensor(values).reshape(3,4) #SE3 matrix without bottom row 0,0,0,1
            poses.append(SE3)
    return torch.stack(poses)

def load_raw_traj_times(traj_idx):
    file_path = f'/home/scur2440/MoTric/KITTI_odometry/dataset/sequences/{traj_idx}/times.txt'
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            times.append(float(line.strip()))
    return torch.tensor(times)



def load_one_and_split(traj, window_size, input_size, use_relative = False, use_normalised = False, overlapping = False): #single trajectory - load and split
    """
    Loads a single trajectory’s poses and timestamps, then splits them into overlapping windows of size window_size.
    """
    poses = load_raw_traj_poses(traj)
    times = load_raw_traj_times(traj)

    if overlapping == False:
        num_windows = poses.shape[0] // window_size 
        pose_windows = torch.stack([poses[i*window_size:(i+1)*window_size, :, :] for i in range(num_windows)]) #save in list and then stack in tensor [num_windows, window_size, 3, 4]
        time_windows = torch.stack([times[i*window_size:(i+1)*window_size] for i in range(num_windows)]) # [num_windows, window_size]
    else:
        num_windows = poses.shape[0] - window_size +1
        pose_windows = torch.stack([poses[i:i+window_size, :, :] for i in range(num_windows)]) #save in list and then stack in tensor [num_windows, window_size, 3, 4]
        time_windows = torch.stack([times[i:i+window_size] for i in range(num_windows)]) # [num_windows, window_size]

    if Danda == True:
        return pose_windows, time_windows # Split into subtrajectories but not input and target


    # print(pose_windows[0])
    if use_relative:
        pose_windows, time_windows = to_relative(pose_windows, time_windows, input_size)

    if use_normalised:
        pose_windows, time_windows = to_normalised(pose_windows, time_windows, input_size)

    #split into input and target trajectory
    input_poses = pose_windows[:, :input_size, :, :]  # [num_windows, input_size, 3, 4]
    target_poses = pose_windows[:, input_size:, :, :]  # [num_windows, window_size-input_size, 3, 4]

    input_times = time_windows[:, :input_size]  # [num_windows, input_size]
    target_times = time_windows[:, input_size:]  # [num_windows, window_size-input_size]

    return input_poses, target_poses, input_times, target_times  # pose windows: [num_windows, window_size, 3, 4]    (SE=3,4)
                                      # For example: [4532, 10, 3, 4]
                                      # time windows: [num_windows, window_size]

def load_all_and_split(traj_list, window_size, input_size, use_relative = False, use_normalised = False, overlapping = False):
    """
    Returns combines pose and time windows of size window_size for all trajectories.
    
    Data from different trajectories is kept separate.
    """
    p_input_all, p_target_all, t_input_all, t_target_all = [],[],[],[]

    for traj in traj_list:
        input_poses, target_poses, input_times, target_times = load_one_and_split(traj, window_size, input_size, use_relative, use_normalised, overlapping)
        
        p_input_all.append(input_poses)
        p_target_all.append(target_poses)
        t_input_all.append(input_times)
        t_target_all.append(target_times)

    combined_p_inputs = torch.cat(p_input_all)
    combined_p_targets = torch.cat(p_target_all)
    combined_t_inputs = torch.cat(t_input_all)
    combined_t_targets = torch.cat(t_target_all)
    return combined_p_inputs, combined_p_targets, combined_t_inputs, combined_t_targets

def train_custom(p_wins, t_wins, N=3, batch_size = 32):
    """
    Window has 13 poses (right now) - first 10 poses would be used as input (mask) and last 3 should be predicted
    """

    print(p_wins.size())
    for subtrajectory in p_wins:
        input_poses = subtrajectory[:-N] #mask poses
        gt_poses_for_loss = subtrajectory[-N:]
        
        print(subtrajectory.size())
        print(input_poses.size())
        print(gt_poses_for_loss.size())
        
        exit()
    
    
    ### TODO switch to batch training
    # dataset = TensorDataset(p_wins)  #TensorDataset(p_wins, t_wins)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # #validation_dataloader = 
    # #test_dataloader = 

    # for batch_i, poses in enumerate(train_dataloader):
    #     print(len(poses)) #list of tensors
    #     print(poses)
    #     exit()


def print_some_stats(future, future_rot=None, translation_dims=3):
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



if __name__ == "__main__":

    if Danda == True:
        traj_idx = '00'
        p_wins, t_wins = load_one_and_split(traj_idx, 13, 10) # pose windows: N windows, window size, SE3 rows, SE3 columns] , time windows: [N windows, window size]
        train_danda(p_wins, t_wins, 32, True)
    
    elif Fully_Custom ==True:
        train_list = ['00', '01', '02', '03', '04', '05', '06'] 
        val_list   = ['07', '10'] 
        test_list  = ['08', '09']

        train_dataset = KITTIDatasetCustom(train_list, 10, 3, True, False) # kitti trajectories to load, input size (n past trajectories), target size (n trajectories to predict), use relative poses/times, use normalised translations/times
        val_dataset = KITTIDatasetCustom(val_list, 10, 3,  True, False) 
        test_dataset = KITTIDatasetCustom(test_list, 10, 3,  True, False) 

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4) #only one trajectory for validation? Bias? TODO
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

        #print(len(train_loader), len(val_loader), len(test_loader)) #Nr of batches: 474 72 177

        for batch in train_loader:
            print(batch.keys())
            print("Batch pre-motion shape:", batch['p_inputs_SE3'].shape)  # [B, 10, 3, 4]
            print("Batch future motion shape:", batch['p_targets_SE3'].shape)  # [B, 3, 3, 4]

            # print(batch['p_inputs_SE3'][0])
            # print(batch['p_targets_SE3'][0])  
            # print(batch['t_inputs_SE3'][0])
            # print(batch['t_targets_SE3'][0])

            exit()
            break

    elif LED2D == True:
        train_dataset = KITTIDatasetLeapfrog(dims = 2, input_size=10, preds_size=20, training=True, relative=True, normalised=False, overlapping=True) #which trajectories to load, window size, out of which past trajectories (rest is target trajectories), no normalisation [WIP]
        print_some_stats(train_dataset.fut_motion_3D, None, 2)
        print('Train trajectories:', len(train_dataset)) # 16106 for input_size=10, preds_size=20
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

        test_dataset = KITTIDatasetLeapfrog(dims=2, input_size=10, preds_size=20, training=False, relative=True, normalised=False, overlapping=True)
        print_some_stats(train_dataset.fut_motion_3D, None, 2)
        print('Test trajectories:', len(test_dataset)) # 6776 for input_size=10, preds_size=20
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

    elif LED3D == True:
        train_dataset = KITTIDatasetLeapfrog(dims=3, input_size=10, preds_size=20, training=True, relative=True, normalised=False) #which trajectories to load, window size, out of which past trajectories (rest is target trajectories), no normalisation [WIP]
        print_some_stats(train_dataset.fut_motion_3D)
        print('Train trajectories:', len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

        test_dataset = KITTIDatasetLeapfrog(dims=3, input_size=10, preds_size=20, training=False, relative=True, normalised=False)
        print_some_stats(test_dataset.fut_motion_3D)
        print('Test trajectories:', len(test_dataset)) 
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

    elif LED6D == True:

        train_dataset = KITTIDatasetLeapfrog(dims=6, input_size=10, preds_size=20, training=True, relative=True, normalised=False) #which trajectories to load, window size, out of which past trajectories (rest is target trajectories), no normalisation [WIP]
        print('Train trajectories:', len(train_dataset)) 
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)
        print_some_stats(train_dataset.fut_motion_3D[..., :3], train_dataset.fut_motion_3D[..., 3:], 3)

        test_dataset = KITTIDatasetLeapfrog(dims=6, input_size=10, preds_size=20, training=False, relative=True, normalised=False)
        print('Test trajectories:', len(test_dataset)) 
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)
        print_some_stats(test_dataset.fut_motion_3D[..., :3], test_dataset.fut_motion_3D[..., 3:], 3)

    for batch in train_loader:
        print(batch.keys())
        print("\nBatch pre-motion shape:", batch['pre_motion_3D'].shape)  # [batch_size, 1, past_poses, 3]
        print("Batch future motion shape:", batch['fut_motion_3D'].shape)  # [batch_size, 1, future_poses, 3]
        print("Batch pre-motion mask shape:", batch['pre_motion_mask'].shape)  # [batch_size, 1, past_poses, 3]
        print("Batch future motion mask shape:", batch['fut_motion_mask'].shape)  # [batch_size, 1, future_poses, 3]
        print("traj_scale:", batch['traj_scale'])
        print("pred_mask:", batch['pred_mask'])
        print("seq:", batch['seq'])
        exit()
        break





"""
Trajectory 00 Length: 4541
Trajectory 01 Length: 1101
Trajectory 02 Length: 4661
Trajectory 03 Length: 801
Trajectory 04 Length: 271
Trajectory 05 Length: 2761
Trajectory 06 Length: 1101
Trajectory 07 Length: 1101
Trajectory 08 Length: 4071
Trajectory 09 Length: 1591
Trajectory 10 Length: 1201

Total Length: 23201
Total N of Windows: 23102

print(f'Trajectory {traj} Length: {times.size()[0]}')
"""



"""
code dump


def flatten_z(p_wins):
    p_wins[:, :, 2, 3] = 0 #FIXME assuming that z (3rd row, 4th column) contains the "up/down" coordinate
    return p_wins



def load_one_and_split(traj, window_size = 100): #single trajectory - load and split

    poses = load_traj_poses(traj)
    times = load_traj_times(traj)

    #make data divisible by 100
    remainder = poses.size()[0] % 100 #100 is Danda authors number of timesteps
    rounded_size = poses.size()[0]-remainder
    poses = poses[:rounded_size, :, :]
    times = times[:rounded_size]
    print(poses.size())
    print(times.size())

    pchunks = torch.split(poses, 100, dim=0)
    pchunks = torch.split(poses, 100)
    print(pchunks[0].shape)
    print(len(pchunks)) 
    tchunks = torch.split(times, 100)
    print(tchunks[0].shape)

"""

class KITTIDatasetLeapfrog2D(Dataset): # corresponds to class NBADataset in dataloader_nba.py

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline.

    NOTE: Loads x y translation (original LED) only instead of full SE3 pose.
    """

    def __init__(self, input_size, preds_size, training, relative=False, normalised=False):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """

        #TODO fixed seed random instead of fixed sequences for train and test
        window_size = input_size+preds_size
        if training ==True:
            traj_list = ['00', '01', '02', '03', '04', '05', '06', '07']
        else:
            traj_list = ['08', '09', '10']

        ### Load and split data from all KITTI trajectories
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size, relative, normalised)
            # p_inputs: [num_windows, input_size, 3, 4]
            # p_targets: [num_windows, window_size - input_size, 3, 4]


        ### 1. Extract only x and y translation
        self.pre_motion_2D = p_inputs[:, :, :2, 3]  # [num_windows, input_size, 2]
        self.fut_motion_2D = p_targets[:, :, :2, 3]  # [num_windows, window_size - input_size, 2]


        self.pre_motion_2D = p_inputs[:,:,[0,2],3] 
        self.fut_motion_2D = p_targets[:, :, [0,2], 3]

        ### 2. LED pipeline uses an extra dimension at index 1 for the number of agents
        # We add this extra dimension and set it to 1 as Kitti is single-agent
        self.pre_motion_3D = self.pre_motion_2D.unsqueeze(1) #[num_windows, 1, input_size, 2]
        self.fut_motion_3D = self.fut_motion_2D.unsqueeze(1) #[num_windows, 1, (window_size - input_size), 2]

        ### 3. Create masks (all 1s) matching the temporal dimensions
        self.pre_motion_mask = torch.ones(self.pre_motion_3D.shape[0], 1, self.pre_motion_3D.shape[2])
        self.fut_motion_mask = torch.ones(self.fut_motion_3D.shape[0], 1, self.fut_motion_3D.shape[2])


    def __len__(self):
        return self.pre_motion_3D.shape[0]

    def __getitem__(self, idx):
        sample = {
            'pre_motion_3D': self.pre_motion_3D[idx],  # [1, input_size, 3, 4]
            'fut_motion_3D': self.fut_motion_3D[idx],    # [1, target_length, 3, 4]
            'pre_motion_mask': self.pre_motion_mask[idx],  # [1, input_size]
            'fut_motion_mask': self.fut_motion_mask[idx],  # [1, target_length]
        }

        return sample

class KITTIDatasetLeapfrog3D(Dataset): #Similar to class NBADataset in dataloader_nba.py

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline.
    NOTE: Loads x z y translation instead of full SE3 pose.
    """

    def __init__(self, input_size, preds_size, training, transform=None):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """

        #TODO fixed seed random instead of fixed sequences for train and test
        window_size = input_size+preds_size
        if training ==True:
            traj_list = ['00', '01', '02', '03', '04', '05', '06', '07']
        else:
            traj_list = ['08', '09', '10']

        ### Load and split data from all KITTI trajectories
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size)
            # p_inputs: [num_windows, input_size, 3, 4]
            # p_targets: [num_windows, output_size, 3, 4]

        ### 1. Extract x, z, y translation
        self.pre_motion_2D = p_inputs[:, :, :3, 3]  # [num_windows, input_size, 3] << 3D adjustment
        self.fut_motion_2D = p_targets[:, :, :3, 3]  # [num_windows, output_size, 3] << 3D adjustment

        ### 2. LED pipeline uses an extra dimension at index 1 for the number of agents
        # We add this extra dimension and set it to 1 as Kitti is single-agent
        self.pre_motion_3D = self.pre_motion_2D.unsqueeze(1) #[num_windows, 1, input_size, 3]
        self.fut_motion_3D = self.fut_motion_2D.unsqueeze(1) #[num_windows, 1, output_size, 3]
        
        

        # print(self.pre_motion_3D.size())
        # print(self.fut_motion_3D.size())

        ### 3. Create masks (all 1s) matching the temporal dimensions
        self.pre_motion_mask = torch.ones(self.pre_motion_3D.shape[0], 1, self.pre_motion_3D.shape[2])
        self.fut_motion_mask = torch.ones(self.fut_motion_3D.shape[0], 1, self.fut_motion_3D.shape[2])

        self.transform = transform #TODO use in the future for relative/normalised trajectories

    def __len__(self):
        return self.pre_motion_3D.shape[0]

    def __getitem__(self, idx):
        sample = {
            'pre_motion_3D': self.pre_motion_3D[idx],  # [1, input_size, 3, 4]
            'fut_motion_3D': self.fut_motion_3D[idx],    # [1, target_length, 3, 4]
            'pre_motion_mask': self.pre_motion_mask[idx],  # [1, input_size]
            'fut_motion_mask': self.fut_motion_mask[idx],  # [1, target_length]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

class KITTIDatasetLeapfrog6D(Dataset): #3D translation and 3D Lie algebra for rotation

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline.
    NOTE: Loads x z y translation (original LED) and Dual Quaternions: full SE3 pose.
    """

    def __init__(self, input_size, preds_size, training, relative=False, normalised=False, train_ratio=0.85, seed=42):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """

        self.lie = Lie()
        window_size = input_size+preds_size

        # if training ==True: # uncomment to deactivate shuffling
        #     traj_list = ['00', '01', '02', '03', '04', '05', '06', '07']
        # else:
        #     traj_list = ['08', '09', '10']

        traj_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] #uncomment

        ### Load and split data from all KITTI trajectories
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size, relative, normalised) # p_inputs: [num_windows, input_size, 3, 4] p_targets: [num_windows, output_size, 3, 4]
        # print(f'Time diff avg (fut): {(t_targets[:,-1]-t_targets[:,0]).mean(dim=-1).item():.5f}')



        ### 1.1. Extract x, z, y translation
        trans_pre = p_inputs[:, :, :3, 3]  # [num_windows, input_size, 3] << 3D adjustment
        trans_fut = p_targets[:, :, :3, 3]  # [num_windows, output_size, 3] << 3D adjustment
        # print(trans_pre[0], '\n', trans_fut[0])


        ### 1.2. Extract rotation and convert to Lie algebra (and make relative)
        if relative:
            rotations_pre, rotations_fut = to_relative_rotations(p_inputs, p_targets, input_size, self.lie)

            # Now convert the relative rotation matrices into Lie algebra vectors.
            num_windows, seq_len_pre, _, _ = rotations_pre.shape
            num_windows_fut, seq_len_fut, _, _ = rotations_fut.shape
            # lie_pre:  [num_windows, input_size, 3]
            # lie_fut:  [num_windows, output_size, 3]
        else:
            #Convert absolute rotations (for each time step) to Lie algebra.
            rotations_pre = p_inputs[:, :, :3, :3]  # [num_windows, input_size, 3, 3]
            rotations_fut = p_targets[:, :, :3, :3]  # [num_windows, output_size, 3, 3]

            num_windows, seq_len_pre = rotations_pre.shape[:2]
            num_windows_fut, seq_len_fut = rotations_fut.shape[:2]


        rotations_pre_flat = rotations_pre.reshape(-1, 3, 3)
        lie_pre_flat = self.lie.SO3_to_so3(rotations_pre_flat)  # [N, 3]
        lie_pre = lie_pre_flat.reshape(num_windows, seq_len_pre, 3)

        rotations_fut_flat = rotations_fut.reshape(-1, 3, 3)
        lie_fut_flat = self.lie.SO3_to_so3(rotations_fut_flat)
        lie_fut = lie_fut_flat.reshape(num_windows_fut, seq_len_fut, 3)

        # print(lie_pre[0])
        # print(lie_fut[0])


        ### 2. Concatenate translations and rotation and add agent dimension (1 for single-agent Kitti)
        SE3_pre = torch.cat([trans_pre, lie_pre], dim=-1)
        SE3_fut = torch.cat([trans_fut, lie_fut], dim=-1)
        # print(SE3_pre[0], '\n', SE3_fut[0])


        ### 3. Add agent dimension
        all_pre = SE3_pre.unsqueeze(1)  # [num_windows, 1, input_size, 6]
        all_fut = SE3_fut.unsqueeze(1)  # [num_windows, 1, output_size, 6]


        ### 4. Create masks (all 1s) matching the temporal dimensions
        all_pre_mask = torch.ones(all_pre.shape[0], 1, all_pre.shape[2])
        all_fut_mask = torch.ones(all_fut.shape[0], 1, all_fut.shape[2])


        # 5. Randomly split the windows using a fixed seed 
        # selected_indices = list(range(all_pre.shape[0])) # uncomment to deactivate shuffling
        total_windows = all_pre.shape[0]
        # print(total_windows)
        indices = list(range(total_windows))
        random.seed(seed)
        random.shuffle(indices)
        split_idx = int(total_windows * train_ratio)

        if training:
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        # Subset data
        self.pre_motion_3D = all_pre[selected_indices]
        self.fut_motion_3D = all_fut[selected_indices]
        self.pre_motion_mask = all_pre_mask[selected_indices]
        self.fut_motion_mask = all_fut_mask[selected_indices]



    def __len__(self):
        return self.pre_motion_3D.shape[0]

    def __getitem__(self, idx):
        sample = {
            'pre_motion_3D': self.pre_motion_3D[idx],  # [1, input_size, 3, 4]
            'fut_motion_3D': self.fut_motion_3D[idx],    # [1, target_length, 3, 4]
            # 'pre_motion_lie': self.pre_motion_lie[idx],    # [1, input_size, 3] Lie algebra representation of rots
            # 'fut_motion_lie': self.fut_motion_lie[idx],    # [1, output_size, 3]
            'pre_motion_mask': self.pre_motion_mask[idx],  # [1, input_size]
            'fut_motion_mask': self.fut_motion_mask[idx],  # [1, target_length]
        }

        return sample


