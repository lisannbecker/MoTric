import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
import random

import pandas as pd
import numpy as np

from scipy.spatial.transform import Rotation as R

posenet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Continuous-Pose-in-NeRF"))
sys.path.append(posenet_dir)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class LoadDatasetLeapfrog(Dataset): #3D translation and 3D Lie algebra for rotation

    """
    Kitti/Spires/Newer dataset class compatible with leapfrog (LED) pipeline.
    NOTE: Loads x z y translation (original LED) and full SE3 poses (rotation either Lie algebra "6D" or 6D "9D").
    """

    def __init__(self, dataset: str, dims: int, input_size: int, preds_size: int, training: bool, final_eval: bool, relative:bool=False, normalised:bool=False, 
                 train_ratio=0.80, eval_ratio=0.10, seed=42, overlapping:bool=False, selected_trajectories:bool=False, synthetic_gt:str = 'straight',
					synthetic_noise:str = 'random_walk'):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """            
        window_size = input_size+preds_size

        print('Overfitting:', overlapping)

        ### 0. Define dataset file paths
        if dataset == 'kitti':
            indices = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] #uncomment
            file_names = [f'/scratch-shared/scur_2440/KITTI_odometry_and_depth_no_exclusions/poses/{idx}.txt' for idx in indices]
        elif dataset == 'synthetic':
            dataset_path = '/home/scur2440/MoTric/synthetic_data'
            gt_type = synthetic_gt #straight, right_curve
            noise_type = synthetic_noise #random_independent, random_walk, right_bias

            #get file with noisy past and clean gt future
            noisy_file = f'{dataset_path}/{gt_type}_{noise_type}/synthetic_noisy_past_poses.txt'

        elif dataset == 'newer':
            dataset_path = '/home/scur2440/MoTric/NewerCollege'
            file_names = [
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-easy.csv', 
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-medium.csv',
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-hard.csv',
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-stairs.csv',
                          f'{dataset_path}/collection2/ground_truth/tum_format/gt-nc-park.csv',
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_easy.csv',
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_medium.csv',
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_hard.csv'
                          ]
        elif dataset == 'spires': 
            dataset_path = '/home/scur2440/MoTric/Spires'
            file_names = [f'{dataset_path}/2024-03-13-observatory-quarter-01/gt-tum.txt',
                          f'{dataset_path}/2024-03-13-observatory-quarter-02/gt-tum.txt',
                          f'{dataset_path}/2024-05-20-bodleian-library-02/gt-tum.txt'
                          ]
        elif dataset == 'pedestrian_prior':
            dataset_path_spires = '/home/scur2440/MoTric/Spires'
            dataset_path_newer = '/home/scur2440/MoTric/NewerCollege'
            file_names = [
                          f'{dataset_path_spires}/2024-03-13-observatory-quarter-01/gt-tum.txt',
                          f'{dataset_path_spires}/2024-03-13-observatory-quarter-02/gt-tum.txt',
                          f'{dataset_path_spires}/2024-05-20-bodleian-library-02/gt-tum.txt',
                          f'{dataset_path_newer}/collection1/ground_truth/tum_format/gt-nc-quad-easy.csv', 
                          f'{dataset_path_newer}/collection1/ground_truth/tum_format/gt-nc-quad-medium.csv',
                          f'{dataset_path_newer}/collection1/ground_truth/tum_format/gt-nc-quad-hard.csv',
                          f'{dataset_path_newer}/collection1/ground_truth/tum_format/gt-nc-stairs.csv',
                          f'{dataset_path_newer}/collection2/ground_truth/tum_format/gt-nc-park.csv',
                          f'{dataset_path_newer}/collection3_maths/ground_truth/tum_format/gt_math_easy.csv',
                          f'{dataset_path_newer}/collection3_maths/ground_truth/tum_format/gt_math_medium.csv',
                          f'{dataset_path_newer}/collection3_maths/ground_truth/tum_format/gt_math_hard.csv'
                          ]

        ### 0.1 overlapping but not overfitting: split train/val/test by selected trajectories
        if selected_trajectories:

            if dataset == 'kitti':
                if training == True:
                    indices = ['00', '01', '02', '04', '06', '07', '08']
                    file_names = [f'/scratch-shared/scur_2440/KITTI_odometry_and_depth_no_exclusions/poses/{idx}.txt' for idx in indices]
                elif final_eval == True:
                    file_names = [f'/scratch-shared/scur_2440/KITTI_odometry_and_depth_no_exclusions/poses/05.txt' ]
                else: #validation
                    indices = ['03', '09', '10']
                    file_names = [f'/scratch-shared/scur_2440/KITTI_odometry_and_depth_no_exclusions/poses/{idx}.txt' for idx in indices]

            elif dataset == 'newer':
                if training == True:
                    file_names = [
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-easy.csv', 
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-medium.csv',
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-stairs.csv',
                          f'{dataset_path}/collection2/ground_truth/tum_format/gt-nc-park.csv',
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_easy.csv',
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_hard.csv'
                          ]
                elif final_eval == True:
                    file_names = [
                          f'{dataset_path}/collection1/ground_truth/tum_format/gt-nc-quad-hard.csv',
                          ]
                else: #validation
                    file_names = [
                          f'{dataset_path}/collection3_maths/ground_truth/tum_format/gt_math_medium.csv'
                          ]
            else:
                print("[ERROR] selected trajectories is not enabled for the current dataset.")
                exit()

            overlapping = True
            print("Dataset split by trajectories, overlapping set to True.")
            if len(file_names) == 0:
                raise ValueError("None of the provided selected_trajectories matched the available files.")


        ### 1. Load and split data from all trajectories
        if dataset == 'synthetic':
            p_inputs, p_targets, t_inputs, t_targets = load_synthetic_and_split(noisy_file, window_size, input_size, dims, relative, normalised, overlapping)
            print("[WARNING] Relativised / normalised loading not implemented for synthetic dataset. Poses are converted to relative within pipeline (data_preprocess).")
        else:    
            p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(dataset, file_names, window_size, input_size, dims, relative, normalised, overlapping) 
            # p_inputs: [num_windows, input_size, 3, 4] p_targets: [num_windows, output_size, 3, 4]

        if t_targets is not None:
            print(f'Time diff avg (fut): {(t_targets[:,-1]-t_targets[:,0]).mean(dim=-1).item():.5f}')

        #================== 2. Convert to correct rotation representation (None, Lie, Quaternion or 6D) =====================
        ### 2.1. 2D: Extract x, y translation, kitti used to be ordered xzy, but already corrected
        if dims == 2:
            trans_pre = p_inputs[:, :, :2, 3] 
            trans_fut = p_targets[:, :, :2, 3] 
        ### 2.1. 3D/6D/7D/9D: Extract x, z, y translation
        elif dims == 7:

            if p_inputs.ndim == 3 and p_inputs.shape[-1] == 7:
                print('Is 7d already')
                SE3_pre = p_inputs       # [N, past_frames, 7]
                SE3_fut = p_targets      # [N, future_frames, 7]
            else:
                print('Assumes homogenous-matrix from')
                R_pre = p_inputs[..., :3, :3]       # [N, W, 3,3]
                t_pre = p_inputs[..., :3,  3]       # [N, W, 3]
                R_fut = p_targets[..., :3, :3]
                t_fut = p_targets[..., :3,  3]

                # 1) bring your 3×3’s onto the CPU and flatten
                R_pre_flat = R_pre.cpu().reshape(-1, 3, 3).numpy()
                R_fut_flat = R_fut.cpu().reshape(-1, 3, 3).numpy()

                # 2) use scipy to convert
                quat_pre_flat = R.from_matrix(R_pre_flat).as_quat()  # shape: (N*W, 4), order [x, y, z, w]
                quat_fut_flat = R.from_matrix(R_fut_flat).as_quat()

                # 3) re-pack into torch tensors of shape [N, W, 4]
                q_pre = torch.from_numpy(quat_pre_flat).view(*R_pre.shape[:-2], 4).to(R_pre.device)
                q_fut = torch.from_numpy(quat_fut_flat).view(*R_fut.shape[:-2], 4).to(R_fut.device)

                # 4) now concatenate to get your 7-vectors
                SE3_pre = torch.cat([t_pre, q_pre], dim=-1)   # [N, W, 7]
                SE3_fut = torch.cat([t_fut, q_fut], dim=-1)
            

        else:
            trans_pre = p_inputs[:, :, :3, 3]  # [num_windows, input_size, 3] << 3D adjustment
            trans_fut = p_targets[:, :, :3, 3]  # [num_windows, output_size, 3] << 3D adjustment
            # print(trans_pre[0], '\n', trans_fut[0])


        ### 2.2 6D: Extract rotation and convert to Lie algebra (and make relative)
        if dims ==6:
            self.lie = Lie()

            if relative:
                rotations_pre, rotations_fut = to_relative_rotations_lie(p_inputs, p_targets, input_size, self.lie)

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
            SE3_pre = torch.cat([trans_pre, lie_pre], dim=-1) # [tx, ty, tz, wx, wy, wz]
            SE3_fut = torch.cat([trans_fut, lie_fut], dim=-1)

        elif dims == 7:
            # For 7D, SE3_pre and SE3_fut have already been set.
            pass

        # p_inputs: [num_windows, input_size, 3, 4] p_targets: [num_windows, output_size, 3, 4]
        elif dims == 9: #6D rotation representation - first two columns of homogenous matrix; flattened by rows
            # print(p_inputs[0,0,:,:])
            # Extract first two columns (6D rotation)
            rotations_pre = p_inputs[:, :, :3, [0, 1]]  # Shape: [num_windows, input_size, 3, 2]
            rotations_fut = p_targets[:, :, :3, [0, 1]]  # Shape: [num_windows, output_size, 3, 2]

            # Extract the last column (Translation)
            translations_pre = p_inputs[:, :, :3, [3]]  # Shape: [num_windows, input_size, 3, 1]
            translations_fut = p_targets[:, :, :3, [3]]  # Shape: [num_windows, output_size, 3, 1]

            # Reshape to flatten rotations into (B, input_size, 6) and translations into (B, input_size, 3)
            rotations_pre = rotations_pre.reshape(rotations_pre.shape[0], rotations_pre.shape[1], -1)  # [B, input_size, 6]
            rotations_fut = rotations_fut.reshape(rotations_fut.shape[0], rotations_fut.shape[1], -1)  # [B, output_size, 6]

            translations_pre = translations_pre.squeeze(-1)  # Remove last dim -> [B, input_size, 3]
            translations_fut = translations_fut.squeeze(-1)  # Remove last dim -> [B, output_size, 3]

            # Concatenate to get final input/output tensors
            SE3_pre = torch.cat((translations_pre, rotations_pre), dim=-1)  # Shape: [B, input_size, 9] - 9 = tx,ty,tz,r...
            SE3_fut = torch.cat((translations_fut, rotations_fut), dim=-1)  # Shape: [B, output_size, 9]

            # print(SE3_pre[0,0,:])
            # print(SE3_pre.size(), SE3_fut.size())  # Debugging
            # exit()

        else: #2D or 3D
            SE3_pre = trans_pre
            SE3_fut = trans_fut


        ### 3. Add agent dimension
        all_pre = SE3_pre.unsqueeze(1)  # [num_windows, 1, input_size, 6]
        all_fut = SE3_fut.unsqueeze(1)  # [num_windows, 1, output_size, 6]

        ### 4. Create masks (all 1s) matching the temporal dimensions
        all_pre_mask = torch.ones(all_pre.shape[0], 1, all_pre.shape[2])
        all_fut_mask = torch.ones(all_fut.shape[0], 1, all_fut.shape[2])

        ### 5. Randomly split the windows using a fixed seed 
        total_windows = all_pre.shape[0]
        if selected_trajectories: # and (not training or final_eval): #no handpicked trajectories FIXME
            selected_indices = list(range(total_windows))
        else: 
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
        self.pre_motion_3D = self.pre_motion_3D.float()
        self.fut_motion_3D = self.fut_motion_3D.float()

        self.pre_motion_mask = all_pre_mask[selected_indices]
        self.fut_motion_mask = all_fut_mask[selected_indices]
        print('Len dataset:', len(self.pre_motion_3D))



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


def seq_collate_custom(data): #allows to have None as pred_mask (as opposed to default collate)
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

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # Normalize the quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R


def TUM_to_homogenous_matrices(file_path):    
    tum_data = pd.read_csv(file_path, sep=" ", header=None, names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])

    transformation_mats = []
    times = []
    for i, row in tum_data.iterrows():
        R = quaternion_to_rotation_matrix(row['qx'], row['qy'], row['qz'], row['qw']) #3x3 rotation
        t = np.array([[row['x']], [row['y']], [row['z']]]) # translation
        T = np.hstack((R,t)) #3x4 matrix

        transformation_mats.append(torch.tensor(T, dtype=torch.float32))
        times.append(row['timestamp'])

    return torch.stack(transformation_mats), torch.tensor(times, dtype=torch.float32)

def load_tum_7d(file_name: str):
    """
    Loads a TUM format file without converting the pose.
    Returns:
      poses: Tensor of shape [N, 7] where each pose is [tx, ty, tz, qx, qy, qz, qw]
      times: Tensor of shape [N] of timestamps.
    """
    poses = []
    times = []
    
    with open(file_name, 'r') as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # skip malformed lines
            # TUM format: timestamp, tx, ty, tz, qx, qy, qz, qw
            t = float(parts[0])
            translation = [float(x) for x in parts[1:4]]
            quaternion = [float(x) for x in parts[4:8]]
            
            poses.append(translation + quaternion)
            times.append(t)
    
    poses = torch.tensor(poses, dtype=torch.float32)  # [N, 7]
    times = torch.tensor(times, dtype=torch.float32)  # [N]
    return poses, times

def to_relative(pose_windows, time_windows, input_size):
    rotations = pose_windows[..., :3]
    translations = pose_windows[..., 3]
    
    last_idx = input_size-1
    last_translation = translations[:, last_idx, :].unsqueeze(1) #[num_windows, 1, 3] Get translation at the last input time step

    translations_relative = translations-last_translation   # subtract last pose of past trajectory / current pose from all poses
                                                            # future will be pos, past neg

    #same for times
    if time_windows is not None:
        last_times = time_windows[:, last_idx].unsqueeze(1) #[num_windows, 1]
        time_windows_relative = time_windows-last_times
    else:
        time_windows_relative = None   

    #concat into one matrix again
    translations_relative = translations_relative.unsqueeze(-1)
    pose_windows_relative = torch.cat([rotations, translations_relative], dim=-1)
    return pose_windows_relative, time_windows_relative

def to_relative_rotations_lie(past_pose_windows, fut_pose_windows, input_size, lie_converter):
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

def to_relative_7d(pose_windows: torch.Tensor, time_windows: torch.Tensor, input_size: int):
    """
    Converts a window of 7D poses (translation + quaternion) to a relative coordinate system.
    For translation, it subtracts the translation at the last input time step.
    For rotation, it computes relative quaternions:
        relative_q = inverse(q_ref) * q.
    Expects pose_windows of shape [num_windows, window_size, 7]
    and time_windows of shape [num_windows, window_size].
    """
    # Split into translation and quaternion components
    translations = pose_windows[..., :3]  # shape: [num_windows, window_size, 3]
    quats = pose_windows[..., 3:7]        # shape: [num_windows, window_size, 4]
    
    last_idx = input_size - 1
    ref_translation = translations[:, last_idx, :].unsqueeze(1)  # [num_windows, 1, 3]
    ref_quat = quats[:, last_idx, :].unsqueeze(1)                # [num_windows, 1, 4]
    
    # Compute relative translations
    relative_translations = translations - ref_translation
    
    # Compute relative quaternions: inverse(ref) * q
    relative_quats = quaternion_relative(ref_quat, quats)
    
    # Combine back into a 7D representation
    relative_poses = torch.cat([relative_translations, relative_quats], dim=-1)
    
    # Make timestamps relative
    if time_windows is not None:
        ref_time = time_windows[:, last_idx].unsqueeze(1)
        relative_times = time_windows - ref_time
    else:
        relative_times = None
    
    return relative_poses, relative_times

def to_normalised_7d(pose_windows: torch.Tensor, time_windows: torch.Tensor, input_size: int):
    """
    Normalises the translation components and timestamps for 7D pose windows to the range [-1, 1].
    The quaternion values are left unchanged.
    Expects pose_windows of shape [num_windows, window_size, 7] and
    time_windows of shape [num_windows, window_size].
    """
    # Normalise time windows per window
    t_min = time_windows.min(dim=1, keepdim=True)[0]
    t_max = time_windows.max(dim=1, keepdim=True)[0]
    if time_windows is not None:
        norm_times = 2 * (time_windows - t_min) / (t_max - t_min) - 1
    else:   
        norm_times = None
    
    # Normalise the translation components
    translations = pose_windows[..., :3]  # shape: [num_windows, window_size, 3]
    t_min_trans = translations.min(dim=1, keepdim=True)[0]
    t_max_trans = translations.max(dim=1, keepdim=True)[0]
    norm_translations = 2 * (translations - t_min_trans) / (t_max_trans - t_min_trans) - 1

    # Keep quaternions unchanged
    quats = pose_windows[..., 3:7]
    norm_poses = torch.cat([norm_translations, quats], dim=-1)
    
    return norm_poses, norm_times

# ------- helpers for quaternion operations -------

def quaternion_conjugate(q: torch.Tensor):
    """
    Compute the quaternion conjugate.
    Assumes quaternions are in the form [qx, qy, qz, qw].
    For a quaternion q, the conjugate is [-qx, -qy, -qz, qw].
    q: tensor of shape (..., 4)
    Returns: tensor of shape (..., 4)
    """
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

def quaternion_multiply(q: torch.Tensor, r: torch.Tensor):
    """
    Multiply two quaternions.
    Assumes each quaternion is of the form [qx, qy, qz, qw].
    q, r: tensors of shape (..., 4)
    Returns: quaternion product of shape (..., 4)
    """
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)
    
    w = qw * rw - qx * rx - qy * ry - qz * rz
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    
    return torch.stack((x, y, z, w), dim=-1)

def quaternion_relative(ref_quat: torch.Tensor, quats: torch.Tensor):
    """
    Computes relative quaternions.
    Given a reference quaternion ref_quat (shape: [num_windows, 1, 4]) and a tensor of
    quaternions quats (shape: [num_windows, window_size, 4]),
    the relative quaternion is computed as:
        relative_q = conjugate(ref_quat) * current_q.
    """
    ref_conj = quaternion_conjugate(ref_quat)
    return quaternion_multiply(ref_conj, quats)


def load_synthetic_and_split(noisy_file, window_size, input_size, dims, use_relative=False, use_normalised=False, overlapping=False):
    """
    Loads trajectories from pairs of files (noisy, GT) and splits them into windows.
    
    Args:
        file_pairs: List of tuples [(noisy_file, gt_file), ...]
        window_size: Total window size (input + future)
        input_size: Number of past frames
        dims: Dimensions to use
        use_relative: Whether to use relative transformation
        use_normalised: Whether to normalise poses
        overlapping: Whether to use overlapping windows
    
    Returns:
        Tensors for input poses, target poses, input times, target times
    """
    # 1) read all poses from the file (skipping blank lines)
    all_poses = load_raw_traj_poses(noisy_file)  # shape [T_total, 3, 4]

    # 2) build windows
    if not overlapping:
        num_windows = all_poses.shape[0] // window_size
        windows = torch.stack([
            all_poses[i*window_size:(i+1)*window_size]
            for i in range(num_windows)
        ])  # [num_windows, window_size, 3, 4]
    else:
        num_windows = all_poses.shape[0] - window_size + 1
        windows = torch.stack([
            all_poses[i:i+window_size]
            for i in range(num_windows)
        ])  # [num_windows, window_size, 3, 4]

    # 3) optional relative / normalisation (exactly as in your other loaders)
    if use_relative:
        windows, _ = to_relative(windows, None, input_size)
    if use_normalised:
        windows, _ = to_normalised(windows, None, input_size)

    # 4) split into past / future
    p_inputs  = windows[:, :input_size, :, :]   # noisy past
    p_targets = windows[:, input_size:, :, :]   # clean future

    # 5) collect & concatenate
    # (here it’s one “file,” so no need for a loop)
    combined_p_inputs  = p_inputs
    combined_p_targets = p_targets

    combined_t_inputs = None
    combined_t_targets = None
    
    return combined_p_inputs, combined_p_targets, combined_t_inputs, combined_t_targets

def load_raw_traj_poses(file_path):
    """assumes format [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():  # Skip empty lines
                continue
            values = list(map(float, line.strip().split())) #list for line
            SE3 = torch.tensor(values).reshape(3,4) #SE3 matrix without bottom row 0,0,0,1
            poses.append(SE3)
    return torch.stack(poses)

def load_raw_traj_times_kitti(file_path):
    # /home/scur2440/MoTric/KITTI_odometry/dataset/poses/05.txt' 
    file_path = file_path.replace('poses', 'sequences').replace('.txt', '/times.txt') # f'/home/scur2440/MoTric/KITTI_odometry/dataset/sequences/{file_idx}/times.txt'
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            times.append(float(line.strip()))
    return torch.tensor(times)




def load_one_and_split(dataset, file_name, window_size, input_size, dims, use_relative=False, use_normalised=False, overlapping=False):
    """
    Loads a single trajectory's poses and timestamps, then splits them into overlapping windows of size window_size.
    """
    if dataset == 'kitti':
        poses = load_raw_traj_poses(file_name)
        ### ===== reorder translation to xyz if necessary =====
        t_raw  = poses[..., :3, 3] 
        # swap KITTI’s (x, z, y) → (x, y, z)
        corrected = t_raw[:, [0, 2, 1]].clone()
        # write back into the homogeneous matrix
        poses[:, 0, 3] = corrected[:, 0]
        poses[:, 1, 3] = corrected[:, 1]
        poses[:, 2, 3] = corrected[:, 2]
        # ==========================

        times = load_raw_traj_times_kitti(file_name)


    elif dataset in ['newer', 'spires', 'pedestrian_prior']:
        if dims == 7:
            poses, times = load_tum_7d(file_name)
        else:
            poses, times = TUM_to_homogenous_matrices(file_name)
            
            ### ===== reorder translation to xyz if necessary =====
            t_raw  = poses[..., :3, 3] 
            if dataset == 'kitti':
                # swap KITTI’s (x, z, y) → (x, y, z)
                corrected = t_raw[:, [0, 2, 1]].clone()
            else:
                corrected = t_raw # already (x,y,z)

            # write back into the homogeneous matrix
            poses[:, 0, 3] = corrected[:, 0]
            poses[:, 1, 3] = corrected[:, 1]
            poses[:, 2, 3] = corrected[:, 2]
            # ==========================

    else:
        raise ValueError("Unsupported dataset type.")
    



    # Create windows for poses and times based on overlapping flag and dims
    if overlapping == False:
        num_windows = poses.shape[0] // window_size 
        if dims == 7:
            pose_windows = torch.stack([poses[i*window_size:(i+1)*window_size, :] for i in range(num_windows)])
        else:
            pose_windows = torch.stack([poses[i*window_size:(i+1)*window_size, :, :] for i in range(num_windows)])
        if times is not None:
            time_windows = torch.stack([times[i*window_size:(i+1)*window_size] for i in range(num_windows)])
        else:  
            time_windows = None
    else:
        num_windows = poses.shape[0] - window_size + 1
        if dims == 7:
            pose_windows = torch.stack([poses[i:i+window_size, :] for i in range(num_windows)])
        else:
            pose_windows = torch.stack([poses[i:i+window_size, :, :] for i in range(num_windows)])
        if times is not None:
            time_windows = torch.stack([times[i:i+window_size] for i in range(num_windows)])
        else:  
            time_windows = None


    # Optionally apply relative transformation and normalisation.
    if use_relative:
        if dims == 7:
            pose_windows, time_windows = to_relative_7d(pose_windows, time_windows, input_size)
        else:
            pose_windows, time_windows = to_relative(pose_windows, time_windows, input_size)
    
    if use_normalised:
        if dims == 7:
            pose_windows, time_windows = to_normalised_7d(pose_windows, time_windows, input_size)
        else:
            pose_windows, time_windows = to_normalised(pose_windows, time_windows, input_size)
    
    # Split into input and target windows.
    if dims == 7:
        input_poses = pose_windows[:, :input_size, :]    # shape: [num_windows, input_size, 7]
        target_poses = pose_windows[:, input_size:, :]     # shape: [num_windows, window_size - input_size, 7]
    else:
        input_poses = pose_windows[:, :input_size, :, :]   # shape: [num_windows, input_size, 3, 4]
        target_poses = pose_windows[:, input_size:, :, :]    # shape: [num_windows, window_size - input_size, 3, 4]

    if times is not None:
        input_times = time_windows[:, :input_size]             # [num_windows, input_size]
        target_times = time_windows[:, input_size:]            # [num_windows, window_size - input_size]
    else:
        input_times, target_times = None, None

    return input_poses, target_poses, input_times, target_times

def load_all_and_split(dataset, file_name_list, window_size, input_size, dims, use_relative = False, use_normalised = False, overlapping = False):
    """
    Returns combines pose and time windows of size window_size for all trajectories.
    
    Data from different trajectories is kept separate.
    """
    p_input_all, p_target_all, t_input_all, t_target_all = [],[],[],[]

    for file_name in file_name_list:
        input_poses, target_poses, input_times, target_times = load_one_and_split(dataset, file_name, window_size, input_size, dims, use_relative, use_normalised, overlapping)
        print('file name & length', file_name, len(input_poses))
        p_input_all.append(input_poses)
        p_target_all.append(target_poses)
        if input_times is not None:
            t_input_all.append(input_times)
            t_target_all.append(target_times)

    combined_p_inputs = torch.cat(p_input_all)
    combined_p_targets = torch.cat(p_target_all)
    combined_t_inputs = torch.cat(t_input_all) if t_input_all else None
    combined_t_targets = torch.cat(t_target_all) if t_target_all else None
    return combined_p_inputs, combined_p_targets, combined_t_inputs, combined_t_targets


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

    dimensions = 7 # possible values: 2, 3, 6 (Lie), 7 (TUM), 9 (6d)
    dataset = 'synthetic' # possible values: kitti (AD), newer (Pedestrian), spires (Pedestrian), synthetic (pedestrian), pedestrian_prior (newer and spires)
    print(f'{dimensions}D {dataset.upper()}\n')

    train_dataset = LoadDatasetLeapfrog(dataset=dataset, dims = dimensions, input_size=10, preds_size=22, training=True, final_eval=False, relative=False, normalised=False, overlapping=False, selected_trajectories=False) #which trajectories to load, window size, out of which past trajectories (rest is target trajectories), no normalisation [WIP]
    # print(train_dataset.pre_motion_3D[0,0,9,:])

    
    if dimensions==9:
        print_some_stats(train_dataset.fut_motion_3D[..., :3], train_dataset.fut_motion_3D[..., 3:], 3)

    else:
        print_some_stats(train_dataset.fut_motion_3D, None, dimensions)
    print('Train trajectories:', len(train_dataset)) # 16106 for input_size=10, preds_size=20
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=seq_collate_custom, pin_memory=True)
    
    test_dataset = LoadDatasetLeapfrog(dataset=dataset, dims=dimensions, input_size=10, preds_size=20, training=False, final_eval=False, relative=False, normalised=False, overlapping=False, selected_trajectories=False)
    if dimensions==9:
        print_some_stats(test_dataset.fut_motion_3D[..., :3], test_dataset.fut_motion_3D[..., 3:], 3)

    else:
        print_some_stats(test_dataset.fut_motion_3D, None, dimensions)
    print('Test trajectories:', len(test_dataset)) # 6776 for input_size=10, preds_size=20
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=4, collate_fn=seq_collate_custom, pin_memory=True)


    for batch in train_loader:
        # print(batch['pre_motion_3D'][0])
        print(batch.keys())
        print("\nBatch pre-motion shape:", batch['pre_motion_3D'].shape)  # [batch_size, 1, past_poses, dimensions]
        print("Batch future motion shape:", batch['fut_motion_3D'].shape)  # [batch_size, 1, future_poses, dimensions]
        print("Batch pre-motion mask shape:", batch['pre_motion_mask'].shape)  # [batch_size, 1, past_poses, dimensions]
        print("Batch future motion mask shape:", batch['fut_motion_mask'].shape)  # [batch_size, 1, future_poses, dimensions]
        print("traj_scale:", batch['traj_scale'])
        print("pred_mask:", batch['pred_mask'])
        print("seq:", batch['seq'])
        exit()
        break


