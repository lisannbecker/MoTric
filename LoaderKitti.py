import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import os
import sys

posenet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Continuous-Pose-in-NeRF"))
sys.path.append(posenet_dir)
from PoseNet import PoseNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


### Only one can be True
Danda = False
Fully_Custom = False
LED = True




class KITTIDatasetLeapfrog(Dataset): #Similar to class NBADataset in dataloader_nba.py

    """
    Kitti dataset class compatible with leapfrog (LED) pipeline.

    NOTE: Loads x y translation only instead of full SE3 pose.
    """

    def __init__(self, input_size, preds_size, training, transform=None):
        """
        input_size (int): number of poses used as input (past trajectory)
        preds_size (int): nr of poses to be predicted (future trajectory)
        training (boolean): train or test set data
        """

        window_size = input_size+preds_size
        if training ==True:
            traj_list = ['00', '01', '02', '03', '04', '05', '06', '07']
        else:
            traj_list = ['08', '09', '10']

        ### Load and split data from all KITTI trajectories
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size)
            # p_inputs: [num_windows, input_size, 3, 4]
            # p_targets: [num_windows, window_size - input_size, 3, 4]


        ### 1. Extract only x and y translation
        self.pre_motion_2D = p_inputs[:, :, :2, 3]  # [num_windows, input_size, 2]
        self.fut_motion_2D = p_targets[:, :, :2, 3]  # [num_windows, window_size - input_size, 2]

        ### 2. LED pipeline uses an extra dimension at index 1 for the number of agents
        # We add this extra dimension and set it to 1 as Kitti is single-agent
        self.pre_motion_3D = self.pre_motion_2D.unsqueeze(1) #[num_windows, 1, input_size, 2]
        self.fut_motion_3D = self.fut_motion_2D.unsqueeze(1) #[num_windows, 1, (window_size - input_size), 2]

        ### 3. Create masks (all 1s) matching the temporal dimensions
        self.pre_motion_mask = torch.ones(self.pre_motion_3D.shape[0], 1, self.pre_motion_3D.shape[2])
        self.fut_motion_mask = torch.ones(self.fut_motion_3D.shape[0], 1, self.fut_motion_3D.shape[2])

        self.transform = transform #TODO use in the future for trajectory normalisation

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


class KITTIDatasetCustom(Dataset): #Similar to class NBADataset in dataloader_nba.py

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



def train_danda(p_wins, t_wins, batch_size = 32, kitti_time = True):
    """
    KittiPoseNet contains simple replication with a single trajectory (loss marginally decreases).

    This implementation moves beyond the author's code in the following ways:
    - Splits KITTI trajectories into overlapping subtrajectories
    - Batch training instead of single trajectory per pass
    - Two options for time input: 
            (1) Close to Danda: using ordinal time for subtrajectories (time of first pose = 0, time of last pose = number of poses in subtrajectory e.g., 10)
            (2) Using KITTI times: utilising the timestamps for each each pose that the KITTI dataset provides

    #TODO - Martin
    # Given that we use subtrajectories, shall I normalise each first timestamp and pose of the subtrajectory (maybe by taking derivative for the pose > translation and by setting t0 to 0)? 
    # This would complete the full split of the original KITTI trajectory.

    # For example:
    # subjacetory first time: 70.21 seconds, last time: 73.98 seconds - normalise to - first time: 0s, last time: 3.77s? (ordinal - first time: 78, last time 87 - normalise to - first time: 0, last time 9?)
    # (same for pose)

    ### My current thoughts
    - Pose: Should normalise pose relative to first frame of the sub trajectory to learn motion prior independent of global position
    - Time: Should move first time of subtrajectory to 0 to learn motion prior independent of how far we are into the specific kitti trajectory
    """

    config = {
        'device': "cpu",#"cuda:0",
        'poseNet_freq': 5, #frequency scaling factor of positional encoding
        'layers_feat': [None,256,256,256,256,256,256,256,256],
        'skip': [4],  
        'min_time': 0,
        'max_time': 100,
        'activ': 'relu',
        'cam_lr': 1e-3,
        'max_iter': 20000,
        'use_scheduler': False
        }
    
    epochs = 10
    
    #improve time normalisation (min and max time) for using kitti time
    #if using kitti timestamps this should be normalised based on batch max and min time instead of total max and min time - right now this predicts based on the time of the whole trajectory instead of the trajectory window
    config['min_time'] = float(t_wins[0][0]) #get first time of first window - minimum time 0s
    config['max_time'] = float(t_wins[t_wins.size(0)-1][-1]) #get last time of last window - maximum time



    if kitti_time ==True:
        print('Using KITTI time stamps instead of ordinal time stamps (original)')

    else: #instead of actual timestamps (t_wins), only order from 0-9
        print('Using ordinal time stamps (original) instead of KITTI time stamps')

        config['max_time'] = float(t_wins.size(1))  #max_time is window size - time of trajectory window instead of whole sequence, 
        repetitions = t_wins.size()[0]

        t_wins = torch.arange(config['max_time'])

        t_wins = torch.stack([t_wins] * repetitions, dim=0) #repeat size time_windows number of times, to match t_wins (the version that uses actual timestamps) dimensions.


    ### Load data in batches
    dataset = TensorDataset(p_wins, t_wins)  # store pairs together
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    posenet = PoseNet(config)
    n_traj_windows = t_wins.size()[0]



    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = len(dataloader)

        # Shuffle trajectories randomly
        shuffle_idx = torch.randperm(n_traj_windows)
        p_wins = p_wins[shuffle_idx]
        t_wins = t_wins[shuffle_idx]


        for batch_i, (pose_batch, time_batch) in enumerate(dataloader): #one window per forward pass
            pose_batch, time_batch = pose_batch.to(config['device']), time_batch.to(config['device'])

            est_c2ws = posenet.forward(time_batch) #pass i-th time window
            est_c2ws = est_c2ws.view(time_batch.size(0), time_batch.size(1), 3, 4) #reshape to [batch_size, window_size, 3, 4] instead of [batch_size * window_size, 3, 4]

            loss = torch.abs(pose_batch - est_c2ws).mean() #compare predicted poses (est_c2ws - predicted from time alone) to actual poses (p_wins)
                                                            # TODO inspect how GT poses (of one trajectory) vary over time
            
            loss.backward()
            posenet.step()

            epoch_loss += loss.item()
            #print(f'[{epoch+1}] Batch: {batch_i}, Loss: {loss}')
        

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.5f}")



def load_raw_traj_poses(traj_idx):
    file_path = f'/home/scur2440/thesis/KITTI_odometry/dataset/poses/{traj_idx}.txt'
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
    file_path = f'/home/scur2440/thesis/KITTI_odometry/dataset/sequences/{traj_idx}/times.txt'
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            times.append(float(line.strip()))
    return torch.tensor(times)


def to_relative(pose_windows, time_windows, input_size):
    last_idx = input_size-1
    last_poses = pose_windows[:, last_idx, :, :].unsqueeze(1) #[num_windows, 1, 3, 4] all last poses that need to subtracted
    # print(last_poses.size())
    pose_windows_relative = pose_windows-last_poses
    # print(pose_windows_relative[0])

    last_times = time_windows[:, last_idx].unsqueeze(1) #[num_windows, 1]
    # print(last_times.size())
    time_windows_relative = time_windows-last_times
    #print(time_windows_relative[100])

    return pose_windows_relative, time_windows_relative

def to_normalised(pose_windows, time_windows, input_size):
    """Normalise times and translations but not rotations"""
    t_min = time_windows.min(dim=1, keepdim=True)[0] #(minimum_value, minimum_value_index)
    t_max = time_windows.max(dim=1, keepdim=True)[0]

    #Scale to [-1,1] TODO is this ideal?
    time_windows = 2 * (time_windows - t_min) / (t_max - t_min) - 1
    # print(time_windows.size()) #[4529, 13]
    # print(time_windows[0])

    # print(pose_windows[0])

    translations_only = pose_windows[:,:,:3,3]
    # print(translations_only[0])
    # print(translations_only.size()) #[4529, 13, 3]
    trans_min = translations_only.min(dim=1, keepdim=True)[0] #takes minimum independently for each coordinate - pose invariant
    trans_max = translations_only.max(dim=1, keepdim=True)[0]

    translations_normalised = 2 * (translations_only - trans_min) / (trans_max - trans_min) - 1
    pose_windows[:,:,:3,3] = translations_normalised
    # print(pose_windows[0])
    
    return pose_windows, time_windows



def load_one_and_split(traj, window_size, input_size, use_relative = False, use_normalised = False): #single trajectory - load and split
    """
    Loads a single trajectory’s poses and timestamps, then splits them into overlapping windows of size window_size.
    """
    poses = load_raw_traj_poses(traj)
    times = load_raw_traj_times(traj)

    num_windows = poses.shape[0] - window_size +1
    pose_windows = torch.stack([poses[i:i+window_size, :, :] for i in range(num_windows)]) #save in list and then stack in tensor [num_windows, window_size, 3, 4]
    time_windows = torch.stack([times[i:i+window_size] for i in range(num_windows)]) # [num_windows, window_size]

    if Danda == True:
        return pose_windows, time_windows # Split into subtrajectories but not input and target

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

def load_all_and_split(traj_list, window_size, input_size, use_relative = False, use_normalised = False):
    """
    Returns combines pose and time windows of size window_size for all trajectories.
    
    Data from different trajectories is kept separate.
    """
    p_input_all, p_target_all, t_input_all, t_target_all = [],[],[],[]

    for traj in traj_list:
        input_poses, target_poses, input_times, target_times = load_one_and_split(traj, window_size, input_size, use_relative, use_normalised)
        
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



"""
- Pose: Should normalise pose relative to first frame of the sub trajectory to learn motion prior independent of global position
- Time: Should move first time of subtrajectory to 0 to learn motion prior independent of how far we are into the specific kitti trajectory
"""

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

    elif LED == True:

        train_dataset = KITTIDatasetLeapfrog(input_size=10, preds_size=20, training=True, transform=None) #which trajectories to load, window size, out of which past trajectories (rest is target trajectories), no normalisation [WIP]
        print(len(train_dataset)) # 16106 for input_size=10, preds_size=20
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

        test_dataset = KITTIDatasetLeapfrog(input_size=10, preds_size=20, training=False, transform=None)
        print(len(test_dataset)) # 6776 for input_size=10, preds_size=20
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True, num_workers=4, collate_fn=seq_collate_kitti, pin_memory=True)

        for batch in train_loader:
            print(batch.keys())
            print("Batch pre-motion shape:", batch['pre_motion_3D'].shape)  # [batch_size, 1, past_poses, 2]
            print("Batch future motion shape:", batch['fut_motion_3D'].shape)  # [batch_size, 1, future_poses, 2]
            print("Batch pre-motion mask shape:", batch['pre_motion_mask'].shape)  # [batch_size, 1, past_poses, 2]
            print("Batch future motion mask shape:", batch['fut_motion_mask'].shape)  # [batch_size, 1, future_poses, 2]
            print("traj_scale:", batch['traj_scale'])
            print("pred_mask:", batch['pred_mask'])
            print("seq:", batch['seq'])
            exit()
            break

    # print(pose_input_wins.size()) #torch.Size([23069, 10, 3, 4])
    # print(pose_target_wins.size()) #torch.Size([23069, 3, 3, 4])

    #train_custom(p_wins, t_wins)





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