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
        self.p_inputs, self.p_targets, self.t_inputs, self.t_targets = load_all_and_split(traj_list, window_size, input_size, dims, use_relative, use_normalised)
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

class LoadDatasetLeapfrog2D(Dataset): # corresponds to class NBADataset in dataloader_nba.py

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
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size, dims, relative, normalised)
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

class LoadDatasetLeapfrog3D(Dataset): #Similar to class NBADataset in dataloader_nba.py

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
        p_inputs, p_targets, t_inputs, t_targets = load_all_and_split(traj_list, window_size, input_size, dims)
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

class LoadDatasetLeapfrog6D(Dataset): #3D translation and 3D Lie algebra for rotation

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


