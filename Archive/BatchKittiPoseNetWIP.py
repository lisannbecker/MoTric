import math
import torch
import torch.nn.functional as torch_F
from torch.optim.lr_scheduler import ExponentialLR
import rotation_conversions

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# example configs
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


class PoseNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg['device']
        # init network
        self.transNet = TransNet(cfg)
        self.transNet.to(self.device)
        self.rotsNet = RotsNet(cfg)
        self.rotsNet.to(self.device)
        self.cam_lr = cfg['cam_lr']

        # init optimizer and scheudler
        self.optimizer_posenet = torch.optim.Adam([dict(params=self.transNet.parameters(),lr=self.cam_lr), dict(params=self.rotsNet.parameters(),lr=self.cam_lr*0.2) ])
        gamma = (1e-2)**(1./cfg['max_iter'])
        if self.cfg['use_scheduler']:
            self.scheduler = ExponentialLR(self.optimizer_posenet, gamma=gamma)

        # set the normalizer mapping
        self.min_time = cfg['min_time']
        self.max_time = cfg['max_time']


    def step(self):
        self.optimizer_posenet.step()
        if self.cfg['use_scheduler']:
            self.scheduler.step()
        self.optimizer_posenet.zero_grad()
 

    def forward(self, time):
        assert torch.all(time >= self.min_time) and torch.all(time <= self.max_time), 'time out of range'
        time = 2*(time - self.min_time) / (self.max_time - self.min_time) - 1 #normalise time
        #print('Time\n',time)

        trans_est = self.transNet.forward(self.cfg, time) #translation estimate
        rots_feat_est = self.rotsNet.forward(self.cfg, time)

        rotmat_est = rotation_conversions.quaternion_to_matrix(rots_feat_est) #rotation estimate in quatnerions

        # make c2w
        c2w_est = torch.cat([rotmat_est, trans_est.unsqueeze(-1)],dim = -1 ) #combine rot and trans into camera-to-world transformation matrix 
        # point in the camera's coordinate system * c2w = point in global world coordinate system

        return c2w_est

class TransNet(torch.nn.Module): #Translation network

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = cfg['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.device = cfg['device']
        self.cfg = cfg

    def define_network(self,cfg):
        self.mlp_transnet = torch.nn.ModuleList()
        layers_list = cfg['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['skip'] : k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.initialize_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_transnet.append(linear)

    def initialize_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, cfg, index):
        index = torch.tensor(index).to(self.device)

        # todo encoding the index
        #encoding time index into high-dimensional vector using sinusoidal function
        index = index.reshape(-1,1).to(torch.float32)
        points_enc = self.positional_encoding(index, L=cfg['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]

        translation_feat = points_enc
        activ_f = getattr(torch_F,self.cfg['activ']) 

        for li,layer in enumerate(self.mlp_transnet): #pass encoded time through MLP to predict 3D translation
            if li in cfg['skip']: translation_feat = torch.cat([translation_feat,points_enc],dim=-1)
            translation_feat = layer(translation_feat)
            if li==len(self.mlp_transnet)-1:
                translation_feat = torch_F.tanh(translation_feat) # note we assume bounds is [-1,1]; final layer thanh constrains output to [-1, 1]
            else:
                translation_feat = activ_f(translation_feat) 
        return translation_feat

    def positional_encoding(self,input,L): # [B,...,N] #L is frequency scaling factor of sinusoidal function = how many frequency terms we need
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*math.pi # [L] # Frequency scaling
        spectrum = input[...,None]*freq # [B,...,N,L] # Expand input and multiply by frequencies
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L] # Compute sine and cosine components
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc


class RotsNet(torch.nn.Module): #Rotation network

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.input_t_dim = cfg['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.device = cfg['device']

    def define_network(self,cfg):
        layers_list = cfg['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        self.mlp_quad = torch.nn.ModuleList()
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4 
            linear = torch.nn.Linear(k_in,k_out)

            self.initialize_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def initialize_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, cfg, index):
        index = torch.tensor(index).to(self.device)
        index = index.reshape(-1,1).to(torch.float32)
        activ_f = getattr(torch_F,self.cfg['activ']) 

        points_enc = self.positional_encoding(index,L=cfg['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]        
        rotation_feat = points_enc
        for li,layer in enumerate(self.mlp_quad):
            if li in cfg['skip']: rotation_feat = torch.cat([rotation_feat,points_enc],dim=-1)
            rotation_feat = layer(rotation_feat)
            if li==len(self.mlp_quad)-1:
                rotation_feat[:,1:] = torch_F.tanh(rotation_feat[:,1:])#torch_F.sigmoid(rotation_feat[:,1:])
                rotation_feat[:,0] = 1*(1 - torch_F.tanh(rotation_feat[:,0]))
            else:
                rotation_feat = activ_f(rotation_feat)

        norm_rots = torch.norm(rotation_feat,dim=-1)
        rotation_feat_norm = rotation_feat / (norm_rots[...,None] +1e-18)
        return rotation_feat_norm

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*math.pi # [L] # ,device=cfg.device
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    


#LB
def load_kitti_poses(traj_idx):
    file_path = f'/home/scur2440/KITTI_odometry/dataset/poses/{traj_idx}.txt'
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


def load_kitti_times(traj_idx):
    file_path = f'/home/scur2440/KITTI_odometry/dataset/sequences/{traj_idx}/times.txt'
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            times.append(float(line.strip()))
    return torch.tensor(times)





if __name__ == "__main__":
    """
    overfit exps
    """
    # data
    # random sample 100 different c2ws

 

    for i in range(train_iters):
        est_c2ws = posenet.forward(times) # passing all times together to capture global structure in motion

        # dummy loss 
        loss = torch.abs(SE3_kitty - est_c2ws).mean() # L1 loss: compare se3 sampled from a smooth trajectory (GT) AND estimated se3 based on time alone
                                                        # if there is any statistical regularity in the motion and not just abrubt, random trajectory changes, t alone will be informative
                                                        # This research: makes single prediction for se(3). Thesis: predicts distribution over all se(3)s because motions could be equally likely
                                                        # This reserach - flaw of L2: for two possible positions, L2 loss takes the mean which might be not appropriate


        loss.backward()
        posenet.step()
        if i in [1, train_iters-1]:
            # print(f'SE3_kitty: {SE3_kitty.size()},\n{SE3_kitty[0]}')
            # print(f'est_c2ws: {est_c2ws.size()},\n{est_c2ws[0]}')
            print(f"step {i}, loss {loss}")

            