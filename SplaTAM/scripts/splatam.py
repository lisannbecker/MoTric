import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


### L
from _rotation_utils import *
from _prior_integration import PriorIntegration
from types import SimpleNamespace
import random
import gc
import easydict
torch.serialization.add_safe_globals([easydict.EasyDict])
from sklearn.neighbors import NearestNeighbors
from scipy.special import logsumexp



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store




### L additions
class CustomKDE:
	def __init__(self, data: np.ndarray,
				bandwidth: np.ndarray,
				range_factor: float = 1.0):
		"""
		A simple multivariate Gaussian KDE that uses per-dim bandwidth
		based on data range and sample count.

		Avoids zero bandwidth / NaNs

		Args:
			data:        shape (n_samples, n_dims)
			bandwidth:   optional user-supplied array of shape (n_dims,)
							if None, computed by rule-of-thumb:
							h_i = range_i * range_factor * n^{-1/(d+4)}
			range_factor: float multiplier on the data range before scaling
		"""
		data = np.asarray(data, dtype=float)
		if data.ndim != 2:
			raise ValueError("data must be 2D: (n_samples, n_dims)")
		self.X = data
		self.n, self.d = data.shape


		bw = np.asarray(bandwidth, float)
		if bw.ndim == 0:
			bw = np.full(self.d, bw)
		if bw.shape != (self.d,):
			raise ValueError("bandwidth must be scalar or shape (n_dims,)")
		self.h = bw

		# normalization constant: (2π)^{d/2} * ∏ h_i
		self._log_norm = (self.d/2) * np.log(2*np.pi) + np.sum(np.log(self.h))
		# we divide by n * norm, so log-denominator = log(n) + _log_norm
		self._log_denominator = np.log(self.n) + self._log_norm

	def logpdf(self, points: np.ndarray) -> np.ndarray:
		"""
		Evaluate log-density at each query point.

		Args:
			points: array of shape (m, d)

		Returns:
			logdens: array of shape (m,)
		"""
		X = self.X
		h = self.h
		pts = np.atleast_2d(points)
		m, d = pts.shape
		if d != self.d:
			raise ValueError(f"Expected {self.d}-dim points, got {d}")

		# shape (m, n_samples, d)
		diffs = (pts[:, None, :] - X[None, :, :]) / h[None, None, :]
		sq = -0.5 * np.sum(diffs**2, axis=-1)  # (m, n_samples)

		# log-sum-exp over samples
		lse = logsumexp(sq, axis=1)           # (m,)
		return lse - self._log_denominator

	def pdf(self, points: np.ndarray) -> np.ndarray:
		"""
		Evaluate density at each query point.

		Args:
			points: array of shape (m, d)

		Returns:
			dens: array of shape (m,)
		"""
		return np.exp(self.logpdf(points))

def prior_gradient_logl_custom(kde, pose, epsilon=1e-5):
    """
    Numerically approximate the gradient of the log likelihood at pose.
    Supports both scipy.stats.gaussian_kde and CustomKDE.
    Args:
        kde:    either a scipy gaussian_kde or an instance of CustomKDE
        pose:   1D numpy array of shape (d,)
        epsilon: finite-difference step
    Returns:
        grad: numpy array of shape (d,)
    """
    pose = pose.astype(float)
    d = pose.shape[0]
    grad = np.zeros(d, dtype=float)

    # helper to get log density
    def logpdf(x):
        x = np.atleast_2d(x)
        if hasattr(kde, 'logpdf'):
            return kde.logpdf(x)
        elif hasattr(kde, 'pdf'):
            # floor the pdf to avoid log(0)
            p = np.maximum(kde.pdf(x), 1e-12)
            return np.log(p)
        else:
            # scipy.gaussian_kde
            p = np.maximum(kde(x), 1e-12)
            return np.log(p)

    base = logpdf(pose)[0]
    for i in range(d):
        delta = np.zeros(d, float)
        delta[i] = epsilon

        lp = logpdf(pose + delta)[0]
        lm = logpdf(pose - delta)[0]

        grad[i] = (lp - lm) / (2 * epsilon)

    return grad

def filter_outliers(pts, radius=None, min_neighbors=1):
    """
    Remove points that have fewer than (min_neighbors) other points
    within `radius`. If radius is None, we set it to 1.5× the
    median pairwise distance.
    
    Args:
        pts: np.ndarray of shape (K, d)
        radius: float, max distance to count neighbors
        min_neighbors: int, minimum number of neighbors (excluding self)

    Returns:
        filtered_pts: np.ndarray of shape (M, d)
    """
    K, d = pts.shape
    if K <= min_neighbors+1:
        return pts.copy()  # too few points to filter
    
    # pick radius by default from median pairwise distance
    if radius is None:
        # fast approximate median: sample 100 pairs
        idx = np.random.choice(K, size=min(K,100), replace=False)
        sub = pts[idx]
        dists = np.linalg.norm(sub[:,None,:] - sub[None,:,:], axis=2)
        median_dist = np.median(dists)
        radius = 1.5 * median_dist

    # build neighbor lookup
    nbrs = NearestNeighbors(radius=radius).fit(pts)
    # for each point, count neighbors within radius (including self)
    neigh_indices = nbrs.radius_neighbors(pts, return_distance=False)
    mask = np.array([len(idxs)-1 >= min_neighbors for idxs in neigh_indices])
    return pts[mask]

def correct_with_kde_custom_motion_prior_translation_only(
    prior_integration,
    prior_config,
    params,
    time_idx,
    candidate_cam_unnorm_rot,  # [1,4] absolute SLAM pose quaternion
    candidate_cam_tran,        # [1,3] absolute SLAM pose translation
    device,
    bandwidth: float,
    lambda_step: float
    ):
    """
    Corrects the SLAM pose at time_idx using a CustomKDE prior built
    from the diffusion model's k-sample predictions.

    Only translation is corrected; rotation is left unchanged.

    Returns:
        corrected_rot: torch.Tensor [1,4]
        corrected_tran: torch.Tensor [1,3]
    """
    with torch.no_grad():
        # --- Gather past absolute poses and convert to relative ---
        start_idx = max(0, time_idx - prior_config.past_frames)
        # absolute history: [1,4,T] and [1,3,T]
        past_rots = params['cam_unnorm_rots'][..., start_idx:time_idx]  # [1,4,T]
        past_trans = params['cam_trans'][..., start_idx:time_idx]       # [1,3,T]

        # shape to [B,1,T,D]
        past_rots = past_rots.permute(0,2,1).unsqueeze(1)   # [1,1,T,4]
        past_trans = past_trans.permute(0,2,1).unsqueeze(1) # [1,1,T,3]
        past_poses = torch.cat([past_trans, past_rots], dim=-1)  # [1,1,T,7]

        # preprocess into model input
        B, traj_mask, past_traj = prior_integration.data_preprocess_past_poses(
            prior_config, past_poses, device
        )

        # --- Generate k-sample predictions ---
        samp_pred, mean_est, var_est = prior_integration.model_initializer(past_traj, traj_mask)
        samp_pred = (torch.exp(var_est/2)[...,None,None] * samp_pred /
                     samp_pred.std(dim=1).mean(dim=(1,2))[:,None,None,None])
        init_preds = samp_pred + mean_est[:,None]
        k_preds = prior_integration.p_sample_loop_accelerate(
            past_traj, traj_mask, init_preds
        )  # [B,K,T,7]

        # extract samples at first future step t=0
        k_t1 = k_preds[0,:,0,:].cpu().numpy()  # [K,7]
        # filter outliers in the sample cloud
        k_filtered = filter_outliers(k_t1, radius=None, min_neighbors=1)

        # --- Build CustomKDE with chosen bandwidth ---
        kde = CustomKDE(k_filtered, bandwidth)

        # --- Convert SLAM absolute pose to relative coordinates ---
        last_abs = past_poses[0,0,-1].cpu().numpy()  # [7]
        # absolute SLAM pose
        slam_abs = torch.cat([candidate_cam_tran[0], candidate_cam_unnorm_rot[0]], dim=-1)
        slam_abs_np = slam_abs.cpu().numpy()

        # relative translation and rotation
        traj_scale = prior_config.traj_scale
        rel_trans = (slam_abs_np[:3] - last_abs[:3]) / traj_scale
        # quaternion-relative
        last_quat = torch.from_numpy(last_abs[3:]).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        slam_quat = candidate_cam_unnorm_rot.unsqueeze(0).unsqueeze(2)  # shape [1,4]->[1,1,1,4]
        rel_quat = quaternion_relative(last_quat, slam_quat)[0,0,0].cpu().numpy()

        slam_rel = np.concatenate([rel_trans, rel_quat], axis=0)  # [7]

        # --- Compute gradient of log-pdf at slam_rel ---
        grad = kde.logpdf(slam_rel[np.newaxis, :])  # logpdf returns M array
        # Actually need gradient: use prior_integration.prior_gradient_logl
        grad = prior_gradient_logl_custom(kde, slam_rel)  # [7]
        # normalize and clip
        norm = np.linalg.norm(grad)
        if norm > 1.0:
            grad = grad / norm

        # --- Apply gradient step to translation only ---
        
        ##### Scale to match the step size of the live data #####
        # scaled_grad = prior_config.gradient_scale * grad[:3]
        # corrected_rel_trans = rel_trans + lambda_step * scaled_grad
        ########################################################

        corrected_rel_trans = rel_trans + lambda_step * grad[:3] #### original
        # rotation unchanged
        corrected_rel_quat = rel_quat

        # --- Convert back to absolute ---
        corrected_abs_trans = last_abs[:3] + corrected_rel_trans * traj_scale

        # pack outputs
        corrected_rot = candidate_cam_unnorm_rot  # [1,4]
        corrected_tran = torch.from_numpy(corrected_abs_trans).to(device).unsqueeze(0).float()

        return corrected_rot, corrected_tran

def correct_with_kde_custom_motion_prior(
    prior_integration,
    prior_config,
    params,
    time_idx,
    candidate_cam_unnorm_rot,  # [1,4] SLAM quaternion
    candidate_cam_tran,         # [1,3] SLAM translation
    device,
    bandwidth: float,
    lambda_step: float
    ):
    """
    Corrects both translation and rotation of the SLAM pose at time_idx
    using a 9-D CustomKDE prior built from the diffusion model’s k-sample predictions.

    Returns:
        corrected_rot: torch.Tensor [1,4]  # absolute quaternion
        corrected_tran: torch.Tensor [1,3] # absolute translation
    """
    with torch.no_grad():
        # --- 1) Gather past absolute SE(3) poses & build model input ---
        start_idx   = max(0, time_idx - prior_config.past_frames)
        abs_rots    = params['cam_unnorm_rots'][..., start_idx:time_idx]  # [1,4,T]
        abs_trans   = params['cam_trans'][...,      start_idx:time_idx]  # [1,3,T]

        # reshape to [B,1,T,7]
        abs_rots  = abs_rots.permute(0,2,1).unsqueeze(1)   # [1,1,T,4]
        abs_trans = abs_trans.permute(0,2,1).unsqueeze(1)  # [1,1,T,3]
        past_poses = torch.cat([abs_trans, abs_rots], dim=-1)  # [1,1,T,7]

        B, traj_mask, past_traj = prior_integration.data_preprocess_past_poses(
            prior_config, past_poses, device
        )

        # --- 2) Diffusion sampler: produce K futures (7-D each) ---
        samp_pred, mean_est, var_est = prior_integration.model_initializer(past_traj, traj_mask)
        samp_pred = (torch.exp(var_est/2)[...,None,None] * samp_pred /
                     samp_pred.std(dim=1).mean(dim=(1,2))[:,None,None,None])
        init_preds = samp_pred + mean_est[:,None]
        k_preds7 = prior_integration.p_sample_loop_accelerate(past_traj, traj_mask, init_preds)
        # shape: [1, K, T, 7]

        # --- 3) Build a 9-D KDE at the first future step t=0 ---
        samples7  = k_preds7[0, :, 0, :]                # [K,7]
        samples9  = pose_to_6d_translation(samples7)   # [K,9]
        samples9_np = samples9.cpu().numpy()
        filtered9   = filter_outliers(samples9_np)
        local_kde   = CustomKDE(filtered9, bandwidth=np.full(9, bandwidth), range_factor=1.0)

        # --- 4) Convert SLAM absolute → relative 7-D pose at t=0 ---
        last_pose   = past_poses[0,0,-1].cpu().numpy()          # [7]
        slam_abs7   = torch.cat([candidate_cam_tran[0], candidate_cam_unnorm_rot[0]], dim=-1)
        slam_abs_np = slam_abs7.cpu().numpy()                   # [7]

        # relative translation
        rel_t = (slam_abs_np[:3] - last_pose[:3]) / prior_config.traj_scale
        # relative quaternion
        last_q = torch.from_numpy(last_pose[3:]).to(device).view(1,1,4)
        slam_q = candidate_cam_unnorm_rot.view(1,1,4)
        rel_q  = quaternion_relative(last_q, slam_q)[0,0].cpu().numpy()  # [4]

        slam_rel7 = np.concatenate([rel_t, rel_q], axis=0)      # [7]

        # --- 5) Embed SLAM rel-pose 7→9, compute ∇ log p in ℝ⁹ ---
        pose9_t  = pose_to_6d_translation(torch.from_numpy(slam_rel7).to(device))
        pose9_np = pose9_t.cpu().numpy()                       # [9]
        grad9    = prior_gradient_logl_custom(local_kde, pose9_np)  # [9]
        norm9    = np.linalg.norm(grad9)
        if norm9 > 1.0:
            grad9 /= norm9

        # --- 6) Take a gradient step in ℝ⁹ ---
        corrected9 = pose9_np + lambda_step * grad9           # [9]

        # --- 7) Map corrected 9-D back → relative 7-D SE(3) ---
        corr9_t   = torch.from_numpy(corrected9).to(device)
        corr7_t   = convert_6d_back_to_original(corr9_t, rotation_type=7)  # [7]
        corr_rel_t = corr7_t[:3].cpu().numpy()
        corr_rel_q = corr7_t[3:].cpu().numpy()

        # --- 8) Reconstruct corrected absolute pose ---
        abs_t = last_pose[:3] + corr_rel_t * prior_config.traj_scale
        abs_q = quaternion_multiply(last_q.view(4), torch.from_numpy(corr_rel_q))

        corrected_tran = torch.from_numpy(abs_t).to(device).view(1,3).float()
        corrected_rot  = abs_q.to(device).view(1,4).float()

        return corrected_rot, corrected_tran

def plot_trajectory(gt_w2c_all, cam_rots, cam_trans, out_path_plot):
    # make sure output directory exists
    os.makedirs(os.path.dirname(out_path_plot), exist_ok=True)

    T = gt_w2c_all.shape[0]
    gt_pos = gt_w2c_all[:, :3, 3]
    gt_xy  = gt_pos[:, [0, 2]]
    
    est_pos = []
    for t in range(T):
        q = cam_rots[..., t]
        q = F.normalize(q, dim=0)
        R = build_rotation(q)
        tvec = cam_trans[..., t].detach().cpu().numpy().reshape(3)
        est_pos.append(tvec)
    est_pos = np.stack(est_pos, 0)
    est_xy  = est_pos[:, [0, 2]]
    
    plt.figure(figsize=(6,6))
    plt.plot(gt_xy[:,0], gt_xy[:,1],   label="Ground Truth")
    plt.plot(est_xy[:,0], est_xy[:,1], label="Estimated")
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Top-down Trajectory (X vs Z)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # use the passed-in path!
    plt.savefig(out_path_plot, dpi=200)
    plt.show()

    print(f"\n=== Trajectory plot saved to {out_path_plot} ===")


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    

    ##### Init prior methods
    progress_bars = False

    PRIOR_MAIN = "tumrgb_prior" #pedestrian_prior, kitti_prior, tumrgb_prior, None
    CORRECTION_TYPE = "KDECustom" # GMM, KDE, KDECustom 
    bandwidth=0.04
    lambda_step=0.001
    
    end_early = None #int or None
    first_10_GT = True


    if PRIOR_MAIN == "pedestrian_prior":
        config_dict = { #TODO get dynamically from config file
            "checkpoint_path": "/home/lbecker/MoTric/7_LED_Prior/results/7_1_PedestrianPrior_10in_15out_k30/7_1_PedestrianPrior_10in_15out_k30/models/best_checkpoint_epoch_41.pth", 
            "traj_scale": 1,
            "gradient_scale": 1.0, #use if prior training data has different step size to slam data
            "past_frames": 10,
            "future_frames": 15,
            "dimensions": 7,
            "k_preds": 30,
            "diffusion": {
                "steps": 100,
                "beta_schedule": "linear",
                "beta_start": 1e-5,
                "beta_end": 1e-2,
            },
            "relative": False,
            "normalised": False,
            "overfitting": False,
            "selected_trajectories": None,
        }
    elif PRIOR_MAIN == "kitti_prior":
        config_dict = { 
            "checkpoint_path": "/home/lbecker/MoTric/7_LED_Prior/results/7_1_KittiPrior_10in_20out_k30/7_1_KittiPrior_10in_20out_k30_Overfit/models/best_checkpoint_epoch_98.pth",
            # "checkpoint_path": "/home/lbecker/MoTric/7_LED_Prior/results/7_1_KittiPrior_10in_20out_k30/7_1_KittiPrior_10in_20out_k30_EXCL04/models/best_checkpoint_epoch_27.pth",
            "traj_scale": 1,
            "gradient_scale": 1.0, #use if prior training data has different step size to slam data
            "past_frames": 10,
            "future_frames": 20,
            "dimensions": 7,
            "k_preds": 30,
            "diffusion": {
                "steps": 150,
                "beta_schedule": "linear",
                "beta_start": 1.e-4,
                "beta_end": 5.e-2,
            },
            "relative": False,
            "normalised": False,
            "overfitting": True,
            "selected_trajectories": None,
        }
    elif PRIOR_MAIN == "tumrgb_prior":
        config_dict = { 
            ###overfit
            "checkpoint_path": "/home/lbecker/MoTric/7_LED_Prior/results/7_2_TUMRGBPrior_10in_15out_k30/7_2_TUMRGBPrior_10in_15out_k30_Desk1_Overfit/models/best_checkpoint_epoch_57.pth",
            ###all tum sequence, not overfit
            # "checkpoint_path": "/home/lbecker/MoTric/7_LED_Prior/results/7_2_TUMRGBPrior_10in_15out_k30/7_2_TUMRGBPrior_10in_15out_k30_All_Sequences/models/best_checkpoint_epoch_34.pth",
            "traj_scale": 1,
            "gradient_scale": 1.0, #use if prior training data has different step size to slam data
            "past_frames": 10,
            "future_frames": 15,
            "dimensions": 7,
            "k_preds": 30,
            "diffusion": {
                "steps": 150,
                "beta_schedule": "linear",
                "beta_start": 1.e-4,
                "beta_end": 5.e-2,
            },
            "relative": False,
            "normalised": False,
            "overfitting": True,
            "selected_trajectories": None,
        }


    
    #intialise prior
    if PRIOR_MAIN:
        def dict_to_namespace(d):
            return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

        prior_config = dict_to_namespace(config_dict)
        print('PRIOR CONFIG:\n', prior_config)
        prior_integration = PriorIntegration(prior_config)

        prior_integration.load_checkpoint(prior_config.checkpoint_path)

        prior_integration.model.eval()
        prior_integration.model_initializer.eval()

        # original_params_storage = {
        #     'cam_unnorm_rots': None,
        #     'cam_trans': None,
        #     'initialized': False
        # }
        
    # Optionally, set  random seed for reproducibility: XXX needed? part of LED eval
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)

            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    



    ### Iterate over Scan
    if end_early: #for testing
        num_frames = end_early

    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])



        # ------------------- Tracking ---------------------
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            if progress_bars:
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation 
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()

                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation << to be corrected by motion prior
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran


                if PRIOR_MAIN:
                    ### ============= use GT as first 10 poses (experiment) ================
                    if first_10_GT and time_idx < prior_config.past_frames: 
                        # Override raw SLAM estimates for first K frames with GT poses
                        gt_rel = curr_gt_w2c[-1]  # last appended GT for this time_idx
                        # Extract translation and quaternion
                        gt_tran = gt_rel[:3, 3].detach()
                        gt_quat = matrix_to_quaternion(gt_rel[:3, :3].unsqueeze(0)).detach().squeeze(0)
                        # Overwrite SLAM parameters
                        params['cam_trans'] = params['cam_trans'].clone()
                        params['cam_unnorm_rots'] = params['cam_unnorm_rots'].clone()
                        params['cam_trans'][..., time_idx] = gt_tran
                        params['cam_unnorm_rots'][..., time_idx] = gt_quat
                        print(f'Overwriting SLAM pose with GT for time_idx {time_idx}')
                        continue
                    ### ===================================================================

                    ###motion prior implementation - uses past 10 poses #make config TODO
                    if time_idx >= prior_config.past_frames:
                        #save pose estimates (incl before correction) for visualisation
                        # if not original_params_storage['initialized']:
                        #     # First time - initialize storage with all current parameters
                        #     original_params_storage['cam_unnorm_rots'] = params['cam_unnorm_rots'].clone()
                        #     original_params_storage['cam_trans'] = params['cam_trans'].clone()
                        #     original_params_storage['initialized'] = True

                        # else:
                        #     # Update storage with current frame's original estimate
                        #     original_params_storage['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot.clone()
                        #     original_params_storage['cam_trans'][..., time_idx] = candidate_cam_tran.clone()

                        if CORRECTION_TYPE == "KDE":
                            corrected_rot, corrected_tran = correct_with_kde_motion_prior(prior_integration, prior_config, params, time_idx, candidate_cam_unnorm_rot, candidate_cam_tran, device, lambda_step=0.1)
                        elif CORRECTION_TYPE =="GMM":
                            corrected_rot, corrected_tran, correction_weight = correct_with_gmm_motion_prior(prior_integration, prior_config, params, time_idx, candidate_cam_unnorm_rot, candidate_cam_tran, device, max_correction_weight=0.1)
                        elif CORRECTION_TYPE == "KDECustom":
                            corrected_rot, corrected_tran = correct_with_kde_custom_motion_prior_translation_only(prior_integration, prior_config, params, time_idx, candidate_cam_unnorm_rot, candidate_cam_tran, device, bandwidth, lambda_step)

                        print('Candidate Pose:', torch.cat([candidate_cam_tran, candidate_cam_unnorm_rot],dim=1).detach().cpu().numpy().squeeze(0))
                        print('Corrected Pose:', torch.cat([corrected_tran, corrected_rot],dim=1).detach().cpu().numpy().squeeze(0))
                        if CORRECTION_TYPE == "GMM":
                            print('correction_weight:', correction_weight)

                        corrected_rot = F.normalize(corrected_rot, p=2, dim=-1) #correct to unit-sphere (left due to the correction)

                        # 3) overwrite the SLAM state with your smoothed estimate
                        params['cam_unnorm_rots'][..., time_idx] = corrected_rot
                        params['cam_trans'][...,       time_idx] = corrected_tran
                        # torch.cuda.empty_cache()
                        # del corrected_rot, corrected_tran
                    # else:
                    #     if not original_params_storage['initialized']:
                    #         original_params_storage['cam_unnorm_rots'] = params['cam_unnorm_rots'].clone()
                    #         original_params_storage['cam_trans'] = params['cam_trans'].clone()
                    #         original_params_storage['initialized'] = True
                    #     else:
                    #         original_params_storage['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot.clone()
                    #         original_params_storage['cam_trans'][..., time_idx] = candidate_cam_tran.clone()
            
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran

        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                if progress_bars:
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 




            # ------------------------ Mapping ------------------------
            mapping_start_time = time.time()
            
            if num_iters_mapping > 0 and progress_bars:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        if time_idx % 10 == 0:  # Every 10 frames
            gc.collect()
            
        # Delete large variables to free memory
        if 'color' in locals():
            del color
        if 'depth' in locals(): 
            del depth

        torch.cuda.empty_cache()



    

    # ─────────────────────────────────────────────────────────────
    ### Visualisation code TODO make config
    print("\n=== Generating Trajectory Visualization ===")
    gt_w2c_all = torch.stack(gt_w2c_all_frames, dim=0).cpu().detach().numpy() #(T,4,4)
    cam_rots   = params['cam_unnorm_rots']            # torch tensor (1,4,T)
    cam_trans  = params['cam_trans']                  # torch tensor (1,3,T)
    
    
    # Save out the raw & corrected pose‐trans tracks
    early_stop = str(end_early)+'_' if end_early else ''
    suffix = f"corr_{bandwidth}b_{lambda_step}l" if PRIOR_MAIN else "uncorr"
    title = prior_config.checkpoint_path.split('/')[-3] if PRIOR_MAIN else "Baseline" #/home/lbecker/MoTric/7_LED_Prior/results/7_2_TUMRGBPrior_10in_15out_k30/7_2_TUMRGBPrior_10in_15out_k30_Desk1_Overfit/models/best_checkpoint_epoch_57.pth
    
    out_path_plot = os.path.join(eval_dir, f"plots/{early_stop}poses_{title}_{suffix}_customkde.png")
    plot_trajectory(gt_w2c_all, cam_rots, cam_trans, out_path_plot)

    out_path = os.path.join(eval_dir, f"plots/{early_stop}poses_{title}_{suffix}_customkde.npz")
    rots_np  = cam_rots.squeeze(0).detach().cpu().numpy().T    # (T,4) quaternion [qx,qy,qz,qw]
    trans_np = cam_trans.squeeze(0).detach().cpu().numpy().T   # (T,3) translation [x,y,z]
    slam_combined = np.concatenate([rots_np, trans_np], axis=1)  # shape (T,7)

    # 3) save both SLAM and ground‐truth
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        slam = slam_combined,
        gt   = gt_w2c_all
    )
    print(f"Saved {'corrected' if PRIOR_MAIN else 'raw'} poses → {out_path}")
    # ─────────────────────────────────────────────────────────────

    
    # Convert gt_w2c_all_frames to numpy if it's not already
    # if isinstance(gt_w2c_all_frames[0], torch.Tensor):
    #     gt_w2c_all = torch.stack(gt_w2c_all_frames, dim=0).cpu().numpy()
    # else:
    #     gt_w2c_all = np.stack(gt_w2c_all_frames, axis=0)
    
    # Create visualization comparing original estimate vs corrected vs GT trajectories    
    # if PRIOR_MAIN and original_params_storage['initialized']:
    #     # Plot comparison with motion prior correction
    #     plot_trajectory_with_prior_correction_tracking(
    #         params, 
    #         original_params_storage, 
    #         gt_w2c_all,
    #         save_dir=eval_dir
    #     )
    # else:
    #     # Plot just current estimates vs GT
    #     plot_trajectory_with_prior_correction_tracking(
    #         params, 
    #         None, 
    #         gt_w2c_all,
    #         save_dir=eval_dir
    #     )


    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)