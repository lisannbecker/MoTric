
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
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pickle
from _rotation_utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #LoaderKitti is two levels up
from PoseLoaderCustom import LoadDatasetLeapfrog, seq_collate_custom

#from trainer.kde_utils import find_max
torch.set_printoptions(precision=6, sci_mode=False)

from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

NUM_Tau = 5

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import joblib


import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import joblib

PRIOR_PATH = './prior_clusters/kitti_kde'

def extract_flat_trajectory(past_traj_tensor):
    """
    Flatten a batch of past trajectories into feature vectors.

    Args:
        past_traj_tensor (torch.Tensor): shape (B, T_past, 7)

    Returns:
        np.ndarray: shape (B, T_past * 7)
    """
    # Move to CPU and convert to numpy
    arr = past_traj_tensor.detach().cpu().numpy()
    B, T, D = arr.shape
    return arr.reshape(B, T * D)

class OfflineTrajectoryClusterer:
    def __init__(self,
                 trainer,
                 num_clusters=4,
                 pca_dim=20,
                 save_path="trajectory_cluster_model.pkl",
                 random_state=42):
        """
        Performs offline clustering of KITTI trajectories (7D poses) using PCA + KMeans.

        Args:
            trainer: Trainer instance with .train_loader and .data_preprocess()
            num_clusters (int): number of clusters to form
            pca_dim (int): dimensionality for PCA reduction
            save_path (str): file to save PCA+KMeans model
            random_state (int): random seed for reproducibility
        """
        self.trainer = trainer
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.save_path = save_path
        self.random_state = random_state

        # placeholders
        self.pca = None
        self.kmeans = None

    def gather_features(self, loader):
        """
        Iterate over loader and collect flattened past trajectories.

        Returns:
            np.ndarray: concatenated features of shape (N_total, T_past*7)
        """
        all_feats = []
        for batch_idx, data in enumerate(loader):
            batch_size, traj_mask, past_traj, fut_traj = self.trainer.data_preprocess(data)
            feats = extract_flat_trajectory(past_traj)
            all_feats.append(feats)
        if not all_feats:
            raise ValueError("No data found in loader to cluster.")
        return np.vstack(all_feats)

    def fit(self):
        """
        Perform PCA followed by KMeans clustering on the KITTI dataset.
        Saves the fitted models to disk.
        """
        # 1) Gather features from training set
        print("[Clustering] Gathering features from train_loader...")
        X = self.gather_features(self.trainer.train_loader)
        print(f"[Clustering] Collected {X.shape[0]} trajectories with dim {X.shape[1]}")

        # 2) PCA dimensionality reduction
        print(f"[Clustering] Performing PCA to {self.pca_dim} dimensions...")
        self.pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
        X_reduced = self.pca.fit_transform(X)

        # 3) KMeans clustering
        print(f"[Clustering] Fitting KMeans with {self.num_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.kmeans.fit(X_reduced)

        # 4) Save model
        model_dict = {
            'pca': self.pca,
            'kmeans': self.kmeans,
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"[Clustering] Saved PCA+KMeans model to {self.save_path}")

    def load(self):
        """
        Load PCA+KMeans model from disk.
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Model file {self.save_path} not found.")
        with open(self.save_path, 'rb') as f:
            model_dict = pickle.load(f)
        self.pca = model_dict['pca']
        self.kmeans = model_dict['kmeans']
        print(f"[Clustering] Loaded PCA+KMeans model from {self.save_path}")

    def predict_cluster(self, past_traj_tensor):
        """
        Given a batch of past trajectories, assign each to a cluster.

        Args:
            past_traj_tensor (torch.Tensor): shape (B, T_past, 7)

        Returns:
            np.ndarray: cluster labels shape (B,)
        """
        feats = extract_flat_trajectory(past_traj_tensor)
        X_red = self.pca.transform(feats)
        return self.kmeans.predict(X_red)

class KittiKdePriorOld:
    """
    Build and apply a motion prior using per-mode KDE bandwidths on KITTI trajectories.
    Assumes translation+rotation state D, but bands are on translation deltas.
    """
    def __init__(self, dataloader, preprocess_fn, n_clusters=5,
                 output_dir=PRIOR_PATH, generate_cluster_vis=False):
        self.dataloader = dataloader
        self.preprocess_fn = preprocess_fn
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.generate_cluster_vis = generate_cluster_vis
        os.makedirs(self.output_dir, exist_ok=True)
        self.cluster_model = None
        self.cluster_bandwidths = None
        self._deltas = None
        self._labels = None

    def fit_clusters_and_bandwidths(self):
        # Extract ground-truth next-step deltas
        deltas = []
        for batch in self.dataloader:
            _, _, past, fut = self.preprocess_fn(batch)
            # compute delta from last past to first future (translation)
            gt = (fut[:, 0, :3] - past[:, -1, :3]).cpu().numpy()
            deltas.append(gt)
        deltas = np.vstack(deltas)
        # Cluster on raw deltas
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(deltas)
        labels = kmeans.labels_
        # Compute per-cluster Silverman bandwidths
        bandwidths = {}
        for i in range(self.n_clusters):
            d_i = deltas[labels == i]
            sigma = np.std(d_i)
            n_i = d_i.shape[0]
            bandwidths[i] = 1.06 * sigma * (n_i ** (-1/5))
        # Save models
        joblib.dump(kmeans, os.path.join(self.output_dir, 'kmeans_model.pkl'))
        joblib.dump(bandwidths, os.path.join(self.output_dir, 'bandwidths.pkl'))
        self.cluster_model = kmeans
        self.cluster_bandwidths = bandwidths
        self._deltas = deltas
        self._labels = labels

    def load_models(self):
        self.cluster_model = joblib.load(os.path.join(self.output_dir, 'kmeans_model.pkl'))
        self.cluster_bandwidths = joblib.load(os.path.join(self.output_dir, 'bandwidths.pkl'))

    def assign_cluster(self, past_traj):
        # compute delta feature: last past to first future not available; instead use cluster on next-step deltas directly
        # for testing, approximate delta prior: use mean last-step from past to last past+prediction? here, we cluster on past delta
        # simple: use last displacement vector's angle to assign cluster nearest centroid
        disp = (past_traj[:, -1, :3] - past_traj[:, -2, :3]).cpu().numpy()
        labels = self.cluster_model.predict(disp)
        return labels

    def get_bandwidths(self, clusters):
        return np.array([self.cluster_bandwidths[int(c)] for c in clusters])

class CustomKDE:
	def __init__(self,
					data: np.ndarray,
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

class MotionPriorGMM:
    def __init__(self, max_components=5, min_samples_per_component=3, rotation_type='none'):
        """
        Initialize Motion Prior with GMM
        
        Args:
            max_components: Maximum number of GMM components
            min_samples_per_component: Minimum samples needed per component
            rotation_type: Type of rotation representation ('none', 'quaternion', '6d', 'lie')
        """
        self.max_components = max_components
        self.min_samples_per_component = min_samples_per_component
        self.rotation_type = rotation_type
        self.gmm = None
        self.fitted = False
        
    def pose_to_6d_translation(self, pose, rotation_type=None):
        """Convert pose to [tx, ty, tz, r1, r2, r3, r4, r5, r6] format"""
        if rotation_type is None:
            rotation_type = self.rotation_type
            
        if rotation_type == 'none':
            return pose  # assume already [tx, ty, tz]
        elif rotation_type == 'quaternion':
            # Convert quat to 6D rotation
            quat = pose[..., 3:7]  # [qw, qx, qy, qz]
            rot_matrix = self.quaternion_to_rotation_matrix(quat)
            rot_6d = self.rotation_matrix_to_6d(rot_matrix)
            return torch.cat([pose[..., :3], rot_6d], dim=-1)
        elif rotation_type == '6d':
            return pose  # already in correct format
        elif rotation_type == 'lie':
            # Convert axis-angle to 6D
            rot_matrix = self.axis_angle_to_rotation_matrix(pose[..., 3:6])
            rot_6d = self.rotation_matrix_to_6d(rot_matrix)
            return torch.cat([pose[..., :3], rot_6d], dim=-1)
        else:
            return pose  # fallback to original
    
    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        qx, qy, qz, qw = quat.unbind(-1)
        
        xx = qx * qx
        yy = qy * qy  
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        
        R = torch.stack([
            torch.stack([1 - 2*yy - 2*zz, 2*xy - 2*wz, 2*xz + 2*wy], dim=-1),
            torch.stack([2*xy + 2*wz, 1 - 2*xx - 2*zz, 2*yz - 2*wx], dim=-1),
            torch.stack([2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*xx - 2*yy], dim=-1)
        ], dim=-2)
        
        return R
    
    def rotation_matrix_to_6d(self, R):
        """Convert rotation matrix to 6D representation"""
        return R[..., :3, :2].reshape(*R.shape[:-2], 6)
    
    def axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix using Rodrigues formula"""
        theta = axis_angle.norm(dim=-1, keepdim=True)
        theta = theta.clamp(min=1e-6)
        k = axis_angle / theta
        
        zeros = torch.zeros_like(k[..., 0])
        k_cross = torch.stack([
            torch.stack([zeros, -k[..., 2], k[..., 1]], dim=-1),
            torch.stack([k[..., 2], zeros, -k[..., 0]], dim=-1),
            torch.stack([-k[..., 1], k[..., 0], zeros], dim=-1)
        ], dim=-2)
        
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(k_cross.shape)
        sin_theta = torch.sin(theta)[..., None]
        cos_theta = torch.cos(theta)[..., None]
        
        R = I + sin_theta * k_cross + (1 - cos_theta) * torch.matmul(k_cross, k_cross)
        return R
        
    def fit_distribution(self, k_samples_6d):
        """
        Fit GMM distribution to k samples with outlier-robust approach
        
        Args:
            k_samples_6d: [K, D] array of pose samples in 6D+translation format
            
        Returns:
            fitted GMM model
        """
        if isinstance(k_samples_6d, torch.Tensor):
            k_samples_6d = k_samples_6d.detach().cpu().numpy()
            
        K = k_samples_6d.shape[0]
        
        # More conservative component selection to avoid outlier components
        # Use stricter requirements for additional components
        n_components = min(
            self.max_components,
            max(1, K // (self.min_samples_per_component * 2))  # Doubled requirement
        )
        
        # Try different numbers of components and select best using BIC/AIC
        best_gmm = None
        best_score = float('inf')
        
        for n_comp in range(1, n_components + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type='full',
                    reg_covar=1e-6,
                    max_iter=100,
                    random_state=42
                )
                gmm.fit(k_samples_6d)
                
                # Use BIC to penalize models with too many components
                # BIC penalizes complexity more than AIC - better for avoiding outlier components
                score = gmm.bic(k_samples_6d)
                
                if score < best_score:
                    best_score = score
                    best_gmm = gmm
                    
            except Exception as e:
                print(f"Failed to fit GMM with {n_comp} components: {e}")
                continue
        
        if best_gmm is None:
            # Fallback to single component
            print("All GMM fits failed, using single component")
            best_gmm = GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-6)
            best_gmm.fit(k_samples_6d)
        
        self.gmm = best_gmm
        self.fitted = True
        
        # Post-process to merge components with very low weights
        self._merge_low_weight_components()
        
        return self.gmm
    
    def _merge_low_weight_components(self, min_weight_threshold=0.1):
        """
        Remove components with very low weights that likely represent outliers
        """
        if not self.fitted or self.gmm.n_components <= 1:
            return
            
        # Identify components with weights above threshold (keep these)
        high_weight_mask = self.gmm.weights_ >= min_weight_threshold
        n_high_weight = np.sum(high_weight_mask)
        n_low_weight = self.gmm.n_components - n_high_weight
        
        print(f"Components below weight threshold {min_weight_threshold}: {n_low_weight}/{self.gmm.n_components}")
        
        if n_low_weight > 0 and n_high_weight > 0:
            # Create new GMM with only high-weight components
            try:
                new_gmm = GaussianMixture(
                    n_components=n_high_weight,
                    covariance_type='full',
                    reg_covar=1e-6,
                    max_iter=100,
                    random_state=42
                )
                
                # Set the fitted state and convergence
                new_gmm.converged_ = True
                new_gmm.n_iter_ = self.gmm.n_iter_
                new_gmm.lower_bound_ = self.gmm.lower_bound_
                
                # Copy parameters from high-weight components
                new_gmm.weights_ = self.gmm.weights_[high_weight_mask]
                new_gmm.weights_ = new_gmm.weights_ / new_gmm.weights_.sum()  # Renormalize
                new_gmm.means_ = self.gmm.means_[high_weight_mask]
                new_gmm.covariances_ = self.gmm.covariances_[high_weight_mask]
                
                # Compute precision matrices if needed for density computation
                from scipy.linalg import inv
                precisions = []
                precisions_chol = []
                
                for cov in new_gmm.covariances_:
                    # Compute precision matrix (inverse of covariance)
                    precision = inv(cov)
                    precisions.append(precision)
                    
                    # Compute Cholesky decomposition of precision
                    from scipy.linalg import cholesky
                    try:
                        prec_chol = cholesky(precision, lower=True)
                        precisions_chol.append(prec_chol)
                    except:
                        # Fallback if cholesky fails
                        prec_chol = np.linalg.cholesky(precision)
                        precisions_chol.append(prec_chol)
                
                new_gmm.precisions_ = np.array(precisions)
                new_gmm.precisions_cholesky_ = np.array(precisions_chol)
                
                print(f"Removed low-weight components: {self.gmm.n_components} -> {new_gmm.n_components} components")
                print(f"New weights: {new_gmm.weights_}")
                
                self.gmm = new_gmm
                
            except Exception as e:
                print(f"Failed to remove low-weight components: {e}")
                print("Keeping original GMM")
                # Keep original GMM
        elif n_high_weight == 0:
            print("All components below threshold, keeping original GMM")
        else:
            print("No components to remove")
    
    def build_motion_prior_from_k_preds(self, k_samples):
        """
        Build motion prior from k predictions for a specific pose
        
        Args:
            k_samples: [K, D] tensor of k predictions for a single pose
            
        Returns:
            Self (fitted GMM model)
        """
        K, D = k_samples.shape
        
        # Convert to 6D representation if needed
        k_samples_6d = torch.stack([
            self.pose_to_6d_translation(sample) for sample in k_samples
        ])  # [K, 6 or 9]
        
        # Fit GMM to these K samples
        self.fit_distribution(k_samples_6d)
        
        return self
    
    def compute_correction_weight(self, slam_pose_6d, max_weight=0.3):
        """
        Compute how much to weight the motion prior vs SLAM
        
        Args:
            slam_pose_6d: SLAM pose in 6D format
            max_weight: Maximum correction weight
            
        Returns:
            weight in [0, max_weight] where 0 = trust SLAM fully
        """
        if not self.fitted:
            return 0.0
            
        if isinstance(slam_pose_6d, torch.Tensor):
            slam_pose_6d = slam_pose_6d.detach().cpu().numpy()
            
        # Get probability density at SLAM pose
        log_prob = self.gmm.score_samples(slam_pose_6d.reshape(1, -1))[0]
        
        # Get max probability from the fitted distribution
        max_log_prob = self.gmm.score_samples(self.gmm.means_).max()
        
        # Convert to weight (high probability = low correction weight)
        normalized_prob = np.exp(log_prob - max_log_prob)  # normalize to [0, 1]
        weight = max_weight * (1 - normalized_prob)
        
        return np.clip(weight, 0, max_weight)
    
    def visualize_motion_prior(self, k_samples, past_traj=None, gt_future=None, 
                              experiment_name="test", batch_idx=0, timestep=0, 
                              save_path="./visualization", show_plot=False):
        """
        Visualize the motion prior GMM with k_samples, past trajectory, and GT future
        
        Args:
            k_samples: [K, D] tensor of k predictions for the specific pose
            past_traj: [T_past, D] tensor of past trajectory points (optional)
            gt_future: [1, D] or [D] tensor of ground truth future pose (optional)
            experiment_name: name for saving the plot
            batch_idx: batch index for naming
            timestep: timestep for naming
            save_path: directory to save the plot
            show_plot: whether to display the plot
        """
        if not self.fitted:
            print("Cannot visualize: GMM not fitted yet")
            return
            
        # Convert tensors to numpy and print shapes for debugging
        if isinstance(k_samples, torch.Tensor):
            k_samples_np = k_samples.detach().cpu().numpy()
        else:
            k_samples_np = k_samples
        print(f"DEBUG: k_samples shape: {k_samples_np.shape}")
            
        if past_traj is not None:
            if isinstance(past_traj, torch.Tensor):
                past_np = past_traj.detach().cpu().numpy()
            else:
                past_np = past_traj
            print(f"DEBUG: past_traj shape: {past_np.shape}")
        else:
            past_np = None
            
        if gt_future is not None:
            if isinstance(gt_future, torch.Tensor):
                gt_np = gt_future.detach().cpu().numpy()
            else:
                gt_np = gt_future
            print(f"DEBUG: gt_future shape: {gt_np.shape}")
            # Handle both [1, D] and [D] shapes
            if gt_np.ndim == 1:
                gt_np = gt_np.reshape(1, -1)  # Convert [D] to [1, D]
        else:
            gt_np = None
        
        # Only use x, y coordinates for 2D visualization
        k_samples_2d = k_samples_np[:, :2]  # [K, 2]
        print(f"DEBUG: k_samples_2d shape: {k_samples_2d.shape}")
        print(f"DEBUG: k_samples_2d range - X: [{k_samples_2d[:, 0].min():.3f}, {k_samples_2d[:, 0].max():.3f}], Y: [{k_samples_2d[:, 1].min():.3f}, {k_samples_2d[:, 1].max():.3f}]")
        
        # Combine all points for dynamic grid limits
        points_for_grid = [k_samples_2d]
        if past_np is not None:
            past_2d = past_np[:, :2]
            points_for_grid.append(past_2d)
            print(f"DEBUG: past_2d shape: {past_2d.shape}")
            print(f"DEBUG: past_2d range - X: [{past_2d[:, 0].min():.3f}, {past_2d[:, 0].max():.3f}], Y: [{past_2d[:, 1].min():.3f}, {past_2d[:, 1].max():.3f}]")
            
        if gt_np is not None:
            gt_2d = gt_np[:, :2] 
            points_for_grid.append(gt_2d)
            print(f"DEBUG: gt_2d shape: {gt_2d.shape}")
            print(f"DEBUG: gt_2d values: {gt_2d}")
            
        all_points = np.concatenate(points_for_grid, axis=0)
        print(f"DEBUG: all_points shape: {all_points.shape}")
        
        # Set up grid with margin
        margin = 0.5  # Increased margin
        min_x = all_points[:, 0].min() - margin
        max_x = all_points[:, 0].max() + margin
        min_y = all_points[:, 1].min() - margin
        max_y = all_points[:, 1].max() + margin
        
        print(f"DEBUG: Grid bounds - X: [{min_x:.3f}, {max_x:.3f}], Y: [{min_y:.3f}, {max_y:.3f}]")
        
        # Create grid for contour plot
        x_grid, y_grid = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T  # [10000, 2]
        
        # For GMM evaluation, we need to match the dimensionality it was trained on
        # Create samples in the same format as the GMM was fitted on
        gmm_input_dim = self.gmm.means_.shape[1]
        print(f"DEBUG: GMM input dimension: {gmm_input_dim}")
        
        if gmm_input_dim > 2:
            # Pad grid points with zeros for extra dimensions or use mean values
            extra_dims = gmm_input_dim - 2
            # Use the mean values from the k_samples for the extra dimensions
            extra_dim_means = k_samples_np[:, 2:].mean(axis=0) if k_samples_np.shape[1] > 2 else np.zeros(extra_dims)
            grid_points_extended = np.hstack([
                grid_points, 
                np.tile(extra_dim_means, (grid_points.shape[0], 1))
            ])
        else:
            grid_points_extended = grid_points
            
        print(f"DEBUG: grid_points_extended shape: {grid_points_extended.shape}")
        
        # Get density from GMM - both overall and individual components
        try:
            log_density = self.gmm.score_samples(grid_points_extended)
            density = np.exp(log_density).reshape(x_grid.shape)
            print(f"DEBUG: Density range: [{density.min():.6f}, {density.max():.6f}]")
            
            # Get individual component densities
            component_densities = []
            for i in range(self.gmm.n_components):
                # Create a GMM with only this component
                single_gmm = GaussianMixture(n_components=1, covariance_type='full')
                single_gmm.weights_ = np.array([1.0])
                single_gmm.means_ = self.gmm.means_[i:i+1]
                single_gmm.covariances_ = self.gmm.covariances_[i:i+1]
                single_gmm.precisions_ = self.gmm.precisions_[i:i+1] if hasattr(self.gmm, 'precisions_') else None
                single_gmm.precisions_cholesky_ = self.gmm.precisions_cholesky_[i:i+1] if hasattr(self.gmm, 'precisions_cholesky_') else None
                
                # Get density for this component
                component_log_density = single_gmm.score_samples(grid_points_extended)
                component_density = np.exp(component_log_density).reshape(x_grid.shape)
                # Weight by the component's mixture weight
                component_density *= self.gmm.weights_[i]
                component_densities.append(component_density)
                
        except Exception as e:
            print(f"Error computing GMM density: {e}")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot overall density as background
        if density.max() > density.min():
            main_contour = plt.contourf(x_grid, y_grid, density, levels=15, 
                                      cmap="viridis", alpha=0.6)
            plt.colorbar(main_contour, label="Overall Probability Density", shrink=0.8)
        
        # Plot individual GMM components as separate contours with different colors
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        for i, comp_density in enumerate(component_densities):
            if comp_density.max() > comp_density.min():
                # Use different colormaps for different components
                cmap = colors[i % len(colors)]
                alpha_val = 0.3 + 0.5 * self.gmm.weights_[i] / self.gmm.weights_.max()  # More transparent for less important components
                
                # Add filled contours for component density
                comp_contour = plt.contourf(x_grid, y_grid, comp_density, 
                                          levels=8, cmap=cmap, alpha=alpha_val)
                
                # Add contour lines for better definition
                plt.contour(x_grid, y_grid, comp_density, levels=5, 
                           colors=['black'], alpha=0.4, linewidths=1.0)
        
        # Add overall density contour lines for reference (white dashed lines)
        if density.max() > density.min():
            plt.contour(x_grid, y_grid, density, levels=8, colors=['white'], 
                       alpha=0.9, linewidths=1.5, linestyles='--')
        
        # Plot past trajectory if provided (first, so it's in the background)
        if past_np is not None:
            plt.scatter(past_2d[:, 0], past_2d[:, 1], 
                       color='lightblue', s=30, alpha=0.9, 
                       label="Past Trajectory", zorder=2)
            # Connect past trajectory with lines
            plt.plot(past_2d[:, 0], past_2d[:, 1], 
                    color='blue', alpha=0.7, linewidth=2, zorder=2)
        
        # Plot GT future if provided (middle layer)
        if gt_np is not None:
            plt.scatter(gt_2d[:, 0], gt_2d[:, 1], 
                       color='gold', marker='*', s=400, 
                       label="GT Future Pose", edgecolors='orange', linewidth=3, zorder=4)
        
        # Plot GMM component centers (middle layer)
        if hasattr(self.gmm, 'means_'):
            means_2d = self.gmm.means_[:, :2]  # Only x, y coordinates
            print(f"DEBUG: GMM means (2D): {means_2d}")
            
            # Size components by their weights (more important = larger)
            sizes = 100 + 300 * (self.gmm.weights_ / self.gmm.weights_.max())
            
            plt.scatter(means_2d[:, 0], means_2d[:, 1], 
                       color='white', marker='X', s=sizes, 
                       label="GMM Component Centers", edgecolors='black', linewidth=2, zorder=5)
            
            # Add weight labels next to each component
            for i, (mean, weight) in enumerate(zip(means_2d, self.gmm.weights_)):
                plt.annotate(f'w={weight:.3f}', 
                           (mean[0], mean[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='black', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                           zorder=6)
            
        # Plot k_samples LAST so they're on top and visible
        plt.scatter(k_samples_2d[:, 0], k_samples_2d[:, 1], 
                   c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=1.5,
                   label=f"K Predictions ({len(k_samples_2d)})", marker='o', zorder=10)
        
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='best')
        plt.title(f"Motion Prior Visualization\n"
                 f"Experiment: {experiment_name}, Batch: {batch_idx}, Timestep: t+{timestep+1}\n"
                 f"GMM Components: {self.gmm.n_components}")
        plt.grid(True, alpha=0.3)
        
        # Make sure the plot shows all data points
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        
        # Save the plot
        os.makedirs(save_path, exist_ok=True)
        filename = f"{experiment_name}_Batch{batch_idx}_Timestep{timestep}_MotionPrior.jpg"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved Motion Prior visualization at '{filepath}'")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_prior_prediction(self):
        """Get the most likely prediction from the motion prior"""
        if not self.fitted:
            return None
            
        # Weighted mean of all components
        return self.gmm.means_.T @ self.gmm.weights_
    
    def sample_from_prior(self, n_samples=1):
        """Sample from the fitted motion prior"""
        if not self.fitted:
            return None
            
        return self.gmm.sample(n_samples)[0]
    
    def correct_slam_pose(self, slam_pose, max_weight=0.3):
        """
        Correct SLAM pose using the motion prior (placeholder for future use)
        
        Args:
            slam_pose: SLAM predicted pose
            max_weight: Maximum correction weight
            
        Returns:
            corrected_pose, correction_weight
        """
        # Convert SLAM pose to 6D representation
        slam_pose_6d = self.pose_to_6d_translation(slam_pose)
        
        if isinstance(slam_pose_6d, torch.Tensor):
            slam_pose_6d_np = slam_pose_6d.detach().cpu().numpy()
        else:
            slam_pose_6d_np = slam_pose_6d
            
        # Get motion prior prediction
        prior_prediction = self.get_prior_prediction()
        if prior_prediction is None:
            return slam_pose, 0.0
            
        # Compute correction weight
        correction_weight = self.compute_correction_weight(slam_pose_6d_np, max_weight)
        
        # Blend SLAM and motion prior
        corrected_pose_6d = (
            (1 - correction_weight) * slam_pose_6d_np + 
            correction_weight * prior_prediction
        )
        
        # Convert back to original representation (simplified for now)
        if isinstance(slam_pose, torch.Tensor):
            corrected_pose = torch.from_numpy(corrected_pose_6d).to(slam_pose.device)
        else:
            corrected_pose = corrected_pose_6d
            
        return corrected_pose, correction_weight

class Trainer:
	def __init__(self, config): 
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)

		self.cfg.dataset = config.dataset #use kitti/oxford spires/newer college dataset if specified in command line
		# ------------------------- prepare logs -------------------------
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		
		print_log("\nConfiguration:", self.log)
		for key, value in self.cfg.yml_dict.items():
			print_log(f"{key}: {value}", self.log)
		print()
		
		# ------------------------- prepare train/test data loader -------------------------

		if self.cfg.dataset.lower() == 'nba':
			print_log("[INFO] NBA dataset (11 agent]s).", self.log)
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

		elif self.cfg.dataset.lower() != 'nba':
			dataloader_class = LoadDatasetLeapfrog
			collate_fn = seq_collate_custom
			print_log(f"[INFO] {self.cfg.dataset.upper()} dataset (1 agent).", self.log)

			if config.train in [1, 4]:
				train_dset = dataloader_class(
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=True,
					final_eval=False,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
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
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=False,
					final_eval=False,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
				)
				self.test_loader = DataLoader(
					test_dset,
					batch_size=self.cfg.test_batch_size,
					shuffle=False,
					num_workers=4,
					collate_fn=collate_fn,
					pin_memory=True
				)
				print_log('[INFO] Now using random trajectory shuffling.\n', self.log)
		
				### Stats about trajectories
				if self.cfg.dimensions == 2:
					print_log("Train dataset:", self.log)
					self.print_some_stats(train_dset.fut_motion_3D, None, 2)
					print_log("\nValidation dataset:", self.log)
					self.print_some_stats(test_dset.fut_motion_3D, None, 2)

				elif self.cfg.dimensions == 3:
					print_log("Train dataset:", self.log)
					self.print_some_stats(train_dset.fut_motion_3D, None, 3)
					print_log("\nValidation dataset:", self.log)
					self.print_some_stats(test_dset.fut_motion_3D, None, 3)
				
				elif self.cfg.dimensions == 6:
					print_log("Train dataset:", self.log)
					self.print_some_stats(train_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)
					print_log("\nValidation dataset:", self.log)
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)

				elif self.cfg.dimensions == 9:
					print_log("Train dataset:", self.log)
					self.print_some_stats(train_dset.fut_motion_3D[..., :3], train_dset.fut_motion_3D[..., 3:], 3)
					print_log("\nValidation dataset:", self.log)
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)
			
			elif config.train in [0, 2, 3]:
				test_dset = dataloader_class(
					dataset=self.cfg.dataset.lower(),
					dims=self.cfg.dimensions,
					input_size=self.cfg.past_frames,
					preds_size=self.cfg.future_frames,
					training=False,
					final_eval=True,
					relative=self.cfg.relative, 
					normalised=self.cfg.normalised, 
					train_ratio=0.80,
					eval_ratio=0.10,
					seed=42,
					overlapping = self.cfg.overfitting,
					selected_trajectories=self.cfg.selected_trajectories,
					synthetic_gt = self.cfg.synthetic_gt,
					synthetic_noise = self.cfg.synthetic_noise
				)
				self.test_loader = DataLoader(
					test_dset,
					batch_size=self.cfg.test_batch_size,
					shuffle=False,
					num_workers=4,
					collate_fn=collate_fn,
					pin_memory=True
				)

				### Stats about trajectories
				if self.cfg.dimensions == 2:
					print_log("\nTest dataset (model evaluation):", self.log)
					self.print_some_stats(test_dset.fut_motion_3D, None, 2)

				elif self.cfg.dimensions == 3:
					print_log("\nTest dataset (model evaluation):", self.log)
					self.print_some_stats(test_dset.fut_motion_3D, None, 3)
				
				elif self.cfg.dimensions == 6:
					print_log("\nTest dataset (model evaluation):", self.log)
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)
				
				elif self.cfg.dimensions == 9:
					print_log("\nTest dataset (model evaluation):", self.log)
					self.print_some_stats(test_dset.fut_motion_3D[..., :3], test_dset.fut_motion_3D[..., 3:], 3)


				#TODO implement for 9D << rotation without last column + translation
			# if config.train == 1:
			# 	self.clusterer = OfflineTrajectoryClusterer(self, num_clusters=5, pca_dim=20)

			if self.cfg.future_frames < 20:
				print_log(f"[Warning] Only {self.cfg.future_frames} future timesteps available, "
					f"ADE/FDE will be computed for up to {self.cfg.future_frames // 5} seconds instead of the full 4 seconds.", self.log)

		
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
			print_log(f'\n[INFO] {self.cfg.dataset.upper()} dataset - skip subtracting mean from absolute positions.', self.log)
			
		
		
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
		self.model = CoreDenoisingModel(
			t_h=self.cfg.past_frames,
			d_f=self.cfg.dimensions
		).cuda()

		if self.cfg.past_frames == 10 and self.cfg.future_frames == 20 and self.cfg.dataset == 'nba':
			# load pretrained models 
			print_log('[INFO] Loading pretrained models... (NBA with standard frame configs)\n', self.log)
			model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu') #LB expects 60 dimensional input (6 x 10 past poses)
			self.model.load_state_dict(model_cp['model_dict'])

			self.model_initializer = InitializationModel(
				t_h=self.cfg.past_frames, 
				t_f=self.cfg.future_frames, 
				d_f=self.cfg.dimensions, 
				k_pred=self.cfg.k_preds
			).cuda()
		
		else:
			print_log('[INFO] Training from scratch - without pretrained models (Not NBA with frame standard configs)\n', self.log)
			# print('Params for model_initialiser: ', self.cfg.past_frames, self.cfg.dimensions*3, self.cfg.future_frames, self.cfg.dimensions, self.cfg.k_preds)
			self.model_initializer = InitializationModel( #DIM update delete
				t_h=self.cfg.past_frames, 
				t_f=self.cfg.future_frames, 
				d_f=self.cfg.dimensions, 
				k_pred=self.cfg.k_preds
			).cuda()

		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		


		self.print_model_param(self.model, name='Core Denoising Model')
		self.print_model_param(self.model_initializer, name='Initialization Model')

		# print(self.model)
		#print(self.model_initializer)

		# temporal reweight in the loss, it is not necessary.
		#self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10
		self.temporal_reweight = torch.FloatTensor([self.cfg.future_frames - i for i in range(1, self.cfg.future_frames + 1)]).cuda().unsqueeze(0).unsqueeze(0) / 10

	def save_checkpoint(self, epoch):
		"""
        Save a checkpoint containing both core denoising model (original) and
        the initialization model, along with optimizer and scheduler states
        """
		checkpoint = {
			'epoch': epoch,
			'cfg': self.cfg.yml_dict,
			'model_initializer_state_dict': self.model_initializer.state_dict(),
			'core_denoising_model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.opt.state_dict(),
			'scheduler_state_dict': self.scheduler_model.state_dict(),
		}
		checkpoint_dir = f'./{self.cfg.model_dir}'
		os.makedirs(checkpoint_dir, exist_ok=True)
		ckpt_path = os.path.join(checkpoint_dir, f'best_checkpoint_epoch_{epoch}.pth')
		
		# delete previous best checkpoint, if it exists
		if hasattr(self, 'best_checkpoint_path') and self.best_checkpoint_path is not None:
			if os.path.exists(self.best_checkpoint_path):
				os.remove(self.best_checkpoint_path)
		
		torch.save(checkpoint, ckpt_path)
		self.best_checkpoint_path = ckpt_path
		print_log(f"[INFO] New best model (ATE)! Checkpoint saved to {ckpt_path}", self.log)

	def load_checkpoint(self, checkpoint_path):
		"""
		Load a checkpoint and restore model, optimizer, and scheduler states.
		"""
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		self.model_initializer.load_state_dict(checkpoint['model_initializer_state_dict'])
		self.model.load_state_dict(checkpoint['core_denoising_model_state_dict'])
		self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler_model.load_state_dict(checkpoint['scheduler_state_dict'])
		print_log(f"[INFO] Checkpoint loaded from {checkpoint_path}", self.log)

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
		#sample = mean + sigma_t * z * 0.1

		return (sample)
	
	def p_sample_loop_mean(self, x, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate_old(self, x, mask, loc):
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
		prediction_total = torch.cat((cur_y, cur_y_), dim=1)

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
		cur_y = loc  # use all k predictions
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		return cur_y  # shape: (B, k_pred, T, d)


	def fit(self):

		# Training loop
		for epoch in range(0, self.cfg.num_epochs):
			loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty = self._train_single_epoch(epoch)

			if self.cfg.dimensions in [2,3]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch+1, loss_total, loss_distance, loss_uncertainty), self.log)
			
			elif self.cfg.dimensions in [6,7,9]:
				print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Translation.: {:.6f}\tLoss Rotation.: {:.6f}\tCombined Loss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
					time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
					epoch+1, loss_total, loss_trans, loss_rot, loss_distance, loss_uncertainty), self.log)


			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples= self._test_single_epoch() #average_euclidean = average total distance start to finish - to contextualise how good the FDE and ADE are

				# Print ADE/FDE metrics as before
				timesteps = list(range(5, self.cfg.future_frames, 5))
				if not timesteps or timesteps[-1] != self.cfg.future_frames:
					timesteps.append(self.cfg.future_frames)
				for i, time_i in enumerate(timesteps):
					print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
						time_i, performance['ADE'][i]/samples,
						time_i, performance['FDE'][i]/samples), self.log)
				
				# Print new ATE metrics
				print_log('--ATE translation: {:.4f}'.format(performance['ATE_trans']), self.log)
				
				# Update best model selection to include ATE if desired
				ate_trans = performance['ATE_trans']
				ade_final = performance['ADE'][-1]/samples
				
				if epoch == 0:
					best_ate = ate_trans
					best_ade = ade_final
				elif ate_trans < best_ate:  # Use ATE_trans as primary metric
					best_ate = ate_trans
					self.save_checkpoint(epoch+1)
				# elif ade_final < best_ade:  # Or keep using ADE as fallback
				# 	best_ade = ade_final
				# 	self.save_checkpoint(epoch+1)
					
				#print ADE/FDE of timesteps that are a multiple of 5 and final timestep ADE/FDE
				# timesteps = list(range(5, self.cfg.future_frames, 5))
				# if not timesteps or timesteps[-1] != self.cfg.future_frames:
				# 	timesteps.append(self.cfg.future_frames)
				

				# for i, time_i in enumerate(timesteps): #self.cfg.future_frames
				# 	print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(
				# 		time_i, performance['ADE'][i]/samples,
				# 		time_i, performance['FDE'][i]/samples), self.log)

				# #save model if it's the best so far
				# ade_final_pose = performance['ADE'][-1]/samples
				# if epoch == 0:
				# 	best_ade = ade_final_pose
				# elif ade_final_pose < best_ade:
				# 	best_ade = ade_final_pose
				# 	self.save_checkpoint(epoch+1)

			self.scheduler_model.step()


	def data_preprocess_with_abs(self, data): #Updated to handle any number of agents
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

		#print(data['pre_motion_3D'][0,0,:,:])

		#Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
		traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
		for i in range(batch_size):
			traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.

		# print('traj_mask: ', traj_mask.size())
		# Get last observed pose (for each agent) as initial position << both translation and rotation
		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:] # 2D: [B, num_agents, 1, 2] or 3D: [B, num_agents, 1, 3] or 6D: [B, num_agents, 1, 6]

		# augment input: absolute position, relative position, velocity TODO augmentation is applied to whole input, not only translation
		if self.cfg.dataset != 'nba':
			past_traj_abs = (data['pre_motion_3D'].cuda() / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #single agent: effectively only (B, 1, Past, Dims) > (B*1, Past, Dims) and scaling
		elif self.cfg.dataset == 'nba':
			past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) 

		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.past_frames, self.cfg.dimensions) #only relativises if initial pos is not 0 already (relative = True)

		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1) #(B, 1, Dim)
		
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, self.cfg.future_frames, self.cfg.dimensions) #relativises (if not done already) and (B, 1, Past, Dims) > (B*1, Past, Dims)

		return batch_size, traj_mask, past_traj, fut_traj

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
		#print(data['pre_motion_3D'][0,0,:,:])


		### 1.0 Create trajectory mask [batch_size * num_agents, batch_size * num_agents]
		traj_mask = torch.zeros(batch_size*num_agents, batch_size*num_agents).cuda()
		for i in range(batch_size):
			traj_mask[i*num_agents:(i+1)*num_agents, i*num_agents:(i+1)*num_agents] = 1.


		### 2.0 compute relative translation and rotation of past and future trajectory, compute velocities of translation
		if self.cfg.dimensions in [2,3]:
			# Get last observed pose (for each agent) as initial position
			last_observed_pos = data['pre_motion_3D'].cuda()[:, :, -1:] # 2D: [B, num_agents, 1, 2] or 3D: [B, num_agents, 1, 3]

			# relative positions
			past_rel = (data['pre_motion_3D'].to(self.device) - last_observed_pos) / self.traj_scale  # [B, num_agents, past_frames, d]
			# velocities (difference between consecutive relative positions)
			past_vel = torch.cat((past_rel[:, :, 1:] - past_rel[:, :, :-1],
								torch.zeros_like(past_rel[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, d]
			
			
			# concat relative positions and velocities
			past_traj = torch.cat((past_rel, past_vel), dim=-1)  # [B, num_agents, past_frames, 2*d]

			# relative future trajectory
			fut_traj = (data['fut_motion_3D'].to(self.device) - last_observed_pos) / self.traj_scale  # [B, num_agents, future_frames, d]
			
			# reshape to merge batch and agent dimensions:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)
		
		elif self.cfg.dimensions ==6:
			# For 6D: Data is expected to be of shape [B, num_agents, past_frames, 6] for past,
			# and [B, num_agents, future_frames, 6] for future.
			past_abs = data['pre_motion_3D'].to(self.device)  # [B, num_agents, past_frames, 6]
			fut_abs  = data['fut_motion_3D'].to(self.device)  # [B, num_agents, future_frames, 6]
			
			# Split into translation and rotation (Lie algebra) parts:
			past_trans = past_abs[..., :3]  # [B, num_agents, past_frames, 3]
			past_rot   = past_abs[..., 3:]  # [B, num_agents, past_frames, 3]
			
			fut_trans = fut_abs[..., :3]    # [B, num_agents, future_frames, 3]
			fut_rot   = fut_abs[..., 3:]    # [B, num_agents, future_frames, 3]
			
			# Get the last observed pose per agent:
			last_obs = past_abs[:, :, -1:]  # [B, num_agents, 1, 6]
			last_trans = last_obs[..., :3]  # [B, num_agents, 1, 3]
			last_rot = last_obs[..., 3:]    # [B, num_agents, 1, 3]
			
			# Compute relative translation (for past and future):
			rel_trans_past = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			rel_trans_fut  = (fut_trans - last_trans) / self.traj_scale   # [B, num_agents, future_frames, 3]
			
			# Compute relative rotation for past:
			# We need to compute, for each time step, the relative rotation given last_rot.
			# We'll reshape to merge batch, agent, and time dimensions, apply our helper, then reshape back.
			B, N, T, _ = past_rot.shape
			past_rot_flat = past_rot.reshape(-1, 3)              # (B*N*T, 3)
			last_rot_expanded = last_rot.expand(B, N, T, 3).reshape(-1, 3)  # (B*N*T, 3)
			rel_rot_past_flat = relative_lie_rotation(past_rot_flat, last_rot_expanded)
			rel_rot_past = rel_rot_past_flat.view(B, N, T, 3)     # [B, num_agents, past_frames, 3]
			
			# Concatenate to obtain past relative pose (6D)
			past_rel = torch.cat((rel_trans_past, rel_rot_past), dim=-1)  # [B, num_agents, past_frames, 6]
			
			# Compute translation velocity for past (only on translation):
			past_vel = torch.cat((rel_trans_past[:, :, 1:] - rel_trans_past[:, :, :-1],
								torch.zeros_like(rel_trans_past[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, 3]
			
			# Concatenate past relative pose and translation velocity → per-timestep input of 9 dims:
			past_traj = torch.cat((past_rel, past_vel), dim=-1)  # [B, num_agents, past_frames, 9]
			
			# For future: compute relative rotation similarly:
			B_f, N_f, T_f, _ = fut_rot.shape
			fut_rot_flat = fut_rot.reshape(-1, 3)
			last_rot_fut = last_rot.expand(B_f, N_f, T_f, 3).reshape(-1, 3)
			rel_rot_fut_flat = relative_lie_rotation(fut_rot_flat, last_rot_fut)
			rel_rot_fut = rel_rot_fut_flat.view(B_f, N_f, T_f, 3)  # [B, num_agents, future_frames, 3]
			
			fut_rel = torch.cat((rel_trans_fut, rel_rot_fut), dim=-1)  # [B, num_agents, future_frames, 6]
			
			# Reshape: merge the batch and agent dimensions
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_rel.view(-1, self.cfg.future_frames, self.cfg.dimensions)  # here self.cfg.dimensions == 6
    
		elif self.cfg.dimensions == 7: # TUM format: Absolute SE(3) poses: [B, num_agents, T, 7] (translations + quaternions)

			last_obs_pose = data['pre_motion_3D'].to(self.device)[:, :, -1:]  # [B, num_agents, 1, 7]
			
			# translation: subtract last observed translation.
			past_abs = data['pre_motion_3D'].to(self.device)  # [B, num_agents, past_frames, 7]
			past_trans = past_abs[..., :3]               # translations: [B, num_agents, past_frames, 3]
			last_trans  = last_obs_pose[..., :3]           # [B, num_agents, 1, 3]
			rel_trans = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			
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


			# relative future trajectory
			fut_abs = data['fut_motion_3D'].to(self.device)  # [B, num_agents, future_frames, 7]
			fut_trans = fut_abs[..., :3]
			fut_rel_trans = (fut_trans - last_trans) / self.traj_scale  # [B, num_agents, future_frames, 3]
			fut_quat = fut_abs[..., 3:7]
			fut_rel_quat = quaternion_relative(last_quat, fut_quat)  # [B, num_agents, future_frames, 4]
			fut_traj = torch.cat((fut_rel_trans, fut_rel_quat), dim=-1)  # [B, num_agents, future_frames, 7]   <<< output size


			# Reshape so that each agent is a separate trajectory:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)

		elif self.cfg.dimensions == 9: 
			# 2. Get absolute poses for past and future
			past_abs = data['pre_motion_3D'].to(self.device)  # shape: [B, num_agents, past_frames, 9]
			fut_abs  = data['fut_motion_3D'].to(self.device)   # shape: [B, num_agents, future_frames, 9]
			
			# 3. Split into translation and rotation components
			past_trans = past_abs[..., :3]      # [B, num_agents, past_frames, 3]
			past_rot6d = past_abs[..., 3:]      # [B, num_agents, past_frames, 6]
			
			fut_trans = fut_abs[..., :3]        # [B, num_agents, future_frames, 3]
			fut_rot6d = fut_abs[..., 3:]        # [B, num_agents, future_frames, 6]
			
			# 4. Last observed absolute pose for each agent (from past)
			last_obs = past_abs[:, :, -1:]      # [B, num_agents, 1, 9]
			last_trans = last_obs[..., :3]       # [B, num_agents, 1, 3]
			last_rot6d = last_obs[..., 3:]       # [B, num_agents, 1, 6]
			
			# 5. Compute relative translation
			rel_trans_past = (past_trans - last_trans) / self.traj_scale  # [B, num_agents, past_frames, 3]
			rel_trans_fut  = (fut_trans  - last_trans) / self.traj_scale   # [B, num_agents, future_frames, 3]
			
			# 6. Compute relative rotation for past:
			# Convert past 6d rotations to rotation matrices:
			R_past = rot6d_to_rotmat(past_rot6d.reshape(-1, 6))  # shape: [B*num_agents*past_frames, 3, 3]
			R_past = R_past.view(batch_size, num_agents, past_abs.shape[2], 3, 3)  # [B, num_agents, past_frames, 3, 3]
			# Convert reference (last) rotation to a matrix:
			R_ref = rot6d_to_rotmat(last_rot6d.squeeze(2))       # [B, num_agents, 3, 3]
			# Expand R_ref along time:
			R_ref_exp = R_ref.unsqueeze(2).expand_as(R_past)       # [B, num_agents, past_frames, 3, 3]
			# Compute relative rotation: R_rel = R_ref^T @ R_past.
			R_rel_past = torch.matmul(R_ref_exp.transpose(-2, -1), R_past)   # [B, num_agents, past_frames, 3, 3]
			# Convert relative rotation matrix back to 6d:
			rel_rot6d_past = rotmat_to_rot6d(R_rel_past.view(-1, 3, 3))  # [B*num_agents*past_frames, 6]
			rel_rot6d_past = rel_rot6d_past.view(batch_size, num_agents, past_abs.shape[2], 6)  # [B, num_agents, past_frames, 6]
			
			# For future
			R_fut = rot6d_to_rotmat(fut_rot6d.reshape(-1, 6))  # [B*num_agents*future_frames, 3, 3]
			R_fut = R_fut.view(batch_size, num_agents, fut_abs.shape[2], 3, 3)  # [B, num_agents, future_frames, 3, 3]
			R_ref_fut = R_ref.unsqueeze(2).expand_as(R_fut)    # [B, num_agents, future_frames, 3, 3]
			R_rel_fut = torch.matmul(R_ref_fut.transpose(-2, -1), R_fut)  # [B, num_agents, future_frames, 3, 3]
			rel_rot6d_fut = rotmat_to_rot6d(R_rel_fut.view(-1, 3, 3))  # [B*num_agents*future_frames, 6]
			rel_rot6d_fut = rel_rot6d_fut.view(batch_size, num_agents, fut_abs.shape[2], 6)  # [B, num_agents, future_frames, 6]
			
			# 7. Concatenate translation and rotation for past relative pose.
			past_rel_pose = torch.cat((rel_trans_past, rel_rot6d_past), dim=-1)  # [B, num_agents, past_frames, 3+6 = 9]
			
			# 8. Compute translational velocity for past (from relative translation only).
			past_vel = torch.cat((rel_trans_past[:, :, 1:] - rel_trans_past[:, :, :-1],
								torch.zeros_like(rel_trans_past[:, :, -1:])), dim=2)  # [B, num_agents, past_frames, 3]
			
			# 9. Final past processed trajectory: concatenate relative pose (9) with translation velocity (3) → 12 per timestep.
			past_traj = torch.cat((past_rel_pose, past_vel), dim=-1)  # [B, num_agents, past_frames, 12]
			
			# 10. Final future processed trajectory: simply the relative pose (9)
			fut_traj = torch.cat((rel_trans_fut, rel_rot6d_fut), dim=-1)  # [B, num_agents, future_frames, 9]
			
			# 11. Merge batch and agent dimensions:
			past_traj = past_traj.view(-1, self.cfg.past_frames, past_traj.shape[-1])
			fut_traj = fut_traj.view(-1, self.cfg.future_frames, self.cfg.dimensions)  # here self.cfg.dimensions should be 9
  
		else:
			raise NotImplementedError("data_preprocess for dimensions 6,7,9,2,3 are implemented; others not yet.")

		return batch_size, traj_mask, past_traj, fut_traj



	### Rotation conversion helpers
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

	def rot6d_to_rotmat_SO3(self, x):
		"""
		Convert a 6D rotation representation to a 3x3 rotation matrix.
		x: tensor of shape (N, 6)
		Returns a tensor of shape (N, 3, 3)
		"""
		# Split into two 3D vectors
		a1 = x[:, :3]
		a2 = x[:, 3:]
		# Normalize the first vector to get v1
		v1 = a1 / torch.norm(a1, dim=1, keepdim=True)
		# Make a2 orthogonal to v1
		a2_proj = (torch.sum(v1 * a2, dim=1, keepdim=True)) * v1
		v2 = a2 - a2_proj
		v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
		# Compute v3 as cross product of v1 and v2
		v3 = torch.cross(v1, v2, dim=1)
		# Stack into rotation matrix
		R = torch.stack([v1, v2, v3], dim=-1)
		return R



	def prior_gradient_logl(self, kde, pose, epsilon=1e-5):
		"""
		Numerically approximate the gradient of the log likelihood at `pose`.
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


	def print_some_stats(self, future, future_rot=None, translation_dims=3):
		print_log('Length:', future.size(0), self.log)
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
			print_log(f"Total x and y distances travelled: {mean_distance_x:.5f}, {mean_distance_y:.5f}", self.log)
		elif translation_dims == 3:
			print_log(f"Total x, y and z distances travelled: {mean_distance_x:.5f}, {mean_distance_y:.5f}, {mean_distance_z:.5f}", self.log)

		print_log(f"Euclidean dist diff avg: {mean_euclidean_distance:.5f}", self.log)

		if future_rot is not None:
			print_log('Still need to implement rotation statistics', self.log)

	def visualise_single_KDE_GT_Past(self, k_preds_at_t, t_kde, all_past, GT_at_t, idx): #TODO make size dynamic
		if hasattr(k_preds_at_t, "cpu"):
			k_preds_at_t = k_preds_at_t.cpu().numpy()
		if hasattr(all_past, "cpu"):
			all_past = all_past.cpu().numpy()
		if hasattr(GT_at_t, "cpu"):
			GT_at_t = GT_at_t.cpu().numpy()
		# k_preds_at_t = k_preds_at_t.cpu().numpy()
		# all_past = all_past.cpu().numpy()
		# GT_at_t = GT_at_t.cpu().numpy()
		
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
		vis_path = f'./visualization/Sanity_Synthetic_Single_Timestep_{idx}.jpg'
		plt.savefig(vis_path)
		print_log(f'[INFO] Visualisation saved to {vis_path}', self.log)
		plt.close()

	def compute_kde_and_vis_full_traj(self, k_preds, past_traj, fut_traj, experiment_name, exclude_last_timestep=True, kpreds=True): #TODO how is KDE computed here
		"""
		Computes a KDE-based motion prior for each sample (batch) using all predictions across time,
		and visualizes the density along with sample points, past trajectories, and GT future poses.

		Args:
			k_preds (Tensor): Predicted trajectories of shape (B, K, T, Dim).
			past_traj (Tensor): Past trajectories of shape (B, TPast, 2).
			fut_traj (Tensor): Ground truth future trajectories of shape (B, T, 2).
		"""
		# Convert tensors to NumPy arrays
		k_preds_np = k_preds.detach().cpu().numpy()
		past_np = past_traj.detach().cpu().numpy()
		fut_np = fut_traj.detach().cpu().numpy()

		if exclude_last_timestep:
			k_preds_np = k_preds_np[:, :, :-1, :]  # Now T becomes T-1
			fut_np = fut_np[:, :-1, :]
		B, K, T, D = k_preds_np.shape

		for b in range(B):
			# Reshape predicted trajectories to combine all time steps (K * T, D)
			all_samples = k_preds_np[b].reshape(K * T, D)

			# Optionally, filter out any outlier samples if needed.
			all_samples = self.filter_k_preds_single_pose(all_samples)
			# Fit the KDE using the (filtered) sample points.
			kde = gaussian_kde(all_samples.T)

			# Combine all points (samples, past, and GT future) for dynamic grid limits.
			points_for_grid = np.concatenate([all_samples, past_np[b], fut_np[b]], axis=0)
			margin = 0.1  # Adjust this margin if necessary.
			min_x = points_for_grid[:, 0].min() - margin
			max_x = points_for_grid[:, 0].max() + margin
			min_y = points_for_grid[:, 1].min() - margin
			max_y = points_for_grid[:, 1].max() + margin

			# Create a grid dynamically based on computed boundaries.
			x_grid, y_grid = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
			grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
			kde_values = kde(grid_points).reshape(100, 100)

			# Begin plotting
			plt.figure(figsize=(10, 8))
			plt.contourf(x_grid, y_grid, kde_values, levels=50, cmap="viridis")
			# Plot the KDE sample points (e.g., small dots)
			plt.scatter(all_samples[:, 0], all_samples[:, 1], s=3, alpha=1, label="Samples")
			# Plot the past trajectory with light blue dots
			plt.scatter(past_np[b][:, 0], past_np[b][:, 1], color='lightblue', s=30, label="Past Trajectory")
			# Plot the GT future poses with red stars
			plt.scatter(fut_np[b][:, 0], fut_np[b][:, 1], color='red', marker='*', s=100, label="GT Future Poses")
			
			plt.xlabel("X Position")
			plt.ylabel("Y Position")
			plt.colorbar(label="Density")
			plt.legend()
			plt.title(f"KDE Visualization for Batch Sample {b}")
			if kpreds==True:
				plt.savefig(f'./visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_denoised_kpreds.jpg')
				print(f"[INFO] Saved KDE visualization with kpreds at './visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_denoised_kpreds.jpg'")
			else:
				plt.savefig(f'./visualization/{experiment_name}_Trajectory{b}_AllTimesteps_dynamic_initialised_only.jpg')

			plt.close()
			exit()

	
	def compute_batch_motion_priors_kde_temporal_independence(self, k_preds):#consider that we only visualise xy TODO
		"""
		Computes a KDE-based motion prior for each sample and each future time step.

		Limitation: Using per-timestep KDE densities assumes that the predictions at each timestep are independent, but they aren’t.
		
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

	def filter_k_preds_single_pose(self, single_pose_all_ks):
		med = np.median(single_pose_all_ks, axis=0)
		#Euclidean distances from the median
		distances = np.linalg.norm(single_pose_all_ks - med, axis=1)
		mad = np.median(np.abs(distances - np.median(distances)))

		# threshold: remove samples that are more than 5 MAD away from the median distance
		threshold = np.median(distances) + 5 * mad
		
		# Filter out outliers (if only one crazy outlier exists, this will remove it)
		filtered_k_poses = single_pose_all_ks[distances <= threshold]
		return filtered_k_poses

	def KDE_single_pose_outlier_filtered(self, single_pose_all_ks):

		filtered_k_poses = self.filter_k_preds_single_pose(single_pose_all_ks)
		single_pose_KDE = gaussian_kde(filtered_k_poses.T)
		return single_pose_KDE

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

	def GT_KDE_density_histograms(self, all_densities_by_time, out_dir):
		for t in range(self.cfg.future_frames):
			plt.figure(figsize=(8, 6))
			plt.hist(all_densities_by_time[t], bins=30, edgecolor='black')
			plt.xlabel("KDE Density")
			plt.ylabel("Frequency")
			plt.title(f"KDE Density Histogram for Future Timestep {t}")
			vis_path = os.path.join(out_dir, f'KDE_Density_Time_{t}.jpg')
			plt.savefig(vis_path)
			plt.close()
			print(f"[INFO] Saved histogram for future timestep {t} at {vis_path}")

	def pose_to_6d_translation(self, pose):
		"""Convert pose to [tx, ty, tz, r1, r2, r3, r4, r5, r6] format"""
		dim = pose.shape[-1]
		
		if dim in [2, 3]:  # translation only
			return pose
		elif dim == 7:  # quaternion
			quat = pose[..., 3:7]
			rot_matrix = quaternion_to_rotation_matrix(quat)
			rot_6d = rotation_matrix_to_6d(rot_matrix)
			return torch.cat([pose[..., :3], rot_6d], dim=-1)
		elif dim == 9:  # already 6D
			return pose
		elif dim == 6:  # axis-angle
			rot_matrix = self.so3_to_SO3(pose[..., 3:6])
			rot_6d = rotation_matrix_to_6d(rot_matrix)
			return torch.cat([pose[..., :3], rot_6d], dim=-1)

	def _train_single_epoch(self, epoch):
		
		self.model.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, loss_trans, loss_rot, count = 0, 0, 0, 0, 0,0
		#LB 3D addition to reshape tensors 
		
		for i, data in enumerate(self.train_loader):
			# torch.set_printoptions(
			# 	precision=4,   # number of digits after the decimal
			# 	sci_mode=False # turn off scientific (e+) notation
			# )
			# print(f"first traj all poses (pre): ", data['pre_motion_3D'].shape, data['pre_motion_3D'][0,0,:,:]) #first traj all poses (pre) (B, A, T, D)
			# print(f"first traj all poses (fut): ", data['fut_motion_3D'][0,0,:,:]) #first traj all poses (fut)

			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data) # past_traj =(past_traj_abs, past_traj_rel, past_traj_vel)

			# if i in [0,1]:
			# 	print("past_traj[0,0,:] on bad batch:", past_traj[0,0,:].detach().cpu())
			#print('fut_traj:', fut_traj[0,0,:]) #first fut timestep


			# print('traj_mask:', traj_mask.size()) # [32, 32]

			# print('past_traj processed:', past_traj.shape, past_traj[0,:,:3]) # [32, T, D+3]  # XXX SHOULD BE THE SAME FORMAT
			# print('fut_traj processed:', fut_traj[0]) # [32, T, D+3] < GT poses for future_frames timesteps
			# exit()
			
			### 1. Leapfrogging Denoising (LED initializer outputs): instead of full denoising process, predicts intermediate, already denoised trajectory 
			### the 9d nan issue happens here on the second batch in the train loop
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) #offset, mean, stdev
			# print("— raw_pred[0,0,0,:] pre‐reparam:", sample_prediction[0,0,0,:].detach().cpu().numpy())
			# print("— mean_est[0,:]           :", mean_estimation[0,:].detach().cpu().numpy())
			# print("— var_est[0,:]            :", variance_estimation[0,:]e.dtach().cpu().numpy())
			#uses the past trajectory (and possibly social context) to produce a mean and variance for the future trajectory - sampled from to get candidate future trajectories next
			#sample prediction: provides the normalized offsets Sbθ,k that, when scaled and added to the mean, yield the final candidate trajectories

			# print("sample_prediction shape:", sample_prediction.shape)
			# print("mean_estimation shape:", mean_estimation.shape)
			# print("variance_estimation shape:", variance_estimation.shape)
		

			#TODO could clip variance to constrain k_preds more
			# variance_scale = torch.exp(variance_estimation/2)
			# variance_scale = torch.clamp(variance_scale, max=threshold_value)  # Set an appropriate threshold
			# sample_prediction = variance_scale[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None] 

			# Reparameterisation with uncertainty (original)
			# denom = sample_prediction.std(dim=1).mean(dim=(1,2))  # a single scalar per batch element
			# with torch.no_grad():
			# 	denom = sample_prediction.std(dim=1).mean(dim=(1,2))  # shape [B]
			# 	print("variance head:", variance_estimation.min().item(), variance_estimation.max().item())
			# 	print("denom min/mean/max:", denom.min().item(), denom.mean().item(), denom.max().item())


			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]

			# print('sample_prediction')
			# print(sample_prediction[0,0,0,:]) #check if prediction is 9D
			# print('This shouldnt be super small - if it is, this explains outliers:', sample_prediction.std(dim=1).mean(dim=(1, 2)))
			
			# Add the mean estimation to the scaled normalized offsets / center predictions around the mean (each candidate trajectory is expressed as a deviation from the mean)

			loc = sample_prediction + mean_estimation[:, None] #prediction before denoising
			# print("— loc min/max:", loc.min().item(), loc.max().item())


			# print('sample_prediction:', sample_prediction.size())
			# print('loc:', loc.size()) #[32, 24, 24, 3]

			### 2. Denoising (Denoising Module): Generate K alternative future trajectories - multi-modal
			k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) #(B, K, T, 2/3/6/7/9)
			# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])
			# exit()

			
			# print('k_alternative_preds:', k_alternative_preds[0,:,0,:2]) #check if prediction is 9D

			# scale = torch.exp(variance_estimation/2)
			# Log statistics for each batch element and each prediction (over K)
			# print("Variance stats per batch element:")
			# for i in range(scale.shape[0]):
			# 	print("Batch {}: min={:.3f}, max={:.3f}, mean={:.3f}".format(
			# 		i, scale[i].min().item(), scale[i].max().item(), scale[i].mean().item()))

			# self.old_compute_kde_and_vis_full_traj(k_alternative_preds)
			# exit()

			### 3D/2D code
			if self.cfg.dimensions in [2,3]:
				#squared distances / Euclidian, equal weight for all timesteps
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)
				loss_distance = ((k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1) * 
								self.temporal_reweight
								).mean(dim=-1).min(dim=1)[0].mean()
				loss_uncertainty = (torch.exp(-variance_estimation)*
									(k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
									variance_estimation
									).mean()
			
			elif self.cfg.dimensions == 6:
				# For loss, unsqueeze the ground truth to have predictions dimension
				"""6D specific code"""
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)

				# Split into translation and rot
				pred_trans = k_alternative_preds[..., :3]    # (B, K, T, 3)
				pred_rot_lie = k_alternative_preds[..., 3:]    # 6D: (B, K, T, 3) or 9D: (B, 1, T, 6)
				gt_trans = fut_traj_wpreds[..., :3]      # (B, 1, T, 3) 
				gt_rot_lie = fut_traj_wpreds[..., 3:]      # 6D: (B, 1, T, 3) or 9D: (B, 1, T, 6)

				### (1) TRANSLATION LOSS
				# L2 Euclidian distance - squared distances, equal weight for timesteps
				trans_diff = pred_trans-gt_trans # (B, K, T, 3)
				trans_error = trans_diff.norm(p=2, dim=-1) # (B, K, T)

				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1)	# (B, K) loss for all k preds
				# print(loss_translation)


				### (2) ROTATION LOSS (Geodesic)
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
			
			elif self.cfg.dimensions == 7:
				# For 7D, we need to split the 7 dimensions: translation (first 3) and quaternion (last 4)
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)  # shape: (B, 1, T, 7)
				# Split predictions and ground truth:
				pred_trans = k_alternative_preds[..., :3]  # shape: (B, K, T, 3)
				pred_quat  = k_alternative_preds[..., 3:]  # shape: (B, K, T, 4)
				gt_trans   = fut_traj_wpreds[..., :3]        # shape: (B, 1, T, 3)
				gt_quat    = fut_traj_wpreds[..., 3:]        # shape: (B, 1, T, 4)

				# Compute translation error (L2 norm)
				trans_error = (pred_trans - gt_trans).norm(p=2, dim=-1)  # (B, K, T)
				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1).min(dim=1)[0].mean()

				# Normalize quaternions (add a small epsilon to avoid div-by-zero)
				pred_quat_norm = pred_quat / (pred_quat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
				gt_quat_norm   = gt_quat   / (gt_quat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6))
				
				# Compute dot product between normalized quaternions and clamp to valid range [-1,1]
				dot = (pred_quat_norm * gt_quat_norm).sum(dim=-1).abs().clamp(max=1.0 - 1e-6)  # (B, K, T)
				# Angular error (in radians)
				rot_error = torch.acos(dot)  # (B, K, T)
				loss_rotation = rot_error.mean(dim=-1).min(dim=1)[0].mean()

				# Combined loss over translation and rotation.
				loss_distance = loss_translation + loss_rotation


			elif self.cfg.dimensions == 9:
				#generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc) 

				# print('Predictions batch:', generated_y.size())
				# print('GT batch:', fut_traj.size())

				# print('Prediction 0 shape:', generated_y[0].size())
				# print('GT 0 shape:', fut_traj[0].size())

				# print('Prediction:', generated_y[0])
				# print('GT:', fut_traj[0])

				# For loss, unsqueeze the ground truth to have predictions dimension
				"""9D specific code"""
				fut_traj_wpreds = fut_traj.unsqueeze(dim=1)

				# Split into translation and rot
				pred_trans = k_alternative_preds[..., :3]    # (B, K, T, 3)
				pred_rot_6D = k_alternative_preds[..., 3:]    # (B, 1, T, 6)
				gt_trans = fut_traj_wpreds[..., :3]      # (B, 1, T, 3) 
				gt_rot_6D = fut_traj_wpreds[..., 3:]      # (B, 1, T, 6)

				### (1) TRANSLATION LOSS
				# L2 Euclidian distance - squared distances, equal weight for timesteps
				trans_diff = pred_trans-gt_trans # (B, K, T, 3)
				trans_error = trans_diff.norm(p=2, dim=-1) # (B, K, T)

				loss_translation = (trans_error * self.temporal_reweight).mean(dim=-1)	# (B, K) loss for all k preds


				### (2) ROTATION LOSS (Geodesic)
				# convert 6D rotations to classic 3x3 rotation matrices: reconstruct column 3 by taking cross product of first two columns
				B, K, T, _ = pred_rot_6D.shape
				# Flatten for conversion
				pred_rot_flat = pred_rot_6D.view(-1, 6)
				gt_rot_flat = gt_rot_6D.expand(B, K, T, 6).contiguous().view(-1, 6) #need to expand to match K dimension


				# Convert to rotation matrices
				pred_R = self.rot6d_to_rotmat_SO3(pred_rot_flat)  # shape: (B*K*T, 3, 3)
				gt_R = self.rot6d_to_rotmat_SO3(gt_rot_flat)      # shape: (B*K*T, 3, 3)

				# Reshape back
				pred_R = pred_R.view(B, K, T, 3, 3)
				gt_R = gt_R.view(B, K, T, 3, 3) # << now we have SO3
				# print(pred_R[0,0,0,...])
				# print(gt_R[0,0,0,...])

			if self.cfg.dimensions in [6,9]:
				# Compute relative rotation: R_diff = R_pred^T * R_gt.
				R_diff = torch.matmul(pred_R.transpose(-2, -1), gt_R)  # (B, K, T, 3, 3)

				#get rotation error angle theta of rot 3x3 rot matrix
				trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # (B, K, T) trace of each individual relative rotation matrix
				# Clamp to avoid numerical issues
				angular_error_theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))  # (B, K, T) take arccosine to get error angle cos(theta) = trace(R)-1 / 2 with clamping -1 + 1e-6, 1 - 1e-6
				loss_rotation = angular_error_theta.mean(dim=-1)  # average over time, so one loss per candidate K, shape (B, K)

				### (1+2) COMBINED DISTANCE LOSS (ROT AND TRANS)
				combined_error = loss_translation + loss_rotation  # (B, K) add translation and rotation error TODO normalise rot and trans loss
				loss_distance = combined_error.min(dim=1)[0].mean()  # some scalar << for whole batch, choose k_pred with lowest error, then average the error into one distance loss scalar

			"""General 2D/3D/6D/7D/9D code continues here"""
			### (3) UNCERTAINTY LOSS (original)
			loss_uncertainty = (
				torch.exp(-variance_estimation) *
				(k_alternative_preds - fut_traj_wpreds).norm(p=2, dim=-1).mean(dim=(1, 2)) + 
				variance_estimation
			).mean()
		
			# print(loss_uncertainty)
			
			
			### TOTAL LOSS
			loss = loss_distance * 50 + loss_uncertainty #make distance loss more important than uncertainty loss (?) TODO maybe not this much
			# print(loss)
			# exit()
			loss_total += loss.item()
			loss_dt += loss_distance.item()*50
			loss_dc += loss_uncertainty.item()

			if self.cfg.dimensions in [6,9]:
				loss_trans += loss_translation.min(dim=1)[0].mean().item()
				loss_rot += loss_rotation.min(dim=1)[0].mean().item()

			if self.cfg.dimensions in [7]:
				loss_trans += loss_translation.item()
				loss_rot += loss_rotation.item()
				
			self.opt.zero_grad()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()

			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total/count, loss_trans/count, loss_rot/count, loss_dt/count, loss_dc/count

	def _test_single_epoch(self):
		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE': [0] * len(timesteps),
			'FDE': [0] * len(timesteps),
			'ATE_trans': 0
		}
		# performance = { 'FDE': [0, 0, 0, 0],
		# 				'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)

		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				# LED initializer outputs
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
			
           		# Generate candidate future trajectories (B, K, T, 6)
				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				#fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1) #expand GT to match k_preds dimension (B, 1, T, 6)
				fut_traj_expanded = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)  # (B, K, T, D)
				
				# ATE metric (translation only)
				ate_results = self.compute_ate(
					pred_traj, 
					fut_traj.unsqueeze(1),  # Add K dimension to ground truth
					self.cfg.dimensions,
					self.traj_scale
				)

				# Add ATE results to performance metrics
				performance['ATE_trans'] += ate_results['ate_trans'] * batch_size

				# ---- Translation Error Metrics (already in your code) ----
				# Calculate traditional ADE/FDE metrics (from your original code)
				# Extract translational components for ADE/FDE calculation
				if self.cfg.dimensions in [2, 3]:
					pred_traj_trans = pred_traj
					fut_traj_trans = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)
				else:
					pred_traj_trans = pred_traj[..., :3]  # (B, K, T, 3)
					fut_traj_trans = fut_traj.unsqueeze(1).repeat(1, self.cfg.k_preds, 1, 1)[..., :3]  # (B, K, T, 3)
				
				distances = torch.norm(fut_traj_trans - pred_traj_trans, dim=-1) * self.traj_scale  ## Euclidian translation errors (B, K, T)
				# print('distances: ', distances)


            	# Compute ADE and FDE at different timesteps.
				# Here we compute ADE and FDE for time steps: every 5th and final
				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps
					"""
					1s: 5 * 1 - 1 = 4 → Requires at least 5 timesteps.
					2s: 5 * 2 - 1 = 9 → Requires at least 10 timesteps.
					3s: 5 * 3 - 1 = 14 → Requires at least 15 timesteps.
					4s: 5 * 4 - 1 = 19 → Requires at least 20 timesteps.
					"""
					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()
				samples += distances.shape[0]
			
		# Normalize ATE metrics by number of samples
		performance['ATE_trans'] /= samples

		return performance, samples

	def compute_ate(self, pred_trajectories, gt_trajectory, dimensions, traj_scale=1.0, align=False):
		"""
		Compute Absolute Trajectory Error (ATE) between predicted trajectories and ground truth.
		Translation error only, regardless of pose dimension.
		
		Args:
			pred_trajectories: Predicted trajectories of shape (B, K, T, D) where:
				B = batch size
				K = number of predictions per sample
				T = number of timesteps
				D = dimensions (2/3/6/7/9)
			gt_trajectory: Ground truth trajectory of shape (B, 1, T, D)
			dimensions: Dimensionality of the pose (2/3/6/7/9)
			traj_scale: Scale factor used during preprocessing
			align: Whether to align trajectories before computing error (default: False)
		
		Returns:
			Dictionary containing ATE metric:
			- 'ate_trans': Translational ATE (for all dimension types)
		"""
		# Initialize variables to store results
		B, K, T, D = pred_trajectories.shape
		results = {}
		
		# Extract translational component based on pose representation
		if dimensions in [2, 3]:
			# For 2D/3D trajectories, the entire representation is positional
			pred_trans = pred_trajectories * traj_scale  # (B, K, T, 2/3)
			gt_trans = gt_trajectory * traj_scale  # (B, 1, T, 2/3)
		else:
			# For 6D/7D/9D poses, extract just the translation part (first 3 dimensions)
			pred_trans = pred_trajectories[..., :3] * traj_scale  # (B, K, T, 3)
			gt_trans = gt_trajectory[..., :3] * traj_scale  # (B, 1, T, 3)
		
		# Compute translational errors for each candidate trajectory
		if align:
			# Align each predicted trajectory to ground truth before computing error
			aligned_pred_trans = []
			for b in range(B):
				for k in range(K):
					# Extract single trajectory
					pred_traj = pred_trans[b, k]  # (T, 2/3)
					gt_traj = gt_trans[b, 0]      # (T, 2/3)
					
					# Align trajectory using rigid transformation (rotation + translation)
					aligned_traj = self.ade_align_trajectory(pred_traj, gt_traj)
					aligned_pred_trans.append(aligned_traj)
			
			# Reshape back to original format
			aligned_pred_trans = torch.stack(aligned_pred_trans, dim=0)
			aligned_pred_trans = aligned_pred_trans.view(B, K, T, -1)
			
			# Compute errors using aligned trajectories
			trans_errors = torch.norm(gt_trans - aligned_pred_trans, dim=-1)  # (B, K, T)
		else:
			# Compute errors directly without alignment
			trans_errors = torch.norm(gt_trans - pred_trans, dim=-1)  # (B, K, T)
		
		# Select the best prediction among K candidates based on translation error only
		k_indices = trans_errors.mean(dim=-1).argmin(dim=-1)  # (B,)
		
		# Extract best trajectory for each sample
		batch_indices = torch.arange(B).to(pred_trajectories.device)
		if align:
			best_trans = aligned_pred_trans[batch_indices, k_indices]  # (B, T, D_trans)
		else:
			best_trans = pred_trans[batch_indices, k_indices]  # (B, T, D_trans)
		
		# Compute RMSE of the best trajectory compared to GT
		# This is the true ATE - square root of the mean squared error across entire trajectory
		ate_trans = torch.sqrt(((best_trans - gt_trans.squeeze(1))**2).sum(dim=-1).mean(dim=-1))  # (B,)
		
		# Average across the batch
		results['ate_trans'] = ate_trans.mean().item()
		
		return results

	def ade_align_trajectory(self, pred_traj, gt_traj):
		"""
		Align a predicted trajectory to ground truth using Umeyama's method.
		
		Args:
			pred_traj: Predicted trajectory of shape (T, D)
			gt_traj: Ground truth trajectory of shape (T, D)
			
		Returns:
			Aligned predicted trajectory of shape (T, D)
		"""
		# Center both trajectories (subtract mean)
		pred_centered = pred_traj - pred_traj.mean(dim=0, keepdim=True)
		gt_centered = gt_traj - gt_traj.mean(dim=0, keepdim=True)
		
		# Get translation components
		t_gt = gt_traj.mean(dim=0)
		t_pred = pred_traj.mean(dim=0)
		
		# Compute optimal rotation (simplified Umeyama's method for 2D/3D)
		# For 2D/3D points, we can use a simpler approach than full SVD
		W = torch.matmul(pred_centered.t(), gt_centered)
		U, _, Vt = torch.linalg.svd(W)
		
		# Ensure proper rotation matrix (handle reflection case)
		S = torch.eye(W.shape[0], device=pred_traj.device)
		if torch.det(torch.matmul(U, Vt)) < 0:
			S[-1, -1] = -1
		
		# Compute rotation matrix
		R = torch.matmul(torch.matmul(U, S), Vt)
		
		# Apply transformation: R(x-t_pred) + t_gt
		aligned_traj = torch.matmul(pred_centered, R.t()) + t_gt
		
		return aligned_traj

	def test_single_model(self, checkpoint_path = None):
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Independent/6_3_9D_Synthetic_Right_Curve_Random_Independent/models/best_checkpoint_epoch_65.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Walk/6_3_9D_Synthetic_Right_Curve_Random_Walk/models/best_checkpoint_epoch_80.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Right_Bias/6_3_9D_Synthetic_Right_Curve_Right_Bias/models/best_checkpoint_epoch_80.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Independent/6_3_9D_Synthetic_Straight_Random_Independent/models/best_checkpoint_epoch_75.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Walk/6_3_9D_Synthetic_Straight_Random_Walk/models/best_checkpoint_epoch_80.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Straight_Right_Bias/6_3_9D_Synthetic_Straight_Right_Bias/models/best_checkpoint_epoch_57.pth'
		checkpoint_path = './results/7_1_KittiPrior_10in_10out_k30/7_1_KittiPrior_10in_10out_k30/models/best_checkpoint_epoch_100.pth'
		# checkpoint_path = './results/7_1_PedestrianPrior_10in_10out_k30/7_1_PedestrianPrior_10in_10out_k30/models/7_1_PedestrianPrior_10in_10out_k30_best_checkpoint.pth'
		experiment_name = checkpoint_path.split('/')[3]

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()

		#pick evaluation horizons
		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE': [0] * len(timesteps),
			'FDE': [0] * len(timesteps)
		}

		# performance = { 'FDE': [0, 0, 0, 0],
		# 				'ADE': [0, 0, 0, 0]}
		samples = 0

		# Initialize motion prior
		# Determine rotation type based on your configuration
		rotation_type = 'none'  # Default, adjust based on your self.cfg.dim
		if hasattr(self.cfg, 'dim'):
			if self.cfg.dim == 7:
				rotation_type = 'quaternion'
			elif self.cfg.dim == 9:
				rotation_type = '6d'
			elif self.cfg.dim == 6:
				rotation_type = 'lie'
			elif self.cfg.dim in [2, 3]:
				rotation_type = 'none'
		
		motion_prior = MotionPriorGMM(
			max_components=5,
			min_samples_per_component=3,
			rotation_type=rotation_type
		)

		# Ensure reproducibility for testing
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(42)

		# hist_out_dir = './visualization/Temporal_Independence_KDE/Overfitting' #to save KDE densities on GT
		# all_densities_by_time = {t: [] for t in range(self.cfg.future_frames)}

		# past_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_past_traj.npy')).to(self.device)
		# traj_mask = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_traj_mask.npy')).to(self.device)
		# fut_traj = torch.from_numpy(np.load('/home/scur2440/MoTric/5_LED_Kitti_BetterCheckpoints/visualization/first_fut_traj.npy')).to(self.device)

		# sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
		# sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
		# loc = sample_prediction + mean_estimation[:, None]
		
		# k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

		### Regular code
		with torch.no_grad():
			for batch_idx, data in enumerate(self.test_loader):
				#print(data['fut_motion_3D']) #not scaled doen
				# print(data['fut_motion_3D'][5,:,:])
				# exit()
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				# print('past poses relative translation:', past_traj.shape, past_traj[0,:,:3])
				# print(past_traj.size())
				# first position past traj - should be relativised to [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00, 0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00]


				# Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			
				initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions


				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
				# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])
				# exit()

				# print(k_alternative_preds.size())  # (B, K, T, D)
				# print(fut_traj.size()) # (B, T, D)
				# print(past_traj.size()) # (32, 10, D)

				
				
				# ### ============= motion prior + visualisation =============

				           
				### BUILD GMM MOTION PRIOR FROM k_preds ###
				print(f"Building motion prior for batch {batch_idx}")
				print(f"k_alternative_preds shape: {k_alternative_preds.shape}")
				
				# Extract k samples for a specific trajectory and timestep
				# Example: batch 0, timestep 0 (t+1 predictions)
				batch_idx, timestep = 0, 0
				k_samples_t1 = k_alternative_preds[batch_idx, :, timestep, :]  # [K, D]
				print(k_samples_t1)
				
				# Build motion prior from just these K samples < for first timestep
				motion_prior_t1 = motion_prior.build_motion_prior_from_k_preds(k_samples_t1)
				
				# Print statistics about the fitted prior
				if motion_prior_t1.fitted:
					print(f"GMM fitted for batch {batch_idx}, timestep t+1:")
					print(f"  K samples shape: {k_samples_t1.shape}")
					print(f"  Number of components: {motion_prior_t1.gmm.n_components}")
					print(f"  Component weights: {motion_prior_t1.gmm.weights_}")
					print(f"  Prior prediction (weighted mean): {motion_prior_t1.get_prior_prediction()}")
					# VISUALIZE THE MOTION PRIOR
					# Extract past trajectory and GT future for this batch
					past_traj_viz = past_traj[batch_idx]  # [T_past, D]
					gt_future_viz = fut_traj[batch_idx, timestep:timestep+1]  # [1, D] - just the t+1 GT pose
					
					motion_prior_t1.visualize_motion_prior(
						k_samples=k_samples_t1,
						past_traj=past_traj_viz,
						gt_future=gt_future_viz,
						experiment_name=experiment_name,
						batch_idx=batch_idx,
						timestep=timestep,
						save_path="./visualization",
						show_plot=False
					)
					# Example: build another prior for a different timestep
					timestep_final = k_alternative_preds.shape[2] - 1  # final timestep
					k_samples_final = k_alternative_preds[batch_idx, :, timestep_final, :]  # [K, D]
					
					motion_prior_final = MotionPriorGMM(
						max_components=5,
						min_samples_per_component=3,
						rotation_type=rotation_type
					)
					motion_prior_final.build_motion_prior_from_k_preds(k_samples_final)
					
					print(f"GMM fitted for batch {batch_idx}, final timestep:")
					print(f"  K samples shape: {k_samples_final.shape}")
					print(f"  Number of components: {motion_prior_final.gmm.n_components}")
					print(f"  Component weights: {motion_prior_final.gmm.weights_}")
					
					# Visualize final timestep too
					gt_future_final = fut_traj[batch_idx, timestep_final:timestep_final+1]  # [1, D]
					motion_prior_final.visualize_motion_prior(
						k_samples=k_samples_final,
						past_traj=past_traj_viz,
						gt_future=gt_future_final,
						experiment_name=experiment_name,
						batch_idx=batch_idx,
						timestep=timestep_final,
						save_path="./visualization",
						show_plot=False
					)
				else:
					print(f"Failed to fit GMM for batch {batch_idx}, timestep t+1")

				exit()
				# self.compute_kde_and_vis_full_traj(k_alternative_preds)
				# initializer_preds_xy = initializer_preds[:,:,:,:2]
				# k_alternative_preds_xy = k_alternative_preds[:,:,:,:2]
				#print(k_alternative_preds_xy.size())


				# priors = self.compute_batch_motion_priors_kde_temporal_independence(k_alternative_preds_xy) #Currently assumes temporal independence
				# 																	#dictionary keys: traj index within batch (0-31); 
				# 	 																#lists: one KDE per predicted time step pose (e.g. 24 KDE's for all poses)
				# # # (B, K, T, 2)
				# # priors = self.compute_batch_motion_priors_kde_joined(k_alternative_preds) # currently ignores time completely/does not work

				# for i in range(24):
				# 	print(k_alternative_preds[0,0,i,:])

				## not used anymore - this actually gets the velocity, use raw traj instead
				# past_traj_relative = past_traj[:, :, 1*self.cfg.dimensions:2*self.cfg.dimensions]
				# past_traj_xy = past_traj_relative[:,:,:2]
				# fut_traj_xy = fut_traj[:,:,:2]

				###raw traj
				# raw_past = data['pre_motion_3D'][:, 0, :, :2]   # [B, past_frames, 2]
				# raw_fut  = data['fut_motion_3D'][:, 0, :, :2]   # [B, fut_frames,   2]
				# relative to the last observed point:
				# last_obs  = raw_past[:, -1:, :]                # [B, 1, 2]
				# past_traj_xy = (raw_past - last_obs) / self.traj_scale
				# fut_traj_xy  = (raw_fut  - last_obs) / self.traj_scale

				# print(k_alternative_preds_xy.size(), past_traj_xy.size(), fut_traj_xy.size())
				# print(k_alternative_preds[0,:,0,:])
				
				# self.compute_kde_and_vis_full_traj(k_alternative_preds_xy, past_traj_xy, fut_traj_xy, experiment_name, True, True)
				# exit()

				""" 
				for traj_idx in range(batch_size):

					for time in range(fut_traj.size(1)):
						single_pose_GT = fut_traj[traj_idx, time, :]

						single_pose_all_ks = k_alternative_preds[traj_idx, :, time, :]
						single_pose_all_ks_xy = single_pose_all_ks[:,:2]
						# print(single_pose_all_ks_xy.size())
						# single_pose_KDE = self.KDE_single_pose_outlier_filtered(single_pose_all_ks.detach().cpu().numpy()) #outlier filtering
						single_pose_KDE = priors[traj_idx][time] # KDE based on all k_preds for the pose #uncomment for no outlier filtering
						# print(single_pose_KDE)

						single_absolute_pose_past = past_traj[traj_idx, :, 1*self.cfg.dimensions:2*self.cfg.dimensions] #there are 3*dims poses - the first dim=n ones are absolute, the middle ones relative, then velocities. we need the middles ones-velocities					
						single_absolute_pose_past_xy = single_absolute_pose_past[:,:2] #9D starts with tx,ty,tz,r....
						# print(single_absolute_pose_past_xy.size())

						single_pose_GT_xy = single_pose_GT[:2]
						# print(single_pose_GT_xy.size())
						# print(single_pose_all_ks_xy)

						self.visualise_single_KDE_GT_Past(single_pose_all_ks_xy, single_pose_KDE, single_absolute_pose_past_xy, single_pose_GT_xy, time) 


						### Probability density
						GT_pose_np = single_pose_GT_xy.detach().cpu().numpy().reshape(-1, 1)
						density = single_pose_KDE(GT_pose_np)[0]

						all_densities_by_time[time].append(density)

						#print(f"Probability Density of GT pose at time {time}: {density}")
					exit()
				"""
				
				### Regular code continues
				fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.future_frames, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - k_alternative_preds, dim=-1) * self.traj_scale


				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()

				# for time_i in range(1, 5):
				# 	ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
				# 	fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
				# 	performance['ADE'][time_i-1] += ade.item()
				# 	performance['FDE'][time_i-1] += fde.item()

				samples += distances.shape[0]
				count += 1
					# if count==2:
					# 	break

			
			### save KDE density-per-t histograms
			self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)

			avg_densities_per_t = {t: np.array(all_densities_by_time[t]).mean() for t in range(self.cfg.future_frames)}
			print_log(f'avg_densities_per_t:\n{avg_densities_per_t}', self.log)

		for time_i in range(1,20,5):
			print_log('--ADE ({} time steps): {:.4f}\t--FDE ({} time steps): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		


	def test_single_model_basic(self, checkpoint_path = None):
			checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Independent/6_3_9D_Synthetic_Right_Curve_Random_Independent/models/best_checkpoint_epoch_65.pth'
			# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Walk/6_3_9D_Synthetic_Right_Curve_Random_Walk/models/best_checkpoint_epoch_80.pth'
			# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Right_Bias/6_3_9D_Synthetic_Right_Curve_Right_Bias/models/best_checkpoint_epoch_80.pth'
			# checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Independent/6_3_9D_Synthetic_Straight_Random_Independent/models/best_checkpoint_epoch_75.pth'
			# checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Walk/6_3_9D_Synthetic_Straight_Random_Walk/models/best_checkpoint_epoch_80.pth'
			# checkpoint_path ='./results/6_3_Synthetic_Straight_Right_Bias/6_3_9D_Synthetic_Straight_Right_Bias/models/best_checkpoint_epoch_57.pth'
			# experiment_name = checkpoint_path.split('/')[3]

			if checkpoint_path is not None:
				self.load_checkpoint(checkpoint_path)

			self.model.eval()
			self.model_initializer.eval()

			timesteps = list(range(5, self.cfg.future_frames, 5))
			if not timesteps or timesteps[-1] != self.cfg.future_frames:
				timesteps.append(self.cfg.future_frames)
				
			performance = {
				'ADE': [0.0] * len(timesteps),
				'FDE': [0.0] * len(timesteps),
				'ATE_trans': 0.0
			}

			samples = 0


			# Ensure reproducibility for testing
			def prepare_seed(rand_seed):
				np.random.seed(rand_seed)
				random.seed(rand_seed)
				torch.manual_seed(rand_seed)
				torch.cuda.manual_seed_all(rand_seed)
			prepare_seed(42)


			with torch.no_grad():
				for batch_idx, data in enumerate(self.test_loader):

					batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

					# Generate initial predictions using the initializer model
					sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
					sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				
					initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions


					# Generate the refined trajectory via the diffusion process
					k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
					# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])

					# print(k_alternative_preds.size())  # (B, K, T, D)
					# print(fut_traj.size()) # (B, T, D)
					# print(past_traj.size()) # (32, 10, D)
					
					
					# ### ============= motion prior + visualisation =============
					# priors = self.compute_batch_motion_priors_kde_temporal_independence(k_alternative_preds)
					# self.compute_kde_and_vis_full_traj(k_alternative_preds)
					# initializer_preds_xy = initializer_preds[:,:,:,:2]
					# k_alternative_preds_xy = k_alternative_preds[:,:,:,:2]
					# print(k_alternative_preds_xy.size())




			
					### Regular code continues
					fut_traj = fut_traj.unsqueeze(1)

					#calculate ATE
					ate_res = self.compute_ate(k_alternative_preds, fut_traj, self.cfg.dimensions, self.traj_scale)
					performance['ATE_trans'] += ate_res['ate_trans'] * batch_size

					#FDE and ADE for translation only
					if self.cfg.dimensions in [2,3]:
						pred_trans = k_alternative_preds  # already xy or xyz
						gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)
					else:
						pred_trans = k_alternative_preds[..., :3]
						gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)[..., :3]
					distances = torch.norm(gt_trans - pred_trans, dim=-1) * self.traj_scale


					for i, time in enumerate(timesteps):
						max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

						ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
						fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

						performance['ADE'][i] += ade.item()
						performance['FDE'][i] += fde.item()


					samples += batch_size


				
				### save KDE density-per-t histograms
				# self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)
				# normalize ATE
				performance['ATE_trans'] /= samples

				# print ADE/FDE every 5 steps, and ATE at the end
				timesteps = list(range(5, self.cfg.future_frames, 5))
				if not timesteps or timesteps[-1] != self.cfg.future_frames:
					timesteps.append(self.cfg.future_frames)

				for i, time_i in enumerate(timesteps):
					ade_avg = performance['ADE'][i] / samples
					fde_avg = performance['FDE'][i] / samples
					print_log('--ADE ({} steps): {:.4f}\t--FDE ({} steps): {:.4f}'.format(time_i, ade_avg, time_i, fde_avg), log=self.log)

				# total ATE
				print_log('--ATE translation: {:.4f}'.format(performance['ATE_trans']),log=self.log)




	def simulate_slam_baseline(self, data):
		"""
		Simulate a constant‐velocity + constant‐rotation 'SLAM' trajectory,
		then convert it into the same relative frame your data_preprocess uses.
		Returns: slam_rel  of shape (B*num_agents, future_frames, D)
		"""
		# unpack raw absolute past: [B, num_agents, past_frames, D]
		raw = data['pre_motion_3D'].to(self.device)
		B, N, Tp, D = raw.shape
		assert D == self.cfg.dimensions

		# flatten agents ↔ (B*N, Tp, D)
		raw_flat = raw.view(B * N, Tp, D)

		# split out pos vs rot
		prev_pos = raw_flat[:, -2, :3]      # (B*N,3)
		last_pos = raw_flat[:, -1, :3]      # (B*N,3)
		velocity = last_pos - prev_pos      # (B*N,3)

		rot_dims = D - 3
		last_rot = raw_flat[:, -1, 3:3+rot_dims]  # (B*N, rot_dims)

		# 1) build absolute SLAM predictions by stepping velocity & copying rotation
		slam_trans, slam_rot = [], []
		cur_pos = last_pos
		for _ in range(self.cfg.future_frames):
			cur_pos = cur_pos + velocity
			slam_trans.append(cur_pos.unsqueeze(1))     # (B*N,1,3)
			slam_rot.append(last_rot.unsqueeze(1))      # (B*N,1,rot_dims)

		slam_trans = torch.cat(slam_trans, dim=1)      # (B*N, T_f, 3)
		slam_rot   = torch.cat(slam_rot,   dim=1)      # (B*N, T_f, rot_dims)
		slam_abs   = torch.cat([slam_trans, slam_rot], dim=-1)  # (B*N, T_f, D)

		# 2) convert to relative exactly as data_preprocess does
		last_obs = raw_flat[:, -1, :]                  # (B*N, D)
		last_t   = last_obs[:, :3]                     # (B*N,3)
		last_r   = last_obs[:, 3:3+rot_dims]           # (B*N,rot_dims)

		# relative translation + scaling
		slam_t_rel = (slam_trans - last_t.unsqueeze(1)) / self.traj_scale  # (B*N, T_f, 3)

		# relative rotation by branch
		if D in [2, 3]:
			slam_rel = slam_t_rel                          # purely translational

		elif D == 6:
			# Lie‐algebra relative: r_rel = log(exp(-r_ref) exp(r))
			Bf, Tf, _ = slam_rot.shape
			rot_flat  = slam_rot.reshape(-1, 3)
			ref_flat  = last_r.unsqueeze(1).expand(-1, Tf, 3).reshape(-1, 3)
			rel_flat  = relative_lie_rotation(rot_flat, ref_flat)
			rel_rot   = rel_flat.view(Bf, Tf, 3)
			slam_rel  = torch.cat([slam_t_rel, rel_rot], dim=-1)

		elif D == 7:
			# quaternion relative: q_rel = q_ref^* * q
			qt_ref    = last_r.unsqueeze(1)                # (B*N,1,4)
			q_slam    = slam_rot                           # (B*N,T_f,4)
			rel_q     = quaternion_relative(qt_ref, q_slam)
			slam_rel  = torch.cat([slam_t_rel, rel_q], dim=-1)

		elif D == 9:
			# 6D‐to‐matrix, relative, then back to 6D
			Bf, Tf, _ = slam_rot.shape
			rot_flat  = slam_rot.reshape(-1, 6)
			ref_flat  = last_r.unsqueeze(1).expand(-1, Tf, 6).reshape(-1, 6)
			R_slam    = rot6d_to_rotmat(rot_flat)
			R_ref     = rot6d_to_rotmat(ref_flat)
			R_rel     = torch.matmul(R_ref.transpose(-2, -1), R_slam)
			rel6d     = rotmat_to_rot6d(R_rel).view(Bf, Tf, 6)
			slam_rel  = torch.cat([slam_t_rel, rel6d], dim=-1)

		else:
			raise NotImplementedError(f"Sim SLAM for D={D}")

		return slam_rel 

	def simulate_slam_baseline_linear(self, data):
		"""
		Fit a linear trajectory through all past positions (in least-squares sense)
		and extrapolate for future_frames, copying the last rotation.
		Returns slam_rel of shape (B*N, future_frames, D), in the same relative frame
		as data_preprocess.
		"""
		# 1) unpack absolute past: [B, N, Tp, D]
		raw = data['pre_motion_3D'].to(self.device)
		B, N, Tp, D = raw.shape
		assert D == self.cfg.dimensions

		# flatten agents to (B*N, Tp, D)
		raw_flat = raw.view(B * N, Tp, D)

		# extract past positions and rotations
		past_pos = raw_flat[:, :, :3]     # (B*N, Tp, 3)
		last_rot = raw_flat[:, -1, 3:]    # (B*N, rot_dims)

		# build a time‐index vector t = [0,1,...,Tp-1]
		t = torch.arange(Tp, device=self.device, dtype=past_pos.dtype)  # (Tp,)

		# compute means
		t_mean = t.mean()                 # scalar
		pos_mean = past_pos.mean(dim=1)   # (B*N, 3)

		# compute Var(t) and Cov(t, pos) for each batch‐agent
		# Var(t) = E[t^2] − E[t]^2
		t2_mean = (t*t).mean()
		var_t   = t2_mean - t_mean*t_mean  # scalar

		# Cov(t,pos) = E[t·pos] − E[t]·E[pos]
		#   first compute E[t·pos] per batch-agent:
		tp_mean = (t[None,:,None] * past_pos).mean(dim=1)  # (B*N, 3)
		cov_tp  = tp_mean - t_mean * pos_mean             # (B*N, 3)

		# slope = Cov(t,pos) / Var(t)  → (B*N, 3)
		slope = cov_tp / (var_t + 1e-6)

		# now build absolute forward predictions
		# last_pos = pos at t = Tp-1
		last_pos = past_pos[:, -1, :]  # (B*N, 3)

		future_steps = torch.arange(1, self.cfg.future_frames+1,
									device=self.device, dtype=slope.dtype)  # (T_f,)
		# for each step k, pred_pos = last_pos + slope * k
		# we want shape (B*N, T_f, 3):
		pred_trans = last_pos[:, None, :] + slope[:, None, :] * future_steps[None, :, None]

		# rotations: just copy the last observed rotation
		rot_dims = D - 3
		rot_repeat = last_rot.unsqueeze(1).expand(-1, self.cfg.future_frames, rot_dims)  # (B*N,T_f,rot_dims)

		# combine
		slam_abs = torch.cat([pred_trans, rot_repeat], dim=-1)  # (B*N, T_f, D)

		# 2) convert to relative frame exactly as in data_preprocess
		#    i.e. translation: (x_t - x_{Tp-1}) / scale, and relative rotation
		last_obs = raw_flat[:, -1, :]              # (B*N, D)
		last_t   = last_obs[:, :3]
		last_r   = last_obs[:, 3:3+rot_dims]

		# relative translation + scaling
		slam_t_rel = (pred_trans - last_t.unsqueeze(1)) / self.traj_scale  # (B*N, T_f, 3)

		# now attach the appropriate relative-rotation branch
		if D in [2,3]:
			slam_rel = slam_t_rel

		elif D == 6:
			# Lie‐algebra relative
			Bf, Tf, _ = rot_repeat.shape
			rot_flat = rot_repeat.reshape(-1,3)
			ref_flat = last_r.unsqueeze(1).expand(-1, Tf, 3).reshape(-1,3)
			rel_flat = relative_lie_rotation(rot_flat, ref_flat)
			rel_rot  = rel_flat.view(Bf, Tf, 3)
			slam_rel = torch.cat([slam_t_rel, rel_rot], dim=-1)

		elif D == 7:
			# quaternion relative
			qt_ref = last_r.unsqueeze(1)              # (B*N,1,4)
			q_slam = rot_repeat                       # (B*N,T_f,4)
			rel_q  = quaternion_relative(qt_ref, q_slam)
			slam_rel = torch.cat([slam_t_rel, rel_q], dim=-1)

		elif D == 9:
			# 6D→matrix→relative→6D
			Bf, Tf, _ = rot_repeat.shape
			rot_flat = rot_repeat.reshape(-1,6)
			ref_flat = last_r.unsqueeze(1).expand(-1, Tf, 6).reshape(-1,6)
			R_slam = rot6d_to_rotmat(rot_flat)
			R_ref  = rot6d_to_rotmat(ref_flat)
			R_rel  = R_ref.transpose(-2,-1) @ R_slam
			rel6d  = rotmat_to_rot6d(R_rel).view(Bf, Tf, 6)
			slam_rel = torch.cat([slam_t_rel, rel6d], dim=-1)

		else:
			raise NotImplementedError(f"Sim SLAM for D={D}")

		return slam_rel

	def visualize_bev_algorithm_vs_prior(
		self,
		past_traj: np.ndarray,    # (T_past,2)
		slam_traj: np.ndarray,    # (T_future,2)
		k_preds_xy: np.ndarray,   # (N,2)
		local_kde: CustomKDE,     # small-σ KDE
		save_path: str
	):
		"""
		Plots BEV of past, SLAM, samples, and the local KDE density.
		"""
		# 1) grid limits
		all_pts = np.vstack([k_preds_xy, past_traj, slam_traj])
		margin = 1.0
		x_min, x_max = all_pts[:,0].min() - margin, all_pts[:,0].max() + margin
		y_min, y_max = all_pts[:,1].min() - margin, all_pts[:,1].max() + margin

		# 2) meshgrid
		xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
		grid = np.vstack([xx.ravel(), yy.ravel()]).T

		# 3) evaluate local KDE on the grid
		zz = local_kde.pdf(grid).reshape(xx.shape)

		# 4) plotting
		plt.figure(figsize=(8,8))
		plt.contourf(xx, yy, zz, levels=50, cmap='viridis', zorder=1)

		# past trajectory
		plt.plot(
			past_traj[:,0], past_traj[:,1],
			'-o', color='tab:blue', label='Past poses', zorder=2
		)

		# SLAM prediction
		plt.plot(
			slam_traj[:,0], slam_traj[:,1],
			'-o', color='tab:orange', label='SLAM pred', zorder=2
		)

		# Leapfrog samples
		plt.scatter(
			k_preds_xy[:,0], k_preds_xy[:,1],
			s=5, alpha=0.4, color='white', edgecolor='k',
			label='Leapfrog samples', zorder=3
		)

		plt.axis('equal')
		plt.legend(loc='upper left')
		plt.title('BEV: Past / SLAM / Leapfrog + KDE density')
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close()

	def visualize_bev_algorithm_vs_corrected(
		self,
		past_traj: np.ndarray,       # (T_past,2)
		slam_traj: np.ndarray,       # (T_future,2)
		corrected_traj: np.ndarray,  # (T_future,2)
		k_preds_xy: np.ndarray,      # (N,2)
		local_kde: CustomKDE,        # small-σ KDE on k_preds_xy
		save_path: str
	):
		"""
		Plots BEV of past, SLAM, samples, local KDE density, and corrected SLAM.
		"""
		# 1) build limits
		all_pts = np.vstack([k_preds_xy, past_traj, slam_traj, corrected_traj])
		margin = 1.0
		x_min, x_max = all_pts[:,0].min() - margin, all_pts[:,0].max() + margin
		y_min, y_max = all_pts[:,1].min() - margin, all_pts[:,1].max() + margin

		# 2) grid
		xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
		grid = np.vstack([xx.ravel(), yy.ravel()]).T

		# 3) evaluate local KDE
		zz = local_kde.pdf(grid).reshape(xx.shape)

		# 4) plotting
		plt.figure(figsize=(8,8))
		plt.contourf(xx, yy, zz, levels=50, cmap='viridis', zorder=1)

		# past trajectory
		plt.plot(
			past_traj[:,0], past_traj[:,1],
			'-o', color='tab:blue', label='Past poses', zorder=2
		)
		# SLAM baseline
		plt.plot(
			slam_traj[:,0], slam_traj[:,1],
			'-o', color='tab:orange', label='SLAM pred', zorder=2
		)
		# corrected SLAM
		plt.plot(
			corrected_traj[:,0], corrected_traj[:,1],
			'-o', color='tab:green', label='Corrected SLAM', zorder=2
		)
		# Leapfrog samples
		plt.scatter(
			k_preds_xy[:,0], k_preds_xy[:,1],
			s=5, alpha=0.4, color='white', edgecolor='k',
			label='Leapfrog samples', zorder=3
		)

		plt.axis('equal')
		plt.legend(loc='upper left')
		plt.title('BEV: Past / SLAM / Leapfrog + KDE + Corrected')
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close()

	def mixture_prior_gradient(self, local_kde, global_kde, x, alpha):
		"""
		x: 1D array shape (3,) current pose
		returns ∇ log [ (1-α) p_local + α p_global ]
		"""
		# 1) evaluate densities
		p1 = local_kde.pdf(x[np.newaxis, :])[0]
		p2 = global_kde.pdf(x[np.newaxis, :])[0]
		p  = (1 - alpha)*p1 + alpha*p2 + 1e-12  # avoid zero

		# 2) get each gradient (your existing finite‐diff)
		g1 = self.prior_gradient_logl(local_kde, x)
		g2 = self.prior_gradient_logl(global_kde, x)

		# 3) combine
		numerator   = (1 - alpha)*p1*g1 + alpha*p2*g2
		return numerator / p


	def simulate_algorithm_and_correct_synthetic(self, checkpoint_path = None):
		ALGORITHM_SETTING = 'simulate_and_correct' # 'simulate', 'simulate_and_correct', ''
		VISUAL_KDE_ALGO = False # visualise algorithm vs KDE for first trajectory
		VISUAL_KDE_ALGO_CORR = True # visualise algorithm vs KDE vs correct algorithm for first trajectory
		
		#determined automatically
		bandw = None #0.5 #bandwidth for local KDE
		λ = None #0.5 #step size for gradient descent
		c = 3 # will move half the bandwidth


		#currently inactive
		# bandw_global = bandw * 10   # eg 10× wider
		# alpha = 0.0  # influence of global kde (between 0 and 1), eg 0.2


		print('[INFO] Entered new function')
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Independent/6_3_9D_Synthetic_Right_Curve_Random_Independent/models/best_checkpoint_epoch_65.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Random_Walk/6_3_9D_Synthetic_Right_Curve_Random_Walk/models/best_checkpoint_epoch_80.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Right_Curve_Right_Bias/6_3_9D_Synthetic_Right_Curve_Right_Bias/models/best_checkpoint_epoch_80.pth'
		checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Independent/6_3_9D_Synthetic_Straight_Random_Independent/models/best_checkpoint_epoch_75.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Straight_Random_Walk/6_3_9D_Synthetic_Straight_Random_Walk/models/best_checkpoint_epoch_80.pth'
		# checkpoint_path ='./results/6_3_Synthetic_Straight_Right_Bias/6_3_9D_Synthetic_Straight_Right_Bias/models/best_checkpoint_epoch_57.pth'
		experiment_name = checkpoint_path.split('/')[3]

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()

		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE':       [0.0] * len(timesteps),
			'FDE':       [0.0] * len(timesteps),
			'ATE_trans': 0.0
		}
		if ALGORITHM_SETTING:
			# metrics for SLAM
			performance_slam = {
				'ADE':       [0.0] * len(timesteps),
				'FDE':       [0.0] * len(timesteps),
				'ATE_trans': 0.0
			}

			if ALGORITHM_SETTING == 'simulate_and_correct':
				# metrics for KDE correction
				performance_corrected_slam = {
					'ADE':       [0.0] * len(timesteps),
					'FDE':       [0.0] * len(timesteps),
					'ATE_trans': 0.0
				}
				# def filter_outliers(pts, m=3.0):
				# 	# pts: (K, d)
				# 	median = np.median(pts, axis=0)
				# 	mad    = np.median(np.abs(pts - median), axis=0) + 1e-6
				# 	# keep only those within m*MAD in *all* dims
				# 	mask = np.all(np.abs(pts - median) <= m * mad, axis=1)
				# 	return pts[mask]
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


		samples = 0

		# Ensure reproducibility for testing
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(42)


		with torch.no_grad():
			for batch_idx, data in enumerate(self.test_loader):
				# 1) Regular leapfrog preprocessing
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				Bflat = past_traj.size(0)


				### --- AUTOMATIC BANDWIDTH & LAMBDA ESTIMATION ---
				# measure std of SLAM relative step sizes 
				raw = data['pre_motion_3D'].cpu().numpy()             # shape [B, N, Tp, D]
				# flatten B×N agents to B*N
				B_, N_, Tp, D_ = raw.shape
				raw_flat = raw.reshape(B_*N_, Tp, D_)

				# just translation dims
				past_abs = raw_flat[:, :, :3]                         # (B*N, Tp, 3)
				# frame‐to‐frame displacemnts over the past window
				deltas = past_abs[:, 1:, :] - past_abs[:, :-1, :]     # (B*N, Tp-1, 3)
				speeds = np.linalg.norm(deltas, axis=2)              # (B*N, Tp-1)
				std_slam = speeds.std()                            # overall std of SLAM motion
				print(std_slam)

				### -------------------------------------------------

				

				if ALGORITHM_SETTING:
					# --------- SLAM baseline simulation --------
					# 2) simulate SLAM baseline (relative coords)
					# slam_rel = self.simulate_slam_baseline(data)       # (B*N, T_f, D)
					slam_rel = self.simulate_slam_baseline_linear(data)       # (B*N, T_f, D)
					slam_preds = slam_rel.unsqueeze(1)                 # (B*N,1,T_f,D)

					# 3) compute SLAM ATE
					slam_ate = self.compute_ate(slam_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_slam['ATE_trans'] += slam_ate * Bflat

					# 4) compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = slam_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = slam_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_slam['ADE'][i] += ade_slam.item()
						performance_slam['FDE'][i] += fde_slam.item()
					# -------------------- SLAM -----------------------

				# 4. Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			
				initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions

				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
				# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])

				# print(k_alternative_preds.size())  # (B, K, T, D)
				# print(fut_traj.size()) # (B, T, D)
				# print(past_traj.size()) # (32, 10, D)


				### --------------- Correct algorithm/SLAM predictions with KDE prior built from LED predictions ---------------
				if ALGORITHM_SETTING == 'simulate_and_correct':

					# 5) Get KDEs on translation dims (one per timestep)
					k_trans = k_alternative_preds[...,:3].detach().cpu().numpy()  # (B*N, K, T, 3)
					batch_kdes = []
					for b in range(k_trans.shape[0]):

						### ----------- determine automatic bandwidth and λ for this batch: -----------
						K, T_f, _ = k_trans[b].shape
						n = K
						# Silverman's 2D rule-of-thumb: h = 1.06 * sigma * n^(-1/5)
						bandw = 1.06 * std_slam * (n ** (-1/5))
						# step size ~ half a sigma:
						λ = c * bandw
						### ------------------------------------------


						per_t = []

						for t in range(self.cfg.future_frames):
							# if t ==0 and b==0:
							# 	print('k_trans at b 0 t 0:', k_trans[b,:,t,:].shape, k_trans[b,:,t,:])
							k_samples = k_trans[b,:,t,:]

							k_pts_filtered = filter_outliers(k_samples, min_neighbors=1) #k needs to have one neighbour within 1.5 x median distance to not be considered an outlier
							### Enable to inspect filtering
							# print('Before and after neighbourhood filtering: ', k_samples.shape, filtered.shape)
							# print('All k:', k_samples)
							# k_set = set(map(tuple, k_samples))
							# f_set = set(map(tuple, filtered))
							# removed = np.array([pt for pt in k_set - f_set]) # points that were removed
							# print(f"Removed: {removed}")



							# kde = CustomKDE(filtered, bandwidth=bandw, range_factor=1.0)

							# original, small bandwidth KDE (per-timestep)
							local_kde  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global KDE: same data but very wide bandwidth
							global_kde = None # CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							per_t.append((local_kde, global_kde))

						batch_kdes.append(per_t)


						# --------- Visualise Algorithm vs KDE prior ----------
						if VISUAL_KDE_ALGO and b == 0:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t      = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							# visualize all of them + density
							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_prior(past_np, slam_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved BEV plot to {save_path}")
							exit()
						# -------------------------------------------------------

					# 6) Correct SLAM with KDE gradients
					slam_rel_np = slam_rel.cpu().numpy()  # [B*N, T, D]
					corrected_rel = slam_rel.clone()
					# print('λ:', λ)

					for b in range(slam_rel_np.shape[0]):
						# print('batch:', b)
						for t in range(slam_rel_np.shape[1]):
							# print('timestep:', t)
							local_kde, global_kde = batch_kdes[b][t]
							pose = slam_rel_np[b,t,:3]  # only translation
							# print('slam original:', pose)
							
							# ----- find mode among the K samples -----
							# density = kde.pdf(pose[np.newaxis, :])[0]  # density at SLAM
							# k_samples = k_trans[b, :, t, :]              # shape (K, 3)
							# dens_s  = kde.pdf(k_samples)                 # (K,)
							# idx     = np.argmax(dens_s)
							# mode    = k_samples[idx]                     # (3,)
							# mode_d  = dens_s[idx]
							# print everything
							# print(f" SLAM pose: {pose}   density: {density:.6f}")
							# print(f" KDE mode: {mode}   mode density: {mode_d:.6f}")



							# ----- gradient‐step correction -----
							grad = self.prior_gradient_logl(local_kde, pose) #previous - single kde
							### global & local KDE - inactive #XXX activate for dual KDE
							# grad = self.mixture_prior_gradient(local_kde, global_kde, pose, alpha) #mixure of global and local kde. will still go to zero eventually but only when global max bandwidth is reached
							# print('grad:', grad)
							norm = np.linalg.norm(grad)
							# print('norm:', norm)
							if norm>1: grad/=norm
							# print('grad:', grad)
							corrected_rel[b,t,:3] = slam_rel[b,t,:3] + λ * torch.from_numpy(grad).to(slam_rel.device)
							# print('corrected pose:', corrected_rel[b,t,:3])
							# print('density at corrected:', kde.pdf(corrected_rel[b,t,:3].detach().cpu().numpy()))
							# exit()


						# --------- Visualise Algorithm vs KDE prior vs corrected algorithm poses ----------
						if VISUAL_KDE_ALGO_CORR and b == 15:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							corrected_np = corrected_rel[b,:, :2].cpu().numpy()    # (T_future, 2)

							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_vs_corrected_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_corrected(past_np, slam_np, corrected_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved comparison plot to {save_path}")
							exit()
						# -------------------------------------------------------


					corrected_preds = corrected_rel.unsqueeze(1) # [B*N,1,T,D]

					# 7) Corrected SLAM metrics
					# SLAM ATE
					slam_ate = self.compute_ate(corrected_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_corrected_slam['ATE_trans'] += slam_ate * Bflat

					# Compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = corrected_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = corrected_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_corrected_slam['ADE'][i] += ade_slam.item()
						performance_corrected_slam['FDE'][i] += fde_slam.item()

				### --------------- End of correction ----------------

				samples += Bflat
				fut_traj = fut_traj.unsqueeze(1)

				# ATE for Leapfrog
				ate_res = self.compute_ate(k_alternative_preds, fut_traj, self.cfg.dimensions, self.traj_scale)
				performance['ATE_trans'] += ate_res['ate_trans'] * Bflat

				# FDE and ADE for translation only on Leapfrog
				if self.cfg.dimensions in [2,3]:
					pred_trans = k_alternative_preds  # already xy or xyz
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)
				else:
					pred_trans = k_alternative_preds[..., :3]
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)[..., :3]
				distances = torch.norm(gt_trans - pred_trans, dim=-1) * self.traj_scale
				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()



			
			### save KDE density-per-t histograms
			# self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)

			### Finalise and print results
			# Leapfrog
			performance['ATE_trans'] /= samples # normalize ATE
			for i, t in enumerate(timesteps):
				
				# convert to Python floats so float-formatting won’t choke on a numpy scalar/array
				ade_avg = performance['ADE'][i] / samples
				fde_avg = performance['FDE'][i] / samples
				print_log(f'--Leapfrog ADE ({t}): {ade_avg:.4f}\tFDE: {fde_avg:.4f}', self.log)



			print_log(f'--Leapfrog ATE: {performance["ATE_trans"]:.4f}', self.log)


			# SLAM
			if ALGORITHM_SETTING:

				# Raw algorithm performance
				performance_slam['ATE_trans'] /= samples
				for i, t in enumerate(timesteps):
					ade_s = performance_slam['ADE'][i] / samples
					fde_s = performance_slam['FDE'][i] / samples
					print_log(f'--SLAM ADE ({t}): {ade_s:.4f}\tFDE: {fde_s:.4f}', self.log)
				print_log(f'--SLAM ATE: {performance_slam["ATE_trans"]:.4f}', self.log)
			
				# Algorithm performance corrected with KDE
				if ALGORITHM_SETTING == 'simulate_and_correct':

					performance_corrected_slam['ATE_trans'] /= samples
					for i, t in enumerate(timesteps):
						ade_s_corr = performance_corrected_slam['ADE'][i] / samples
						fde_s_corr = performance_corrected_slam['FDE'][i] / samples
						print_log(f'--SLAM ADE CORRECTED ({t}): {ade_s_corr:.4f}\tFDE: {fde_s_corr:.4f}', self.log)
					print_log(f'--SLAM ATE CORRECTED: {performance_corrected_slam["ATE_trans"]:.4f}', self.log)

	def simulate_algorithm_and_correct_kitti_clusters_for_bandw(self, checkpoint_path = None):
		ALGORITHM_SETTING = 'simulate_and_correct' # 'simulate', 'simulate_and_correct', ''
		VISUAL_KDE_ALGO = False # visualise algorithm vs KDE for first trajectory
		VISUAL_KDE_ALGO_CORR = True # visualise algorithm vs KDE vs correct algorithm for first trajectory
		
		#determined automatically
		# bandw = 0.5 #0.5 #bandwidth for local KDE
		# λ = 0.5 #0.5 #step size for gradient descent
		# c = 1 # will move half the bandwidt
		
		cluster_to_h = {
			0: 1,
			1: 1,
			2: 1,
			3: 1,
			4: 0.5,
		}
		cluster_to_lambda = { j: cluster_to_h[j] * 0.5 for j in cluster_to_h }


		#currently inactive
		# bandw_global = bandw * 10   # eg 10× wider
		# alpha = 0.0  # influence of global kde (between 0 and 1), eg 0.2


		print('[INFO] Entered new function')
		checkpoint_path ='./results/7_1_KittiPrior_10in_20out_k30/7_1_KittiPrior_10in_20out_k30_EXCL04/models/best_checkpoint_epoch_48.pth'
		checkpoint_path = './results/7_1_KittiPrior_10in_20out_k30/7_1_KittiPrior_10in_20out_k30_Overfit/models/best_checkpoint_epoch_26.pth'
		experiment_name = checkpoint_path.split('/')[3]

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()
		self.clusterer = OfflineTrajectoryClusterer(self, num_clusters=5, pca_dim=20)
		self.clusterer.load()

		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE':       [0.0] * len(timesteps),
			'FDE':       [0.0] * len(timesteps),
			'ATE_trans': 0.0
		}
		if ALGORITHM_SETTING:
			# metrics for SLAM
			performance_slam = {
				'ADE':       [0.0] * len(timesteps),
				'FDE':       [0.0] * len(timesteps),
				'ATE_trans': 0.0
			}

			if ALGORITHM_SETTING == 'simulate_and_correct':
				# metrics for KDE correction
				performance_corrected_slam = {
					'ADE':       [0.0] * len(timesteps),
					'FDE':       [0.0] * len(timesteps),
					'ATE_trans': 0.0
				}
				# def filter_outliers(pts, m=3.0):
				# 	# pts: (K, d)
				# 	median = np.median(pts, axis=0)
				# 	mad    = np.median(np.abs(pts - median), axis=0) + 1e-6
				# 	# keep only those within m*MAD in *all* dims
				# 	mask = np.all(np.abs(pts - median) <= m * mad, axis=1)
				# 	return pts[mask]
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


		samples = 0

		# Ensure reproducibility for testing
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(42)


		with torch.no_grad():
			for batch_idx, data in enumerate(self.test_loader):
				# 1) Regular leapfrog preprocessing
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				Bflat = past_traj.size(0)


				# assign each trajectory in the batch to a cluster
				labels = self.clusterer.predict_cluster(past_traj)  # shape: (B,)
				# look up pre‐tuned bandwidths & step‐sizes per cluster
				# assume you have dicts cluster_to_h and cluster_to_lambda
				bandw_batch  = np.array([ cluster_to_h[l]      for l in labels ])  # shape (B,)
				lambda_batch = np.array([ cluster_to_lambda[l] for l in labels ])  # shape (B,)



				### --- AUTOMATIC BANDWIDTH & LAMBDA ESTIMATION ---
				# measure std of SLAM relative step sizes 
				# raw = data['pre_motion_3D'].cpu().numpy()             # shape [B, N, Tp, D]
				# # flatten B×N agents to B*N
				# B_, N_, Tp, D_ = raw.shape
				# raw_flat = raw.reshape(B_*N_, Tp, D_)

				# # just translation dims
				# past_abs = raw_flat[:, :, :3]                         # (B*N, Tp, 3)
				# # frame‐to‐frame displacemnts over the past window
				# deltas = past_abs[:, 1:, :] - past_abs[:, :-1, :]     # (B*N, Tp-1, 3)
				# speeds = np.linalg.norm(deltas, axis=2)              # (B*N, Tp-1)
				# std_slam = speeds.std()                            # overall std of SLAM motion
				# print(std_slam)

				### -------------------------------------------------

				

				if ALGORITHM_SETTING:
					# --------- SLAM baseline simulation --------
					# 2) simulate SLAM baseline (relative coords)
					# slam_rel = self.simulate_slam_baseline(data)       # (B*N, T_f, D)
					slam_rel = self.simulate_slam_baseline_linear(data)       # (B*N, T_f, D)
					slam_preds = slam_rel.unsqueeze(1)                 # (B*N,1,T_f,D)

					# 3) compute SLAM ATE
					slam_ate = self.compute_ate(slam_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_slam['ATE_trans'] += slam_ate * Bflat

					# 4) compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = slam_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = slam_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_slam['ADE'][i] += ade_slam.item()
						performance_slam['FDE'][i] += fde_slam.item()
					# -------------------- SLAM -----------------------

				# 4. Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			
				initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions

				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
				# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])

				# print(k_alternative_preds.size())  # (B, K, T, D)
				# print(fut_traj.size()) # (B, T, D)
				# print(past_traj.size()) # (32, 10, D)


				### --------------- Correct algorithm/SLAM predictions with KDE prior built from LED predictions ---------------
				if ALGORITHM_SETTING == 'simulate_and_correct':

					# 5) Get KDEs on translation dims (one per timestep)
					k_trans = k_alternative_preds[...,:3].detach().cpu().numpy()  # (B*N, K, T, 3)
					batch_kdes = []
					for b in range(k_trans.shape[0]):

						### ----------- determine automatic bandwidth and λ for this batch: -----------
						bandw = bandw_batch[b]
						λ = lambda_batch[b]
						# if bandw is None or λ is None: # std heuristic
						# 	print('[INFO] Determine bandwidth automatically')
						# 	K, T_f, _ = k_trans[b].shape
						# 	n = K
						# 	# Silverman's 2D rule-of-thumb: h = 1.06 * sigma * n^(-1/5)
						# 	bandw = 1.06 * std_slam * (n ** (-1/5))
						# 	# step size ~ half a sigma:
						# 	λ = c * bandw
						# 	### ------------------------------------------


						per_t = []

						for t in range(self.cfg.future_frames):
							# if t ==0 and b==0:
							# 	print('k_trans at b 0 t 0:', k_trans[b,:,t,:].shape, k_trans[b,:,t,:])
							k_samples = k_trans[b,:,t,:]

							k_pts_filtered = filter_outliers(k_samples, min_neighbors=1) #k needs to have one neighbour within 1.5 x median distance to not be considered an outlier
							### Enable to inspect filtering
							# print('Before and after neighbourhood filtering: ', k_samples.shape, filtered.shape)
							# print('All k:', k_samples)
							# k_set = set(map(tuple, k_samples))
							# f_set = set(map(tuple, filtered))
							# removed = np.array([pt for pt in k_set - f_set]) # points that were removed
							# print(f"Removed: {removed}")



							# kde = CustomKDE(filtered, bandwidth=bandw, range_factor=1.0)

							# original, small bandwidth KDE (per-timestep)
							local_kde  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global KDE: same data but very wide bandwidth
							global_kde = None # CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							per_t.append((local_kde, global_kde))

						batch_kdes.append(per_t)


						# --------- Visualise Algorithm vs KDE prior ----------
						if VISUAL_KDE_ALGO and b == 0:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t      = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							# visualize all of them + density
							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_prior(past_np, slam_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved BEV plot to {save_path}")
							exit()
						# -------------------------------------------------------

					# 6) Correct SLAM with KDE gradients
					slam_rel_np = slam_rel.cpu().numpy()  # [B*N, T, D]
					corrected_rel = slam_rel.clone()
					# print('λ:', λ)

					for b in range(slam_rel_np.shape[0]):
						# print('batch:', b)
						for t in range(slam_rel_np.shape[1]):
							# print('timestep:', t)
							local_kde, global_kde = batch_kdes[b][t]
							pose = slam_rel_np[b,t,:3]  # only translation
							# print('slam original:', pose)
							
							# ----- find mode among the K samples -----
							# density = kde.pdf(pose[np.newaxis, :])[0]  # density at SLAM
							# k_samples = k_trans[b, :, t, :]              # shape (K, 3)
							# dens_s  = kde.pdf(k_samples)                 # (K,)
							# idx     = np.argmax(dens_s)
							# mode    = k_samples[idx]                     # (3,)
							# mode_d  = dens_s[idx]
							# print everything
							# print(f" SLAM pose: {pose}   density: {density:.6f}")
							# print(f" KDE mode: {mode}   mode density: {mode_d:.6f}")



							# ----- gradient‐step correction -----
							grad = self.prior_gradient_logl(local_kde, pose) #previous - single kde
							### global & local KDE - inactive #XXX activate for dual KDE
							# grad = self.mixture_prior_gradient(local_kde, global_kde, pose, alpha) #mixure of global and local kde. will still go to zero eventually but only when global max bandwidth is reached
							# print('grad:', grad)
							norm = np.linalg.norm(grad)
							# print('norm:', norm)
							if norm>1: grad/=norm
							# print('grad:', grad)
							corrected_rel[b,t,:3] = slam_rel[b,t,:3] + λ * torch.from_numpy(grad).to(slam_rel.device)
							# print('corrected pose:', corrected_rel[b,t,:3])
							# print('density at corrected:', kde.pdf(corrected_rel[b,t,:3].detach().cpu().numpy()))
							# exit()


						# --------- Visualise Algorithm vs KDE prior vs corrected algorithm poses ----------
						if VISUAL_KDE_ALGO_CORR and b == 7:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							corrected_np = corrected_rel[b,:, :2].cpu().numpy()    # (T_future, 2)

							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_vs_corrected_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_corrected(past_np, slam_np, corrected_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved comparison plot to {save_path}")
							exit()
						# -------------------------------------------------------


					corrected_preds = corrected_rel.unsqueeze(1) # [B*N,1,T,D]

					# 7) Corrected SLAM metrics
					# SLAM ATE
					slam_ate = self.compute_ate(corrected_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_corrected_slam['ATE_trans'] += slam_ate * Bflat

					# Compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = corrected_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = corrected_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_corrected_slam['ADE'][i] += ade_slam.item()
						performance_corrected_slam['FDE'][i] += fde_slam.item()

				### --------------- End of correction ----------------

				samples += Bflat
				fut_traj = fut_traj.unsqueeze(1)

				# ATE for Leapfrog
				ate_res = self.compute_ate(k_alternative_preds, fut_traj, self.cfg.dimensions, self.traj_scale)
				performance['ATE_trans'] += ate_res['ate_trans'] * Bflat

				# FDE and ADE for translation only on Leapfrog
				if self.cfg.dimensions in [2,3]:
					pred_trans = k_alternative_preds  # already xy or xyz
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)
				else:
					pred_trans = k_alternative_preds[..., :3]
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)[..., :3]
				distances = torch.norm(gt_trans - pred_trans, dim=-1) * self.traj_scale
				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()



			
			### save KDE density-per-t histograms
			# self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)

			### Finalise and print results
			# Leapfrog
			performance['ATE_trans'] /= samples # normalize ATE
			for i, t in enumerate(timesteps):
				
				# convert to Python floats so float-formatting won’t choke on a numpy scalar/array
				ade_avg = performance['ADE'][i] / samples
				fde_avg = performance['FDE'][i] / samples
				print_log(f'--Leapfrog ADE ({t}): {ade_avg:.4f}\tFDE: {fde_avg:.4f}', self.log)



			print_log(f'--Leapfrog ATE: {performance["ATE_trans"]:.4f}', self.log)


			# SLAM
			if ALGORITHM_SETTING:

				# Raw algorithm performance
				performance_slam['ATE_trans'] /= samples
				for i, t in enumerate(timesteps):
					ade_s = performance_slam['ADE'][i] / samples
					fde_s = performance_slam['FDE'][i] / samples
					print_log(f'--SLAM ADE ({t}): {ade_s:.4f}\tFDE: {fde_s:.4f}', self.log)
				print_log(f'--SLAM ATE: {performance_slam["ATE_trans"]:.4f}', self.log)
			
				# Algorithm performance corrected with KDE
				if ALGORITHM_SETTING == 'simulate_and_correct':

					performance_corrected_slam['ATE_trans'] /= samples
					for i, t in enumerate(timesteps):
						ade_s_corr = performance_corrected_slam['ADE'][i] / samples
						fde_s_corr = performance_corrected_slam['FDE'][i] / samples
						print_log(f'--SLAM ADE CORRECTED ({t}): {ade_s_corr:.4f}\tFDE: {fde_s_corr:.4f}', self.log)
					print_log(f'--SLAM ATE CORRECTED: {performance_corrected_slam["ATE_trans"]:.4f}', self.log)

	def simulate_algorithm_and_correct_kitti_std_for_band(self, checkpoint_path = None):
		ALGORITHM_SETTING = 'simulate_and_correct' # 'simulate', 'simulate_and_correct', ''
		VISUAL_KDE_ALGO = False # visualise algorithm vs KDE for first trajectory
		VISUAL_KDE_ALGO_CORR = False # visualise algorithm vs KDE vs correct algorithm for first trajectory
		
		#determined automatically
		bandw = 0.5 #0.5 #bandwidth for local KDE
		λ = 0.5 #0.5 #step size for gradient descent
		c = 1 # will move half the bandwidth


		#currently inactive
		# bandw_global = bandw * 10   # eg 10× wider
		# alpha = 0.0  # influence of global kde (between 0 and 1), eg 0.2


		print('[INFO] Entered new function')
		checkpoint_path ='./results/7_1_KittiPrior_10in_20out_k30/7_1_KittiPrior_10in_20out_k30_EXCL04/models/best_checkpoint_epoch_48.pth'
		experiment_name = checkpoint_path.split('/')[3]

		if checkpoint_path is not None:
			self.load_checkpoint(checkpoint_path)

		self.model.eval()
		self.model_initializer.eval()

		### most recent, to activate XXX
		# clusterer = OfflineTrajectoryClusterer(self, num_clusters=5, pca_dim=20)
		# clusterer.load()  # loads PCA+KMeans
		# with open('kde_hyperparams.pkl','rb') as f:
		# 	maps = pickle.load(f)
		# cluster_to_h, cluster_to_lambda = maps['h'], maps['λ']

		# # for each incoming past_traj:
		# label = clusterer.predict_cluster(past_traj[None])[0]
		# h  = cluster_to_h[label]
		# λ  = cluster_to_lambda[label]


		# self.clusterer = OfflineTrajectoryClusterer(self, num_clusters=5, pca_dim=20)
		# self.clusterer.load()

		timesteps = list(range(5, self.cfg.future_frames, 5))
		if not timesteps or timesteps[-1] != self.cfg.future_frames:
			timesteps.append(self.cfg.future_frames)
			
		performance = {
			'ADE':       [0.0] * len(timesteps),
			'FDE':       [0.0] * len(timesteps),
			'ATE_trans': 0.0
		}
		if ALGORITHM_SETTING:
			# metrics for SLAM
			performance_slam = {
				'ADE':       [0.0] * len(timesteps),
				'FDE':       [0.0] * len(timesteps),
				'ATE_trans': 0.0
			}

			if ALGORITHM_SETTING == 'simulate_and_correct':
				# metrics for KDE correction
				performance_corrected_slam = {
					'ADE':       [0.0] * len(timesteps),
					'FDE':       [0.0] * len(timesteps),
					'ATE_trans': 0.0
				}
				# def filter_outliers(pts, m=3.0):
				# 	# pts: (K, d)
				# 	median = np.median(pts, axis=0)
				# 	mad    = np.median(np.abs(pts - median), axis=0) + 1e-6
				# 	# keep only those within m*MAD in *all* dims
				# 	mask = np.all(np.abs(pts - median) <= m * mad, axis=1)
				# 	return pts[mask]
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


		samples = 0

		# Ensure reproducibility for testing
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(42)


		with torch.no_grad():
			for batch_idx, data in enumerate(self.test_loader):
				# 1) Regular leapfrog preprocessing
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
				Bflat = past_traj.size(0)


				### --- AUTOMATIC BANDWIDTH & LAMBDA ESTIMATION ---
				# measure std of SLAM relative step sizes 
				raw = data['pre_motion_3D'].cpu().numpy()             # shape [B, N, Tp, D]
				# flatten B×N agents to B*N
				B_, N_, Tp, D_ = raw.shape
				raw_flat = raw.reshape(B_*N_, Tp, D_)

				# just translation dims
				past_abs = raw_flat[:, :, :3]                         # (B*N, Tp, 3)
				# frame‐to‐frame displacemnts over the past window
				deltas = past_abs[:, 1:, :] - past_abs[:, :-1, :]     # (B*N, Tp-1, 3)
				speeds = np.linalg.norm(deltas, axis=2)              # (B*N, Tp-1)
				std_slam = speeds.std()                            # overall std of SLAM motion
				print(std_slam)

				### -------------------------------------------------

				

				if ALGORITHM_SETTING:
					# --------- SLAM baseline simulation --------
					# 2) simulate SLAM baseline (relative coords)
					# slam_rel = self.simulate_slam_baseline(data)       # (B*N, T_f, D)
					slam_rel = self.simulate_slam_baseline_linear(data)       # (B*N, T_f, D)
					slam_preds = slam_rel.unsqueeze(1)                 # (B*N,1,T_f,D)

					# 3) compute SLAM ATE
					slam_ate = self.compute_ate(slam_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_slam['ATE_trans'] += slam_ate * Bflat

					# 4) compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = slam_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = slam_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_slam['ADE'][i] += ade_slam.item()
						performance_slam['FDE'][i] += fde_slam.item()
					# -------------------- SLAM -----------------------

				# 4. Generate initial predictions using the initializer model
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			
				initializer_preds = sample_prediction + mean_estimation[:, None] #initialiser predictions

				# Generate the refined trajectory via the diffusion process
				k_alternative_preds = self.p_sample_loop_accelerate(past_traj, traj_mask, initializer_preds)
				# print('k_alternative_preds first T:',k_alternative_preds[0,:,0,:3])

				# print(k_alternative_preds.size())  # (B, K, T, D)
				# print(fut_traj.size()) # (B, T, D)
				# print(past_traj.size()) # (32, 10, D)


				### --------------- Correct algorithm/SLAM predictions with KDE prior built from LED predictions ---------------
				if ALGORITHM_SETTING == 'simulate_and_correct':

					# 5) Get KDEs on translation dims (one per timestep)
					k_trans = k_alternative_preds[...,:3].detach().cpu().numpy()  # (B*N, K, T, 3)
					batch_kdes = []
					for b in range(k_trans.shape[0]):

						### ----------- determine automatic bandwidth and λ for this batch: -----------
						if bandw is None or λ is None:
							print('[INFO] Determine bandwidth automatically')
							K, T_f, _ = k_trans[b].shape
							n = K
							# Silverman's 2D rule-of-thumb: h = 1.06 * sigma * n^(-1/5)
							bandw = 1.06 * std_slam * (n ** (-1/5))
							# step size ~ half a sigma:
							λ = c * bandw
							### ------------------------------------------


						per_t = []

						for t in range(self.cfg.future_frames):
							# if t ==0 and b==0:
							# 	print('k_trans at b 0 t 0:', k_trans[b,:,t,:].shape, k_trans[b,:,t,:])
							k_samples = k_trans[b,:,t,:]

							k_pts_filtered = filter_outliers(k_samples, min_neighbors=1) #k needs to have one neighbour within 1.5 x median distance to not be considered an outlier
							### Enable to inspect filtering
							# print('Before and after neighbourhood filtering: ', k_samples.shape, filtered.shape)
							# print('All k:', k_samples)
							# k_set = set(map(tuple, k_samples))
							# f_set = set(map(tuple, filtered))
							# removed = np.array([pt for pt in k_set - f_set]) # points that were removed
							# print(f"Removed: {removed}")



							# kde = CustomKDE(filtered, bandwidth=bandw, range_factor=1.0)

							# original, small bandwidth KDE (per-timestep)
							local_kde  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global KDE: same data but very wide bandwidth
							global_kde = None # CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							per_t.append((local_kde, global_kde))

						batch_kdes.append(per_t)


						# --------- Visualise Algorithm vs KDE prior ----------
						if VISUAL_KDE_ALGO and b == 0:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t      = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							# visualize all of them + density
							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_prior(past_np, slam_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved BEV plot to {save_path}")
							exit()
						# -------------------------------------------------------

					# 6) Correct SLAM with KDE gradients
					slam_rel_np = slam_rel.cpu().numpy()  # [B*N, T, D]
					corrected_rel = slam_rel.clone()
					# print('λ:', λ)

					for b in range(slam_rel_np.shape[0]):
						# print('batch:', b)
						for t in range(slam_rel_np.shape[1]):
							# print('timestep:', t)
							local_kde, global_kde = batch_kdes[b][t]
							pose = slam_rel_np[b,t,:3]  # only translation
							# print('slam original:', pose)
							
							# ----- find mode among the K samples -----
							# density = kde.pdf(pose[np.newaxis, :])[0]  # density at SLAM
							# k_samples = k_trans[b, :, t, :]              # shape (K, 3)
							# dens_s  = kde.pdf(k_samples)                 # (K,)
							# idx     = np.argmax(dens_s)
							# mode    = k_samples[idx]                     # (3,)
							# mode_d  = dens_s[idx]
							# print everything
							# print(f" SLAM pose: {pose}   density: {density:.6f}")
							# print(f" KDE mode: {mode}   mode density: {mode_d:.6f}")



							# ----- gradient‐step correction -----
							grad = self.prior_gradient_logl(local_kde, pose) #previous - single kde
							### global & local KDE - inactive #XXX activate for dual KDE
							# grad = self.mixture_prior_gradient(local_kde, global_kde, pose, alpha) #mixure of global and local kde. will still go to zero eventually but only when global max bandwidth is reached
							# print('grad:', grad)
							norm = np.linalg.norm(grad)
							# print('norm:', norm)
							if norm>1: grad/=norm
							# print('grad:', grad)
							corrected_rel[b,t,:3] = slam_rel[b,t,:3] + λ * torch.from_numpy(grad).to(slam_rel.device)
							# print('corrected pose:', corrected_rel[b,t,:3])
							# print('density at corrected:', kde.pdf(corrected_rel[b,t,:3].detach().cpu().numpy()))
							# exit()


						# --------- Visualise Algorithm vs KDE prior vs corrected algorithm poses ----------
						if VISUAL_KDE_ALGO_CORR and b == 7:
							# only XY dims 
							past_np  = past_traj[b].cpu().numpy()[:, :2]      # (T_past, 2)
							slam_np  = slam_rel[b].cpu().numpy()[:, :2]       # (T_future, 2)
							k_np_all = k_alternative_preds[b].cpu().numpy()[..., :2]  # (K, T_future, 2)

							# filter per‐timestep
							filtered_list = []
							for t in range(k_np_all.shape[1]):
								pts_t = k_np_all[:, t, :]                  # (K,2) at time t
								filtered_t = filter_outliers(pts_t, min_neighbors=1)      # maybe fewer than K
								filtered_list.append(filtered_t)                # list of arrays

							# stack into one big (N,2) array
							k_pts_filtered = np.vstack(filtered_list)          # (sum_Kt, 2)

							# fit 2D KDE on those filtered samples
							# kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0) #single local kde, previous
							local_kde_xy  = CustomKDE(k_pts_filtered, bandwidth=bandw, range_factor=1.0)
							# global_kde_xy = CustomKDE(k_pts_filtered, bandwidth=bandw_global, range_factor=1.0)

							corrected_np = corrected_rel[b,:, :2].cpu().numpy()    # (T_future, 2)

							save_path = os.path.join(self.cfg.log_dir, f"bev_alg_vs_kde_vs_corrected_{bandw:.4f}_band_{λ:.4f}_λ.png")
							self.visualize_bev_algorithm_vs_corrected(past_np, slam_np, corrected_np, k_pts_filtered, local_kde_xy, save_path)
							print(f"[INFO] Saved comparison plot to {save_path}")
							exit()
						# -------------------------------------------------------


					corrected_preds = corrected_rel.unsqueeze(1) # [B*N,1,T,D]

					# 7) Corrected SLAM metrics
					# SLAM ATE
					slam_ate = self.compute_ate(corrected_preds, fut_traj.unsqueeze(1), self.cfg.dimensions, self.traj_scale)['ate_trans']
					performance_corrected_slam['ATE_trans'] += slam_ate * Bflat

					# Compute SLAM ADE/FDE
					if self.cfg.dimensions in [2,3]:
						sl_p = corrected_preds
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)
					else:
						sl_p = corrected_preds[..., :3]
						sl_g = fut_traj.unsqueeze(1).repeat(1,1,1,1)[..., :3]
					dists_sl = torch.norm(sl_g - sl_p, dim=-1) * self.traj_scale
					for i, t in enumerate(timesteps):
						idx = min(t-1, dists_sl.size(2)-1)
						ade_slam = dists_sl[:,:, :idx+1].mean(dim=-1).min(dim=-1)[0].sum()
						fde_slam = dists_sl[:,:, idx   ].min(dim=-1)[0].sum()
						performance_corrected_slam['ADE'][i] += ade_slam.item()
						performance_corrected_slam['FDE'][i] += fde_slam.item()

				### --------------- End of correction ----------------

				samples += Bflat
				fut_traj = fut_traj.unsqueeze(1)

				# ATE for Leapfrog
				ate_res = self.compute_ate(k_alternative_preds, fut_traj, self.cfg.dimensions, self.traj_scale)
				performance['ATE_trans'] += ate_res['ate_trans'] * Bflat

				# FDE and ADE for translation only on Leapfrog
				if self.cfg.dimensions in [2,3]:
					pred_trans = k_alternative_preds  # already xy or xyz
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)
				else:
					pred_trans = k_alternative_preds[..., :3]
					gt_trans   = fut_traj.repeat(1, k_alternative_preds.size(1), 1, 1)[..., :3]
				distances = torch.norm(gt_trans - pred_trans, dim=-1) * self.traj_scale
				for i, time in enumerate(timesteps):
					max_index = min(time - 1, distances.shape[2] - 1)  # Ensure index does not exceed the array size = future timesteps

					ade = (distances[:, :, :max_index + 1]).mean(dim=-1).min(dim=-1)[0].sum() # ADE: average error over the time window, choose best candidate
					fde = (distances[:, :, max_index]).min(dim=-1)[0].sum() # FDE: error at the final time step in the window, choose best candidate

					performance['ADE'][i] += ade.item()
					performance['FDE'][i] += fde.item()



			
			### save KDE density-per-t histograms
			# self.GT_KDE_density_histograms(all_densities_by_time, hist_out_dir)

			### Finalise and print results
			# Leapfrog
			performance['ATE_trans'] /= samples # normalize ATE
			for i, t in enumerate(timesteps):
				
				# convert to Python floats so float-formatting won’t choke on a numpy scalar/array
				ade_avg = performance['ADE'][i] / samples
				fde_avg = performance['FDE'][i] / samples
				print_log(f'--Leapfrog ADE ({t}): {ade_avg:.4f}\tFDE: {fde_avg:.4f}', self.log)



			print_log(f'--Leapfrog ATE: {performance["ATE_trans"]:.4f}', self.log)


			# SLAM
			if ALGORITHM_SETTING:

				# Raw algorithm performance
				performance_slam['ATE_trans'] /= samples
				for i, t in enumerate(timesteps):
					ade_s = performance_slam['ADE'][i] / samples
					fde_s = performance_slam['FDE'][i] / samples
					print_log(f'--SLAM ADE ({t}): {ade_s:.4f}\tFDE: {fde_s:.4f}', self.log)
				print_log(f'--SLAM ATE: {performance_slam["ATE_trans"]:.4f}', self.log)
			
				# Algorithm performance corrected with KDE
				if ALGORITHM_SETTING == 'simulate_and_correct':

					performance_corrected_slam['ATE_trans'] /= samples
					for i, t in enumerate(timesteps):
						ade_s_corr = performance_corrected_slam['ADE'][i] / samples
						fde_s_corr = performance_corrected_slam['FDE'][i] / samples
						print_log(f'--SLAM ADE CORRECTED ({t}): {ade_s_corr:.4f}\tFDE: {fde_s_corr:.4f}', self.log)
					print_log(f'--SLAM ATE CORRECTED: {performance_corrected_slam["ATE_trans"]:.4f}', self.log)

# ----------- not used ---------------
"""
def save_vis_data(self):
	
	### Save the visualization data.
	
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

def p_sample_loop(self, x, mask, shape):
	self.model.eval()
	prediction_total = torch.Tensor().cuda()
	for _ in range(20):
		cur_y = torch.randn(shape).to(x.device)
		for i in reversed(range(self.n_steps)):
			cur_y = self.p_sample(x, mask, cur_y, i)
		prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
	return prediction_total


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
"""
