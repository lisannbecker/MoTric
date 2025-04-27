import torch
import torch.nn.functional as F

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


########################################
# SO(3) Helper Functions for Lie Algebra
########################################
def so3_exponential(r: torch.Tensor):
    """
    Converts a rotation vector r (in R^3) to a rotation matrix using the Rodrigues formula.
    r: tensor of shape (..., 3)
    Returns: rotation matrices of shape (..., 3, 3)
    """
    theta = r.norm(dim=-1, keepdim=True)  # (..., 1)
    theta = theta.clamp(min=1e-6)
    r_hat = r / theta                     # normalized axis, shape (..., 3)
    
    # Build the skew-symmetric matrix of r_hat.
    zeros = torch.zeros_like(r_hat[..., 0])
    r_hat_x = torch.stack([zeros, -r_hat[..., 2], r_hat[..., 1]], dim=-1)
    r_hat_y = torch.stack([r_hat[..., 2], zeros, -r_hat[..., 0]], dim=-1)
    r_hat_z = torch.stack([-r_hat[..., 1], r_hat[..., 0], zeros], dim=-1)
    r_hat_skew = torch.stack([r_hat_x, r_hat_y, r_hat_z], dim=-2)  # (..., 3, 3)
    
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand(r_hat_skew.shape)
    sin_term = torch.sin(theta)[..., None]
    cos_term = torch.cos(theta)[..., None]
    R = I + sin_term * r_hat_skew + (1 - cos_term) * torch.matmul(r_hat_skew, r_hat_skew)
    return R

def so3_logarithm(R: torch.Tensor):
    """
    Computes the logarithm map from a rotation matrix R in SO(3)
    to a rotation vector in R^3.
    
    R: tensor of shape (..., 3, 3)
    Returns: tensor of shape (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...,)
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)[..., None]  # shape: (..., 1)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    # Ensure sin_theta is broadcastable: make sure it has two trailing singleton dims
    while sin_theta.dim() < R.dim():
        sin_theta = sin_theta.unsqueeze(-1)
    skew = (R - R.transpose(-2, -1)) / (2 * sin_theta)
    # Compute the rotation vector from the skew-symmetric matrix
    r = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)  # shape: (..., 3)
    # Do NOT squeeze theta: keep shape (..., 1)
    return theta * r  # theta will broadcast properly with r, yielding shape (..., 3)


def relative_lie_rotation(r: torch.Tensor, r_ref: torch.Tensor):
    """
    Given two Lie algebra representations r and r_ref (both of shape (..., 3)),
    compute the relative rotation in Lie algebra form via:
         r_rel = log(exp(-[r_ref]_x) exp([r]_x))
    Args:
       r: tensor of shape (..., 3) for the current rotation.
       r_ref: tensor of shape (..., 3) for the reference rotation.
    Returns: relative rotation vector of shape (..., 3)
    """
    # Compute rotation matrices:
    R = so3_exponential(r)             # (..., 3, 3)
    R_ref = so3_exponential(r_ref)       # (..., 3, 3)
    R_rel = torch.matmul(R_ref.transpose(-2, -1), R)  # relative rotation matrix
    r_rel = so3_logarithm(R_rel)         # (..., 3)
    return r_rel


#############################################
# Helper functions for 6D rotation representation
#############################################
def rot6d_to_rotmat(x: torch.Tensor):
    """
    Converts a 6D rotation representation to a 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".
    x: tensor of shape (..., 6)
    Returns: tensor of shape (..., 3, 3)
    """
    # Split x into two 3D vectors
    x1 = x[..., :3]
    x2 = x[..., 3:]
    # Normalize the first vector
    b1 = F.normalize(x1, dim=-1)
    # Project x2 to be orthogonal to b1
    dot = torch.sum(b1 * x2, dim=-1, keepdim=True)
    x2_ortho = x2 - dot * b1
    b2 = F.normalize(x2_ortho, dim=-1)
    # b3 is the cross product.
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack b1, b2, b3
    rot_matrix = torch.stack((b1, b2, b3), dim=-1)  # shape (..., 3, 3)
    return rot_matrix

def rotmat_to_rot6d(R: torch.Tensor):
    """
    Converts a rotation matrix (or batched rotation matrices) to a 6D rotation representation.
    We'll use the first two columns of the rotation matrix.
    R: tensor of shape (..., 3, 3)
    Returns: tensor of shape (..., 6)
    """
    return R[..., :3, :2].reshape(*R.shape[:-2], 6)