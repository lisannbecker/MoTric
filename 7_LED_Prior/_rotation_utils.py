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


def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix.
    quat: tensor of shape (..., 4) in format [qw, qx, qy, qz] or [qx, qy, qz, qw]
    Returns: rotation matrix of shape (..., 3, 3)
    """
    # Assuming quat is [qx, qy, qz, qw] based on your existing quaternion functions
    if quat.shape[-1] == 4:
        qx, qy, qz, qw = quat.unbind(-1)
    else:
        raise ValueError("Quaternion must have 4 components")
    
    # Compute rotation matrix elements
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    
    # Build rotation matrix
    R = torch.stack([
        torch.stack([1 - 2*yy - 2*zz, 2*xy - 2*wz, 2*xz + 2*wy], dim=-1),
        torch.stack([2*xy + 2*wz, 1 - 2*xx - 2*zz, 2*yz - 2*wx], dim=-1),
        torch.stack([2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*xx - 2*yy], dim=-1)
    ], dim=-2)
    
    return R

def rotation_matrix_to_6d(R):
    """
    Convert rotation matrix to 6D representation.
    R: tensor of shape (..., 3, 3)
    Returns: tensor of shape (..., 6)
    """
    # Take first two columns of rotation matrix
    return R[..., :3, :2].reshape(*R.shape[:-2], 6)

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula.
    axis_angle: tensor of shape (..., 3)
    Returns: rotation matrix of shape (..., 3, 3)
    """
    # This is essentially the same as your so3_to_SO3 function
    # but using the existing so3_exponential function from your paste
    return so3_exponential(axis_angle)

def convert_6d_back_to_original(pose_6d, rotation_type):
    """
    Convert 6D+translation pose back to original rotation representation.
    pose_6d: tensor of shape (..., 9) - [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    rotation_type: str indicating target rotation type
    Returns: pose in original representation
    """
    translation = pose_6d[..., :3]
    rot_6d = pose_6d[..., 3:9]
    
    if rotation_type in [2, 3, 'none']:
        return translation
    elif rotation_type in [7, 'quaternion']:
        # Convert 6D back to rotation matrix, then to quaternion
        rot_matrix = rot6d_to_rotmat(rot_6d)
        quat = rotation_matrix_to_quaternion(rot_matrix)
        return torch.cat([translation, quat], dim=-1)
    elif rotation_type in [9, '6d']:
        return pose_6d  # already in 6D format
    elif rotation_type in [6, 'lie', 'axis_angle']:
        # Convert 6D back to rotation matrix, then to axis-angle
        rot_matrix = rot6d_to_rotmat(rot_6d)
        axis_angle = so3_logarithm(rot_matrix)
        return torch.cat([translation, axis_angle], dim=-1)
    else:
        raise ValueError(f"Unknown rotation type: {rotation_type}")

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion [qx, qy, qz, qw].
    R: tensor of shape (..., 3, 3)
    Returns: quaternion of shape (..., 4)
    """
    # Shepperd's method for numerical stability
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # Case 1: trace > 0
    s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
    qw = 0.25 * s
    qx = (R[..., 2, 1] - R[..., 1, 2]) / s
    qy = (R[..., 0, 2] - R[..., 2, 0]) / s
    qz = (R[..., 1, 0] - R[..., 0, 1]) / s
    
    # For numerical stability, we should handle other cases too
    # but this is the most common case and works for most situations
    
    return torch.stack([qx, qy, qz, qw], dim=-1)

def pose_to_6d_translation(pose, rotation_type=None):
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