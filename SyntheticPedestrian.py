import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_SE3_poses(
    timesteps: int,
    step_size_x: float,
    step_size_y: float,
    noise_trans_std: Tuple[float, float, float] = (0.05, 0.05, 0.01),
    noise_rot_std: Tuple[float, float, float] = (
        np.deg2rad(1),   # roll σ
        np.deg2rad(1),   # pitch σ
        np.deg2rad(2)    # yaw σ
    ), is_gt: bool = False
    ) -> np.ndarray:
    """
    Generate a stream of SE(3) poses [R|t] of shape (timesteps, 3, 4).

    - If is_gt=True: perfect, noiseless motion with (dx,dy)=(step_size_x,step_size_y) each step,
      and each R is a pure yaw aligning body-X to that motion vector.
    - If is_gt=False: add zero-mean Gaussian noise to both translation and rotation
      around a random unit axis (σ = noise_rot_std), but use the same base (step_size_x,step_size_y).

    Args:
      timesteps      : total T
      step_size_x    : nominal Δx per step
      step_size_y    : nominal Δy per step
      noise_trans_std: (σx,σy,σz) translation noise if is_gt=False
      noise_rot_std  : (σ_roll,σ_pitch,σ_yaw) rotation noise if is_gt=False
      is_gt          : True → perfect, False → add noise
    """
    poses = np.zeros((timesteps, 3, 4), dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    for i in range(timesteps):
        if i > 0:
            # --- translation update ---
            if is_gt:
                dx, dy, dz = step_size_x, step_size_y, 0.0
            else:
                sx, sy, sz = noise_trans_std
                dx = step_size_x + np.random.normal(0, sx)
                dy = step_size_y + np.random.normal(0, sy)
                dz = np.random.normal(0, sz)
            t += np.array([dx, dy, dz], dtype=np.float32)

            # --- base yaw along XY motion ---
            base_yaw = np.arctan2(dy, dx)

            # --- sample roll/pitch/yaw_noise ---
            if is_gt:
                roll = pitch = yaw_noise = 0.0
            else:
                σ_roll, σ_pitch, σ_yaw = noise_rot_std
                roll      = np.random.normal(0, σ_roll)
                pitch     = np.random.normal(0, σ_pitch)
                yaw_noise = np.random.normal(0, σ_yaw)

            yaw = base_yaw + yaw_noise

            # --- build rotation matrices ---
            c_y, s_y = np.cos(yaw),   np.sin(yaw)
            c_p, s_p = np.cos(pitch), np.sin(pitch)
            c_r, s_r = np.cos(roll),  np.sin(roll)

            R_z = np.array([[ c_y, -s_y, 0],
                            [ s_y,  c_y, 0],
                            [   0,    0, 1]], dtype=np.float32)

            R_y = np.array([[ c_p, 0, s_p],
                            [   0, 1,   0],
                            [-s_p, 0, c_p]], dtype=np.float32)

            R_x = np.array([[1,    0,     0],
                            [0,  c_r, -s_r],
                            [0,  s_r,  c_r]], dtype=np.float32)

            # final R = R_z @ R_y @ R_x
            R = R_z @ R_y @ R_x

        else:
            R = np.eye(3, dtype=np.float32)

        poses[i] = np.hstack((R, t.reshape(3,1)))

    return poses


def write_poses_to_file(SE3_poses, filename):
    """Write each 3×4 pose in row-major order ([R|t] flattened) so loader can reshape it directly."""
    with open(filename, 'w') as f:
        for pose in SE3_poses:
            # pose is 3×4 already in [R | t] form
            flat = pose.flatten(order='C')  # [r11,r12,r13,tx,  r21,r22,r23,ty,  r31,r32,r33,tz]
            line = " ".join(f"{v:.6f}" for v in flat)
            f.write(line + "\n")

def load_raw_traj_poses(file_path):
    """assumes format [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split())) #list for line
            SE3 = np.array(values).reshape(3,4) #SE3 matrix without bottom row 0,0,0,1
            poses.append(SE3)
    return np.stack(poses)



def plot_top_down(SE3_poses, filename, arrow_step=1, arrow_scale=0.1):
    """
    Plot the forward (X) vs lateral (Y) trajectory, drawing the BODY‐X arrow.
    """
    xs = SE3_poses[:, 0, 3]
    ys = SE3_poses[:, 1, 3]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, '-x', lw=1, markersize=4, label="Trajectory")

    for i in range(0, len(xs), arrow_step):
        # BODY‐X in WORLD = first column of R
        fx = SE3_poses[i, 0, 0]
        fy = SE3_poses[i, 1, 0]
        ax.arrow(
            xs[i], ys[i],
            fx * arrow_scale, fy * arrow_scale,
            head_width=0.02, head_length=0.03,
            fc='C1', ec='C1'
        )

    ax.set_xlabel('Forward X (m)')
    ax.set_ylabel('Lateral Y (m)')
    ax.set_title('Top-down Trajectory with Body-X Heading')
    ax.axis('equal')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":

    # ================ Parameters and prepare filenames ================
    TIMESTEPS = 500000
    STEP_SIZE_X = 0.1
    STEP_SIZE_Y = 0.2

    # for file names
    data_type = 'RandomNoise'
    if str(TIMESTEPS)[-3:] == '000':
        filename_steps = str(TIMESTEPS)[:-3]+'k'
    else:
        filename_steps = str(TIMESTEPS)
    # ================ Parameters and prepare filenames ================


    # ================ Generate synthetic poses ================
    gt_poses = generate_SE3_poses(
        timesteps=TIMESTEPS,
        step_size_x=STEP_SIZE_X,
        step_size_y=STEP_SIZE_Y,
        is_gt=True
    ) 
    write_poses_to_file(gt_poses, f'Synthetic/{data_type}/GT_{filename_steps}_synthetic_poses.txt')
    # plot_top_down(gt_poses[80:100], 'Synthetic/GT_top_down_trajectory.png')


    noisy_poses = generate_SE3_poses(
        timesteps=TIMESTEPS,
        step_size_x=STEP_SIZE_X,
        step_size_y=STEP_SIZE_Y,
        is_gt=False
    )  
    write_poses_to_file(noisy_poses, f'Synthetic/{data_type}/NOISY_{filename_steps}_synthetic_poses.txt')
    # plot_top_down(noisy_poses[80:100], 'Synthetic/NOISY_top_down_trajectory.png')
    # ================ Generate synthetic poses ================







    # ================ Code to check the generated files ================
    exit()
    noisy_poses = load_raw_traj_poses('/home/scur2440/MoTric/Synthetic/NOISY_synthetic_poses_200000.txt')
    plot_top_down(noisy_poses[80:100], 'Synthetic/top_down_trajectory_NOISY_CHECK.png')

    gt_poses = load_raw_traj_poses('/home/scur2440/MoTric/Synthetic/GT_synthetic_poses_200000.txt')
    plot_top_down(gt_poses[80:100], 'Synthetic/top_down_trajectory_GT_CHECK.png')
    # ================ Code to check the generated files ================





### Query dump

def generate_SE3_poses(timesteps, step_size, right_drift=False):
    SE3_poses = np.zeros((timesteps, 3, 4))  # x, y, z positions

    R = np.eye(3) # initial rotation (identity matrix)
    translation = np.zeros(3)  # initial translation (x, y, z)

    if right_drift:
        # fixed small clockwise yaw (negative = CW)
        angle = np.deg2rad(2)        # 2° per step
        c, s = np.cos(angle), np.sin(angle)
        # R_z(-angle) = [[ cos,  sin, 0],
        #                [-sin,  cos, 0],
        #                [   0,    0,  1]]
        R_yaw_cw = np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1],
        ])


    for t in range(timesteps):
        if t > 0:

            #translation incl step size
            dx = np.random.normal(0, 0.01)  # lateral movement TODO step size should affect this too
            dy = step_size + np.random.normal(0, 0.01)  # forward movement
            dz = np.random.normal(0, 0.005)  # vertical movement
            translation = translation + np.array([dx, dy, dz])  

            if right_drift:
                R = R @ R_yaw_cw  # apply the fixed 2 degree clockwise yaw
            # now add small noise to rotation
            angle = np.random.normal(0, np.deg2rad(2))  # small rotation angle
            axis = np.random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)  # normalise axis TODO explain this
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R = R @ R_delta 

        
        SE3 = np.hstack((R, translation.reshape(3, 1)))
        SE3_poses[t] = SE3

    return SE3_poses

def generate_SE3_poses_GT(timesteps, step_size_x, step_size_y):
    """
    Generate SE(3) ground‐truth poses with no noise. At each step:
      - translation += (step_size_x, step_size_y, 0)
      - rotation is a pure yaw so that the body‐X axis points along (Δx, Δy)
    Args:
      timesteps   : int
      step_size_x : float (forward step)
      step_size_y : float (lateral step)
    Returns:
      SE3_poses   : (timesteps, 3, 4) array of [R|t]
    """
    SE3_poses = np.zeros((timesteps, 3, 4))
    translation = np.zeros(3)

    for t in range(timesteps):
        if t > 0:
            # 1) translate
            dx, dy, dz = step_size_x, step_size_y, 0.0
            translation += np.array([dx, dy, dz])

            # 2) yaw so body‐X → (dx, dy)
            yaw = np.arctan2(dy, dx)
            c, s = np.cos(yaw), np.sin(yaw)

            # 3) build R_z(yaw)
            R = np.array([
                [ c, -s, 0],
                [ s,  c, 0],
                [ 0,  0, 1],
            ])
        else:
            R = np.eye(3)

        SE3_poses[t] = np.hstack((R, translation.reshape(3, 1)))

    return SE3_poses
