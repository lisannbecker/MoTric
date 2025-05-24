import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import os


def generate_SE3_poses_GT(timesteps, step_size_x, step_size_y, motion_type='straight', curve_radius=5.0):
    """
    Generate SE(3) ground-truth poses with no noise. At each step:
    - For 'straight' motion: translation += (step_size_x, step_size_y, 0)
    - For 'right_curve' motion: follows a right-turning curve with specified radius
    - Rotation is a pure yaw so that the body-X axis points along direction of movement
    
    Args:
        timesteps : int
        step_size_x : float (forward step)
        step_size_y : float (lateral step)
        motion_type : str, either 'straight' or 'right_curve'
        curve_radius : float, radius of the curve in meters (only used when motion_type='right_curve')
    
    Returns:
        SE3_poses : (timesteps, 3, 4) array of [R|t]
    """
    import numpy as np
    
    SE3_poses = np.zeros((timesteps, 3, 4))
    translation = np.zeros(3)
    
    # For curved motion, calculate angular velocity based on step size and radius
    if motion_type == 'right_curve':
        # Calculate step length (approximate arc length per step)
        step_length = np.sqrt(step_size_x**2 + step_size_y**2)
        # Calculate angular displacement per step (radians)
        delta_angle = step_length / curve_radius
        # Start with no yaw
        current_yaw = 0
    
    for t in range(timesteps):
        if t > 0:
            if motion_type == 'straight':
                # Standard straight-line motion
                dx, dy, dz = step_size_x, step_size_y, 0.0
                translation += np.array([dx, dy, dz])
                # Yaw so body-X → (dx, dy)
                yaw = np.arctan2(dy, dx)
            
            elif motion_type == 'right_curve':
                # Update yaw for right-turning curve
                current_yaw -= delta_angle  # Negative for right turn
                
                # Calculate displacement in global frame based on current heading
                dx = step_length * np.cos(current_yaw)
                dy = step_length * np.sin(current_yaw)
                dz = 0.0
                
                translation += np.array([dx, dy, dz])
                # Yaw matches the current heading
                yaw = current_yaw
            
            # Build rotation matrix
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ])
        else:
            R = np.eye(3)
            
        SE3_poses[t] = np.hstack((R, translation.reshape(3, 1)))
    
    return SE3_poses

def generate_noisy_past_clean_future_dataset(timesteps, past_window_size, future_window_size,
                                            trans_noise_level=0.02, rot_noise_level=1.0,
                                            step_size_x=0.1, step_size_y=0.02,
                                            noise_type='random_independent',
                                            motion_type='straight',
                                            curve_radius=5.0,
                                            right_bias_factor=1.5):
    """
    Generate a dataset with noisy past poses and clean GT future poses.
    
    Args:
        timesteps: Total number of timesteps to generate
        past_window_size: Number of poses in the past window
        future_window_size: Number of poses in the future window
        trans_noise_level: Standard deviation of translation noise (meters)
        rot_noise_level: Standard deviation of rotation noise (degrees)
        step_size_x: Forward step size for GT trajectory
        step_size_y: Lateral step size for GT trajectory
        noise_type: Type of noise to add ('random_independent', 'random_walk', or 'right_bias')
        motion_type: Type of GT motion ('straight' or 'right_curve')
        curve_radius: Radius of the curve in meters (only used when motion_type='right_curve')
        right_bias_factor: Factor to multiply noise in the right direction (only used when noise_type='right_bias')
    
    Returns:
        SE3_poses_gt: Clean ground truth poses (timesteps, 3, 4)
        SE3_poses_noisy: Poses with noise added to past window (timesteps, 3, 4)
    """
    import numpy as np
    
    # Generate GT trajectory - either straight or right curve
    SE3_poses_gt = generate_SE3_poses_GT(
        timesteps, 
        step_size_x, 
        step_size_y, 
        motion_type=motion_type,
        curve_radius=curve_radius
    )
    SE3_poses_noisy = SE3_poses_gt.copy()
    
    # to keep track of accumulated noise in rando walk
    if noise_type == 'random_walk':
        accumulated_trans_noise = np.zeros(3)
        accumulated_rot_noise_axis = np.random.normal(0, 1, 3)
        accumulated_rot_noise_axis = accumulated_rot_noise_axis / np.linalg.norm(accumulated_rot_noise_axis)
        accumulated_rot_noise_angle = 0

    # Initialize accumulated bias and noise if this is the first timestep in past window
    if noise_type == 'right_bias':
        accumulated_right_bias = np.zeros(3)
        accumulated_rot_noise   = np.eye(3)
    
    # Add noise only to past poses (0 to past_window_size)
    for t in range(1, past_window_size):
        # Get the GT pose
        R_gt = SE3_poses_gt[t, :3, :3].copy()
        t_gt = SE3_poses_gt[t, :3, 3].copy()
        
        if noise_type == 'random_independent':
            ### Noise type: Standard random and independent noise (original) - so no noise accumulation
            t_noise = np.random.normal(0, trans_noise_level, size=3)
            angle = np.random.normal(0, np.deg2rad(rot_noise_level))
            axis = np.random.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)  # Normalize to unit vector
            
        elif noise_type == 'random_walk': #TODO won't rotation develop independently from translation now - also not realistic, rotation should somewhat follow translation
            ### Noise Type: Random walk - accumulate the noise over steps
            # Translation random walk
            step_noise = np.random.normal(0, trans_noise_level/np.sqrt(2), size=3)  # Reduced by sqrt(2) to maintain same overall variance
            accumulated_trans_noise += step_noise
            t_noise = accumulated_trans_noise
            
            # Rotation random walk
            angle_step = np.random.normal(0, np.deg2rad(rot_noise_level)/np.sqrt(2))
            accumulated_rot_noise_angle += angle_step
            angle = accumulated_rot_noise_angle
            axis = accumulated_rot_noise_axis
            
        elif noise_type == 'right_bias':            
            # 1. Determine the direction of movement from the GT pose
            # Extract forward vector from rotation matrix (first column of R_gt)
            forward_direction = R_gt[:, 0]
            # Extract right vector from rotation matrix (second column of R_gt)
            right_direction = -R_gt[:, 1]            
            
            # 2. Add incremental bias in the right direction for this timestep
            # This creates a gradual drift to the right
            incremental_right_bias = right_direction * trans_noise_level * (right_bias_factor - 1.0) / past_window_size
            
            # 3. Accumulate the bias
            accumulated_right_bias += incremental_right_bias
            
            # 4. Add random noise on top of the accumulated bias
            random_noise = np.random.normal(0, trans_noise_level / 2, size=3)  # Reduced random component
            
            # 5. Combine accumulated bias and random noise for translation
            t_noise = accumulated_right_bias + random_noise
            
            # 6. Generate random rotation noise (independent)
            angle = np.random.normal(0, np.deg2rad(rot_noise_level))
            axis = np.random.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)  # Normalize to unit vector

            # Asymmetric noise with right bias relative to direction of movement - THIS GENERATES AN OFFSET
            
            # 1. First determine the direction of movement from the GT pose
            # Extract forward vector from rotation matrix (first column of R_gt)
            # forward_direction = R_gt[:, 0]
            # # Extract right vector from rotation matrix (second column of R_gt)
            # right_direction = R_gt[:, 1]
            
            # # 2. Add a bias component in the right direction
            # # This is a systematic bias (not random)
            # right_bias_component = right_direction * trans_noise_level * (right_bias_factor - 1.0)
            
            # # 3. Add normal random noise on top of the bias
            # random_noise = np.random.normal(0, trans_noise_level, size=3)
            
            # # 4. Combine bias and random noise for translation
            # t_noise = right_bias_component + random_noise
            
            # # 5. Generate random rotation noise (independent)
            # angle = np.random.normal(0, np.deg2rad(rot_noise_level))
            # axis = np.random.normal(0, 1, 3)
            # axis = axis / np.linalg.norm(axis)  # Normalize to unit vector

        
        # Apply translation noise
        t_noisy = t_gt + t_noise
        
        # Create rotation matrix for the noise using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Apply noise to rotation
        R_noisy = R_gt @ R_noise
        
        # Update the noisy pose
        SE3_poses_noisy[t, :3, :3] = R_noisy
        SE3_poses_noisy[t, :3, 3] = t_noisy
    
    return SE3_poses_gt, SE3_poses_noisy

def write_synthetic_dataset(output_dir, num_trajectories, timesteps, 
                           past_window_size, future_window_size,
                           trans_noise_level=0.02, rot_noise_level=1.0,
                           step_size_x=0.1, step_size_y=0.02,
                           noise_type='random_independent',
                           motion_type='straight',
                           curve_radius=5.0,
                           right_bias_factor=1.5):
    """
    Write synthetic dataset to disk with the specified parameters.
    
    Args:
        output_dir: Directory to write files to
        num_trajectories: Number of trajectories to generate
        timesteps: Total timesteps per trajectory
        past_window_size: Number of poses in past window
        future_window_size: Number of poses in future window
        trans_noise_level: Translation noise level (meters)
        rot_noise_level: Rotation noise level (degrees)
        step_size_x: Forward step size for GT trajectory
        step_size_y: Lateral step size for GT trajectory
        noise_type: Type of noise to add ('random_independent', 'random_walk', or 'right_bias')
        motion_type: Type of GT motion ('straight' or 'right_curve')
        curve_radius: Radius of the curve in meters (only used when motion_type='right_curve')
        right_bias_factor: Factor to multiply noise in the right direction (only used when noise_type='right_bias')
    """
    out_dir = os.path.join(output_dir, f"{motion_type}_{noise_type}")
    os.makedirs(out_dir, exist_ok=True)

    # open the two output text files once
    gt_file_path    = os.path.join(out_dir, "synthetic_gt_poses.txt")
    noisy_file_path = os.path.join(out_dir, "synthetic_noisy_past_poses.txt")
    with open(gt_file_path, 'w') as gt_file, open(noisy_file_path, 'w') as noisy_file:

        for i in range(num_trajectories):
            # generate one trajectory
            SE3_poses_gt, SE3_poses_noisy = generate_noisy_past_clean_future_dataset(
                timesteps=timesteps,
                past_window_size=past_window_size,
                future_window_size=future_window_size,
                trans_noise_level=trans_noise_level,
                rot_noise_level=rot_noise_level,
                step_size_x=step_size_x,
                step_size_y=step_size_y,
                noise_type=noise_type,
                motion_type=motion_type,
                curve_radius=curve_radius,
                right_bias_factor=right_bias_factor
            )

            # write GT poses
            for pose in SE3_poses_gt:
                flat = pose.flatten(order='C')
                gt_file.write(" ".join(f"{v:.6f}" for v in flat) + "\n")

            # write noisy past poses
            for pose in SE3_poses_noisy:
                flat = pose.flatten(order='C')
                noisy_file.write(" ".join(f"{v:.6f}" for v in flat) + "\n")

            # blank line between trajectories
            if i < num_trajectories - 1:
                gt_file.write("\n")
                noisy_file.write("\n")

    # save a single metadata file
    metadata_path = os.path.join(out_dir, "metadata.npz")
    np.savez(
        metadata_path,
        num_trajectories=num_trajectories,
        timesteps=timesteps,
        past_window_size=past_window_size,
        future_window_size=future_window_size,
        trans_noise_level=trans_noise_level,
        rot_noise_level=rot_noise_level,
        step_size_x=step_size_x,
        step_size_y=step_size_y,
        noise_type=noise_type,
        motion_type=motion_type,
        curve_radius=curve_radius,
        right_bias_factor=right_bias_factor,
    )

    print(f"Generated {num_trajectories} trajectories in {out_dir}")
    print(f"  GT → {gt_file_path}")
    print(f"  Noisy → {noisy_file_path}")
    print(f"  Metadata → {metadata_path}")

def plot_noisy_vs_gt_trajectory(SE3_poses_gt, SE3_poses_noisy, past_window_size, output_file):
    """
    Create a visualization showing both the GT and noisy trajectories.
    Highlights the transition point between past and future.
    """
    xs_gt = SE3_poses_gt[:, 0, 3]
    ys_gt = SE3_poses_gt[:, 1, 3]
    
    xs_noisy = SE3_poses_noisy[:, 0, 3]
    ys_noisy = SE3_poses_noisy[:, 1, 3]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ground truth trajectory
    ax.plot(xs_gt, ys_gt, '-o', color='green', lw=2, markersize=4, label="GT Trajectory")
    
    # Plot noisy past trajectory
    ax.plot(xs_noisy[:past_window_size], ys_noisy[:past_window_size], 
            '-x', color='red', lw=2, markersize=4, label="Noisy Past")
    
    # Plot clean future trajectory (should match GT)
    ax.plot(xs_noisy[past_window_size-1:], ys_noisy[past_window_size-1:], 
            '-o', color='blue', lw=2, markersize=4, label="Clean Future")
    
    # Mark the transition point
    ax.plot(xs_noisy[past_window_size-1], ys_noisy[past_window_size-1], 
            'o', color='purple', markersize=8, label="Last Measured Point")
    
    # Draw orientation arrows for selected poses
    arrow_step = 2
    arrow_scale = 0.1
    
    for i in range(0, len(xs_gt), arrow_step):
        # Draw GT orientation
        fx_gt = SE3_poses_gt[i, 0, 0]
        fy_gt = SE3_poses_gt[i, 1, 0]
        ax.arrow(
            xs_gt[i], ys_gt[i],
            fx_gt * arrow_scale, fy_gt * arrow_scale,
            head_width=0.02, head_length=0.03,
            fc='darkgreen', ec='darkgreen', alpha=0.7
        )
        
        # Draw noisy orientation (past only)
        if i < past_window_size:
            fx_noisy = SE3_poses_noisy[i, 0, 0]
            fy_noisy = SE3_poses_noisy[i, 1, 0]
            ax.arrow(
                xs_noisy[i], ys_noisy[i],
                fx_noisy * arrow_scale, fy_noisy * arrow_scale,
                head_width=0.02, head_length=0.03,
                fc='darkred', ec='darkred', alpha=0.7
            )
    
    # Add vertical separator line at transition point
    transition_x = xs_noisy[past_window_size-1]
    transition_y = ys_noisy[past_window_size-1]
    ax.axvline(x=transition_x, color='purple', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Noisy Past vs Clean Future Trajectory')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    output_dir = "synthetic_data/"
    
    ### GENERATE TRAJECTORY DATA
    # Generate dataset with right curve motion
    write_synthetic_dataset(output_dir, 
                            num_trajectories=10000, #10k
                            timesteps=32, #timesteps per trajectory
                            past_window_size=10, #out  of which past (noise to be added)
                            future_window_size=22, #out of which future (GT)
                            trans_noise_level=0.1, 
                            rot_noise_level=2.0,
                            step_size_x=0.2, 
                            step_size_y=0.1,
                            noise_type='right_bias', # Type of noise - either 'random_independent', 'random_walk', or 'right_bias'
                            motion_type='right_curve',  # GT motion - either 'straight' or 'right_curve'
                            curve_radius=6.0, # radius of curve in meters (if selected right curve)
                            right_bias_factor=4.5 # magnitude of right bias (if selected right bias)
                        )

    
    
    ### VISUALISATION ONLY
    #  Create a visualization of one trajectory
    # timesteps = 30
    # past_window_size = 10
    # future_window_size = 20

    # noise = 'random_walk' # random_independent random_walk right_bias
    # motion = 'straight' # straight right_curve
    
    # SE3_poses_gt, SE3_poses_noisy = generate_noisy_past_clean_future_dataset(timesteps, 
    #                                         past_window_size, 
    #                                         future_window_size,
    #                                         trans_noise_level=0.1, 
    #                                         rot_noise_level=2.0,
    #                                         step_size_x=0.2, 
    #                                         step_size_y=0.1,
    #                                         noise_type=noise, 
    #                                         motion_type=motion, 
    #                                         curve_radius=6.0,
    #                                         right_bias_factor=4.5)
    
    # plot_noisy_vs_gt_trajectory(
    #     SE3_poses_gt, 
    #     SE3_poses_noisy, 
    #     past_window_size,
    #     os.path.join(output_dir, f"{motion}_{noise}.png")
    # )