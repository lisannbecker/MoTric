from scipy.optimize import minimize
import numpy as np
from scipy.stats import gaussian_kde
import torch

def find_max(kde, k_poses):
    # Negate the KDE to turn the maximization into minimization
    
    neg_kde = lambda x: -kde(x)

    # Use the mean of the samples as a starting point
    x0 = np.mean(k_poses.numpy(), axis=0)

    # Minimize the negative density
    res = minimize(neg_kde, x0)

    mode_location = res.x
    mode_density = kde(mode_location)

    print("Mode at:", mode_location)
    print("Max KDE density:", mode_density)
    #print("Some other density:", kde([0.00, 0.00, 0.00]))
    return mode_location

def kde_metrics(kde):
    # gaussian_kde expects shape (D, N), 
    # Kernel density estimation places a smooth "kernel" (Gaussian) at each sample point and sums them to create an overall density estimate
    # Parameter: bandwidth = how smooothly the points are summed. Eg affects whether two close modes merge into one or not
    
    print("Covariance matrix:\n", kde.covariance)
    print("Standard deviation (per dimension):", np.sqrt(np.diag(kde.covariance)))
    # print("Density at point:", kde(point_to_evaluate))

def kde_logl(kde, pose_estimate):
    log_likelihood = np.log(kde(pose_estimate))
    print(f'Log likelihood of {pose_estimate}: {log_likelihood}')


# ----- Approach 1: Gradient Ascent Correction -----
def gradient_logl(kde, pose, epsilon=1e-5):
    """
    Numerically approximate the gradient of the log likelihood at the given pose.
    pose: numpy array of shape (d,)
    Returns: numpy array of shape (d,) representing the gradient.
    """
    pose = np.array(pose, dtype=float)
    grad = np.zeros_like(pose)
    for i in range(len(pose)):
        delta = np.zeros_like(pose)
        delta[i] = epsilon
        # Evaluate log likelihood at pose + delta and pose - delta.
        logl_plus = np.log(kde(pose + delta))[0]
        logl_minus = np.log(kde(pose - delta))[0]
        grad[i] = (logl_plus - logl_minus) / (2 * epsilon)
    return grad

def correct_pose_gradient(kde, slam_pose, lambda_step=0.1):
    """
    Corrects the SLAM pose using a gradient ascent step on the log likelihood.
    
    x_corrected = x_slam + lambda * gradient(log p(x_slam))
    """
    # Ensure slam_pose is a numpy array.
    if isinstance(slam_pose, torch.Tensor):
        slam_pose = slam_pose.cpu().numpy()
    grad = gradient_logl(kde, slam_pose)
    corrected = slam_pose + lambda_step * grad
    return corrected



single_pose_all_ks = torch.tensor([[-3.0443e-02,  1.7418e-02, -1.1957e-02, -6.0314e-02,  2.1405e-02,
         -3.5358e-02, -1.3018e-02,  4.6410e-02,  5.6319e-02],
        [-2.7947e-02,  1.1016e-02, -1.0805e-03, -2.3613e-01,  1.3437e-01,
         -1.2380e-01, -3.0337e-02,  3.6546e-01,  3.8254e-01],
        [-6.6725e-02,  1.8243e-02, -8.5694e-03,  8.1638e-03,  1.9735e-01,
         -5.8389e-02,  4.2411e-02, -3.3412e-01,  2.8631e-01],
        [-1.7327e+00, -7.0685e-01,  2.2681e+00, -2.0996e+00,  1.8899e+00,
          1.8811e+00,  1.5009e+00, -1.6631e+00,  1.9568e+00],
        [-1.1624e+00, -9.4078e-01,  1.6057e+00,  1.8525e+00, -1.3371e+00,
         -1.7642e+00, -9.2766e-01, -1.5936e+00,  1.7733e+00],
        [-1.9213e-02,  4.5991e-02,  1.9228e-02,  4.4708e-03,  2.7786e-02,
         -7.7177e-02, -1.9540e-02,  6.8996e-02,  4.2833e-02],
        [-2.5933e-02,  2.2917e-02,  6.6621e-03, -2.4101e-01,  1.4625e-01,
         -1.1032e-01, -6.6693e-01,  4.1575e-04,  1.6581e-01],
        [ 7.0465e-01, -4.3170e-01, -7.3909e-01, -7.8394e-01, -7.0794e-01,
         -7.0888e-01, -6.9698e-01,  7.5668e-01, -9.8772e-01],
        [-1.1008e+00, -7.8502e-01,  1.1021e+00, -1.4639e+00, -1.2067e+00,
          1.1655e+00,  9.4804e-01,  7.4588e-01, -9.2594e-01],
        [ 9.9972e-01,  1.0863e+00, -9.0652e-01,  8.7903e-01, -9.5084e-01,
         -7.8523e-01,  9.1904e-01, -9.8057e-01, -6.9257e-01],
        [-4.3792e-02,  3.1937e-02,  1.4657e-03, -2.8102e-02,  1.0934e-02,
          2.3059e-03,  1.3970e-03, -6.6511e-03,  3.8950e-02],
        [-1.9483e+00, -2.7255e+00,  2.5546e+00,  9.2509e-01, -2.6891e+00,
         -1.6411e+00, -3.2647e+00, -2.3406e+00, -1.9235e+00],
        [-2.7599e-02,  1.9340e-02, -2.9875e-03,  2.3479e-01, -2.8384e-01,
          3.4059e-01,  3.6068e-01, -3.5402e-02,  3.6102e-01],
        [ 8.2145e-02,  3.2547e-01,  4.8256e-02, -1.3767e-01,  8.6543e-02,
         -1.1495e-01,  5.7329e-02, -2.7180e-02,  1.9028e-01],
        [-2.7857e+00,  2.0784e+00, -2.3722e+00,  2.1217e+00, -1.5058e+00,
          2.0137e+00,  2.8131e+00,  1.7004e+00,  1.4750e+00],
        [-3.8594e-02,  2.7124e-02,  1.6749e-02, -1.1263e-03,  2.5908e-02,
         -5.5816e-02, -9.1835e-04,  5.9111e-02,  5.0891e-02],
        [ 8.8170e-01,  1.7074e+00,  6.3905e-01, -1.7777e+00,  9.0523e-01,
          1.0219e+00,  7.5526e-01,  7.5588e-01, -1.6048e+00],
        [-3.3889e-02,  3.6593e-02,  1.2466e-02, -9.4380e-02, -3.5898e-02,
         -7.8364e-03, -2.1412e-02,  9.1716e-02,  2.5587e-02],
        [-4.3746e-02,  2.8465e-02, -9.9482e-03, -4.6415e-01, -5.0379e-02,
          2.0151e-01,  1.7946e-02,  9.8250e-02,  4.0914e-01],
        [ 2.0267e+00, -1.7759e+00, -1.2854e+00,  1.4164e+00,  1.0004e+00,
          1.7775e+00, -1.3513e+00,  1.7990e+00, -9.6485e-01],
        [-4.0976e-02,  1.2573e-03,  3.2508e-02,  5.3740e-02, -1.6522e-01,
          7.1825e-02,  1.1597e-01, -1.6252e-01, -3.9368e-02],
        [-1.0080e-01,  5.6722e-03,  5.2025e-02, -3.6958e-01,  1.4053e-01,
         -2.7433e-01, -9.6403e-02,  1.3060e-01,  3.4758e-01],
        [-1.7377e-02,  1.5110e-02, -7.0514e-03, -3.1556e-01,  2.3438e-01,
         -2.4486e-01,  8.5691e-02, -1.7716e-01,  6.9994e-01],
        [-3.1277e-02,  2.0817e-02, -1.2423e-03,  1.9068e-01, -2.8649e-01,
          2.3830e-01,  5.2416e-01, -1.3790e-02,  3.0325e-01],
        [ 1.5069e+00, -1.5600e+00, -1.5447e+00,  7.0636e-01,  1.9965e+00,
         -9.0832e-01, -1.3210e+00,  2.1970e+00,  1.6291e+00],
        [-2.1591e-02,  2.2366e-02,  1.0214e-02, -3.5093e-01,  3.1091e-01,
         -3.9558e-01, -1.3048e-01, -1.9371e-01,  3.0805e-01],
        [-1.9454e-02,  7.2674e-03, -2.6223e-02, -3.5426e-01,  1.0443e-01,
          3.6795e-02,  6.4835e-02, -3.5104e-01,  3.9138e-01],
        [-2.3513e-02,  2.5206e-02, -1.7479e-03, -2.9853e-01,  3.3740e-01,
         -4.1943e-01, -8.5463e-02, -4.3323e-02,  3.2800e-01],
        [-4.6400e-02,  1.9315e-02,  1.0833e-02, -1.3200e-02,  1.6645e-02,
         -2.5490e-02, -1.6192e-02,  5.1315e-02,  1.4058e-02],
        [-6.6332e-01, -3.7723e-01,  5.2196e-01,  6.7367e-01,  9.9552e-01,
         -1.6341e+00,  5.8745e-01,  8.7210e-01, -8.2130e-01]], device='cuda:0').cpu()


"""
print(single_pose_all_ks.size())
kde = gaussian_kde(single_pose_all_ks[:,:3].T) 
kde_metrics(kde)
mode=find_max(kde, single_pose_all_ks[:,:3])


point= torch.tensor([0.00, 0.00, 0.00])
kde_logl(kde, point)
kde_logl(kde, mode)
"""


# Dummy SLAM pose
slam_pose = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
kde = gaussian_kde(single_pose_all_ks.T) 
mode_9dof=find_max(kde, single_pose_all_ks)

#approach 1 - gradient update
corrected_pose_grad = correct_pose_gradient(kde, slam_pose, lambda_step=0.1)
print("Corrected SLAM pose (Gradient Ascent):", corrected_pose_grad)
