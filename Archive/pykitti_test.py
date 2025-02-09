import pykitti



basedir = '/home/scur2440/thesis/KITTI_odometry/dataset'

# Specify the dataset to load
sequence = '00'

dataset = pykitti.odometry(basedir, sequence)

print(dataset.poses[1])