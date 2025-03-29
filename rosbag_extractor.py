import bagpy
from bagpy import bagreader
import pandas as pd


bag = bagreader("./hilti/exp04_construction_upper_level.bag")

# List available topics
print(bag.topic_table)

exit()
# Extract pose data (if using TUM format, check for `/tf` or `/odom`)
pose_data = bag.message_by_topic("/tf")  # Adjust topic as needed

# Load into pandas for easy processing
df = pd.read_csv(pose_data)
print(df.head())
