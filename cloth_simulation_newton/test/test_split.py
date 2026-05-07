import numpy as np

#Load the numpy array
data = np.load('/data/zhoucy/sim/cloth_simulation_newton/examples/output/data/cloth_data_cloth_twist_90s_iter15.npy')

#Print the shape of the array
print(data.shape)

# Split the array into two halves
half = data.shape[0] // 2
data1 = data[800:1600]

# Print the shapes of the two halves
print(data1.shape)

# 保存两个数组
np.save('cloth_data_cloth_twist_90s_iter15_2.npy', data1)
