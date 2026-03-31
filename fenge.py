import numpy as np
import joblib

# 读取数据并分割
data = np.load('./data/NYC.npy')
poi = np.load('./data/NYC_POI.npy')

# 数据从中间分割
mid_point = len(data) // 2
data_part1 = data[:mid_point]
data_part2 = data[mid_point:]

# 保存分割后的数据
np.save('./data/NYC_part1.npy', data_part1)
np.save('./data/NYC_part2.npy', data_part2)
