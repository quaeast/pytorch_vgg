import numpy as np


a = np.ones(shape=[2, 3, 4, 5])

a = a.transpose((1, 0, 2, 3))

print(a.shape)

