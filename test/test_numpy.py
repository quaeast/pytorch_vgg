import numpy as np

a = np.ones(shape=[25, 25, 25])
a = a.mean(axis=2, keepdims=True)



print(a.shape)
