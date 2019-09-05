import numpy as np

x = np.ones([100, 3, 100, 100])

x1 = x[:10]

x2 = x[10:]

print(x2.shape[0])



