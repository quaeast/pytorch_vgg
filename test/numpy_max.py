import numpy as np

x = np.array([[1, 2, 4], [1, 2, 3], [1, 2, 3]])


x_max = x.argmax(1)

print(x_max)
