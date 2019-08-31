import torch
import numpy as np

x = np.array([1, 2, 3])

y = np.eye(10)[x]

print(y)
