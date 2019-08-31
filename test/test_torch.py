import torch
import numpy as np

x_np = np.ones(shape=[2])

x = torch.from_numpy(x_np)

x = x.long()

print(x.dtype)
