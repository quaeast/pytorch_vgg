import numpy as np
import matplotlib.pyplot as plt


loss = np.load('./result/loss.npy')
x = np.arange(loss.size)
loss = loss.clip(min=0, max=10)
plt.title('loss plot(clip: 10)')
plt.xlabel('batch step')
plt.ylabel('loss')
plt.plot(loss)
plt.show()
