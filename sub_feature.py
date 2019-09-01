import torchvision
import torch
import numpy as np
import plant_classify
import matplotlib.pyplot as plt


my_vgg = plant_classify.get_vgg16out15()
sub_vgg = my_vgg.features[0:7]

loader = plant_classify.get_data_loader(plant_classify.get_data())

for step, (batch_x, batch_y) in enumerate(loader):
    print(step, '>-----------------------------------')
    img = sub_vgg(batch_x)
    print(img.size())
    img = torch.squeeze(img, dim=0)
    img = img.detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    img = img.mean(axis=2)
    print(img.shape)
    plt.subplot(221+step)
    plt.imshow(img)
    if step == 1:
        break

plt.show()

# sub_vgg(dataset[0:5])


