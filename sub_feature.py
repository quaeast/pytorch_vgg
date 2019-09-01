import torchvision
import torch
import numpy as np
import plant_classify
import matplotlib.pyplot as plt


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = extracted_layers
#
#     def forward(self, x):
#         outputs = []
#         for name, module in self.submodule._modules.items():
#             if name is "fc": x = x.view(x.size(0), -1)
#             x = module(x)
#             print(name)
#             if name in self.extracted_layers:
#                 outputs.append(x)
#         return outputs
#
#
# conv1 = FeatureExtractor()

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


