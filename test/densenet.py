import torchvision
import torch

dens_net = torchvision.models.DenseNet()


print(dens_net.classifier)
