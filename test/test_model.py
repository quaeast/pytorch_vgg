import torchvision
import torch


net = torch.nn.Sequential(
    torch.nn.Linear(in_features=10, out_features=20)
)


print((torchvision.models.vgg16().append(torch.nn.ReLU(inplace=True))))

