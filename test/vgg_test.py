import torch
import torchvision

loss_func = torch.nn.CrossEntropyLoss()

vgg = torchvision.models.vgg16()

x = torch.ones([1, 3, 255, 255])
y = torch.ones([1]).long()

out = vgg(x)

print(out.type())
print(y.type())

loss = loss_func(out, y)

print(loss.item())
