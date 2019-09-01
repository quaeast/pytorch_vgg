import torch

pthfile = r'../param/vgg16-397923af.pth'

a = torch.load(pthfile)

print(a)


