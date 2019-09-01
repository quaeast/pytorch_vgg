import torch
import torchvision
import plant_classify
import numpy as np
import matplotlib.pyplot as plt


trained_my_vgg = plant_classify.get_vgg16out15(pre=True)
loader = plant_classify.get_data_loader(plant_classify.get_data())

for step, (test_x, test_y) in enumerate(loader):
    predict = trained_my_vgg(test_x)
    print('>------------------------------------')
    print('pre: ', predict.detach().numpy().argmax(1))
    print('tar: ', test_y)


