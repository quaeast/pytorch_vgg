import torch
import torchvision
import plant_classify
import numpy as np
import matplotlib.pyplot as plt
import config

END = 19

# vgg
trained_model = plant_classify.get_vgg16out15(pre=True).cuda()

# dens
# trained_model = plant_classify.get_densenet(pre=True).cuda()

loader = plant_classify.get_data_loader(plant_classify.get_data_cuda())

acc_list = np.empty(shape=[END+1])

# 20*5

for step, (test_x, test_y) in enumerate(loader):
    predict = trained_model(test_x)
    print(step, '>------------------------------------')
    pre = predict.detach().cpu().numpy().argmax(1)
    tar = test_y.cpu().numpy()
    acc = np.sum((pre-tar) == 0)
    acc_rate = acc/config.BATCH_SIZE
    print('acr: ', acc_rate)
    acc_list[step] = acc_rate
    if step == END:
        break

# print(acc_list)

print(acc_list.mean())

