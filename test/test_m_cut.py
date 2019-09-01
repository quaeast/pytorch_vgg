import torchvision
import torch
import plant_classify


vgg = torchvision.models.vgg16()
sub_vgg = vgg.features[0:5]

data_set = plant_classify.get_vgg16out15()


print(sub_vgg)
