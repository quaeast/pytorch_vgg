import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import numpy as np
import random
import config


def get_vgg16out15(pre=False):
    my_vgg = torchvision.models.vgg16()
    my_vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=16, bias=True)
    if pre:
        my_vgg.load_state_dict(torch.load(r'param/my_vgg_out_15_step_50.pth'))
    return my_vgg


def get_densenet(pre=False):
    my_densenet = torchvision.models.DenseNet()
    my_densenet.classifier = torch.nn.Linear(in_features=1024, out_features=16, bias=True)
    if pre:
        my_densenet.load_state_dict(torch.load(r'./param/my_dens_out_15_step_50.pth'))
    return my_densenet


def get_data_cuda():
    np_img = np.load('./data_set/torch/img_np_list.npy')
    np_target = np.load('./data_set/torch/target_np_list.npy')
    state = np.random.get_state()
    np.random.shuffle(np_img)
    np.random.set_state(state)
    np.random.shuffle(np_target)
    torch_img = torch.from_numpy(np_img[:1000]).float().cuda()
    torch_target = torch.from_numpy(np_target[:1000]).cuda()

    # normalize

    # print(torch_img.size())
    # normalize = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     transforms.ToTensor()
    # ])

    # torch_img = normalize(torch_img)

    # torch_target = torch.unsqueeze(torch_target, dim=1)
    data_set = Data.TensorDataset(torch_img, torch_target)
    return data_set


def get_data():
    np_img = np.load('./data_set/torch/img_np_list.npy')
    np_target = np.load('./data_set/torch/target_np_list.npy')
    torch_img = torch.from_numpy(np_img).float()
    torch_target = torch.from_numpy(np_target)
    # torch_target = torch.unsqueeze(torch_target, dim=1)
    data_set = Data.TensorDataset(torch_img, torch_target)
    return data_set


def get_data_loader(my_data_set):
    loader = Data.DataLoader(
        dataset=my_data_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    return loader


if __name__ == "__main__":
    # data_set_ = get_data_cuda()
    # print(data_set_[:100])
    # print(get_vgg16out15())
    # get_vgg16out15(True)

    # net = get_densenet(True)
    # print(net)
    data = get_data_cuda()
    loader = get_data_loader(data)
