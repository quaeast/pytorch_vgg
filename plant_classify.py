import torch
import torch.utils.data as Data
import torchvision
import numpy as np
import config


def get_vgg16out15():
    my_vgg = torchvision.models.vgg16()
    my_vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=15, bias=True)
    return my_vgg


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
        shuffle=True
    )
    return loader


if __name__ == "__main__":
    data_set_ = get_data()


    # print(get_vgg16out15())
