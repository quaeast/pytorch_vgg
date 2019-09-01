import torch
import torch.utils.data as Data
import torchvision
import numpy as np
import config


def get_vgg16out15(pre=False):
    my_vgg = torchvision.models.vgg16()
    if not pre:
        my_vgg.load_state_dict(torch.load(r'param/vgg16-397923af.pth'))
        my_vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=15, bias=True)
    else:
        my_vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=15, bias=True)
        my_vgg.load_state_dict(torch.load(r'param/my_vgg_out_15_step_50.pth'))
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
    # data_set_ = get_data()
    # print(get_vgg16out15())
    get_vgg16out15(True)
