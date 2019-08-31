import numpy as np
import torch
import torchvision


def read_img_package():
    buffer = np.load('./data_set/numpy_dict.npy', allow_pickle=True)
    print(type(buffer[0]))
    for step, i in enumerate(buffer[0]):
        print(step, '>-----------------------')
        print(i)
        print(buffer[0][i].shape)
    return buffer[0]


def get_img_list(img_dict):
    img_list = []
    for i in img_dict:
        img_list.append(img_dict[i])
    img_np_list = np.concatenate(img_list)
    img_np_list = img_np_list.transpose((0, 3, 1, 2))
    np.save('./data_set/torch/img_np_list.npy', img_np_list)
    print(img_np_list.shape)


def get_target_list(img_dict):
    target_list = []
    for step, i in enumerate(img_dict):
        cur_list = np.zeros(shape=[img_dicts[i].shape[0]])
        cur_list[:] = step
        target_list.append(cur_list)
    target_np_list = np.concatenate(target_list)
    # target_np_list = np.eye(15)[target_np_list]
    np.save('./data_set/torch/target_np_list.npy', target_np_list)
    # print(target_np_list)
    print(target_np_list.shape)


if __name__ == '__main__':
    img_dicts = read_img_package()
    # get_img_list(img_dicts)
    get_target_list(img_dicts)

