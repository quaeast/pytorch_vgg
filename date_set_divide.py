import numpy as np


def read_img_package():
    buffer = np.load('./data_set/numpy_dict.npy', allow_pickle=True)
    print(type(buffer[0]))
    for step, i in enumerate(buffer[0]):
        print(step, '>-----------------------')
        print(i)
        print(buffer[0][i].shape)
    return buffer[0]


def save_namelist(img_dict):
    name_list = []
    for step, i in enumerate(img_dict):
        name_list.append(i)
    print(name_list)
    np.save('./data_set/divided/name_list.npy', np.array(name_list))


def read_name_list():
    return np.load('./data_set/divided/name_list.npy')


def date_set_divide(img_dict):
    test_img_list = []
    test_target_list = []
    train_img_list = []
    train_target_list = []
    for step, key in enumerate(img_dict):
        # test set
        test_img_list.append(img_dict[key][:20])
        img_dict[key] = img_dict[key][20:]
        test_step_tar = np.empty(shape=20)
        test_step_tar[:] = step
        test_target_list.append(test_step_tar)
        # train set
        train_img_list.append(img_dict[key])
        train_step_tar = np.empty(shape=img_dict[key].shape[0])
        train_step_tar[:] = step
        train_target_list.append(train_step_tar)
    test_img_list_np = np.concatenate(test_img_list)
    test_target_list_np = np.concatenate(test_target_list)
    train_img_list_np = np.concatenate(train_img_list)
    train_target_list_np = np.concatenate(train_target_list)

    np.save('./data_set/divided/test_img_list_np.npy', test_img_list_np)
    np.save('./data_set/divided/test_target_list_np.npy', test_target_list_np)
    np.save('./data_set/divided/train_img_list_np.npy', train_img_list_np)
    np.save('./data_set/divided/train_target_list_np.npy', train_target_list_np)


def train_data_set_divide():




if __name__ == '__main__':
    # save_namelist(read_img_package())
    # print(date_set_divide(read_img_package()))
    # date_set_divide(read_img_package())
