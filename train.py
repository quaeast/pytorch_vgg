import torch
import plant_classify
import config
import numpy as np
# torch.manual_seed(1)

my_vgg = plant_classify.get_vgg16out15().cuda()
loss_func = torch.nn.CrossEntropyLoss()


loader = plant_classify.get_data_loader(plant_classify.get_data_cuda())

loss_list = []

for decay_index in range(config.DECAY_INDEX):
    optimizer = torch.optim.SGD(my_vgg.parameters(), lr=(config.LR/pow(10, decay_index)))
    for e in range(config.EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            output = my_vgg(batch_x)
            print(e, '>---------------------------------------------')
            # print(output.type())
            # print(batch_y.type())
            loss = loss_func(output, batch_y.long())
            loss.backward()
            optimizer.step()
            print(
                'step:', step,
                ' | loss:', loss.item()
            )
            loss_list.append(loss.item())

# print(loss_list)
np.save('result/loss.npy', np.array(loss_list))
torch.save(my_vgg.state_dict(), './param/my_vgg_out_15_step_50.pth')

