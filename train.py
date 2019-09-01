import torch
import plant_classify
import config
import numpy as np
torch.manual_seed(1)

my_vgg = plant_classify.get_vgg16out15()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_vgg.parameters(), lr=config.LR)

loader = plant_classify.get_data_loader(plant_classify.get_data())

loss_list = []

for e in range(config.EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        output = my_vgg(batch_x)
        print('>---------------------------------------------')
        print(output.type())
        print(batch_y.type())
        loss = loss_func(output, batch_y.long())
        loss.backward()
        optimizer.step()
        print(
            'step:', step,
            ' | loss:', loss.item()
              )
        loss_list.append(loss.item())
        if step == 100:
            break

print(loss_list)
np.save('result/loss.npy', np.array(loss_list))
torch.save(my_vgg.state_dict(), './param/my_vgg_out_15_step_50.pth')

