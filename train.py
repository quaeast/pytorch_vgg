import torch
import plant_classify
import config
torch.manual_seed(1)

my_vgg = plant_classify.get_vgg16out15()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_vgg.parameters(), lr=config.LR)

loader = plant_classify.get_data_loader(plant_classify.get_data())


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
            ' | loss:', loss
              )
