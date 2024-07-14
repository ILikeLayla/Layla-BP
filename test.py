from component import *
import torch

BATCH_SIZE = 5
NODE = [100, 64, 32, 10]

# dl_dz = torch.rand(1, BATCH_SIZE, NODE[-1])
inputs = torch.rand(BATCH_SIZE, 1, 10, 10)
label = torch.randint(0, 10, (BATCH_SIZE, 1))
one_hot = torch.zeros(BATCH_SIZE, 10).scatter(1, label, 1)

network = Network(*NODE)
bp_net = BPNet(NODE, 2, BATCH_SIZE)

# print(network.get_parameters_shape())

predict, a_list = network(inputs)
dl_dz = CrossEntropy_derivative(one_hot, predict)

out, h = bp_net(network.get_parameters(), dl_dz, a_list)

for i in out:
    print(i.shape, end="/ ")

updates = [torch.mean(i, dim=1).t() for i in out]
network.update_weights(updates)