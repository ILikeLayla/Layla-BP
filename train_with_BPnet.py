import matplotlib.pyplot as plt
from component import *
import torch
import torchvision
from torchvision.transforms import transforms
from functools import reduce
import time

# NODE = [128, 256, 512, 256]
NODE = [128, 256]
BP_LR = 2e-5
NET_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
NUM_LAYERS = 2

if __name__ == '__main__':
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU.")
    buffer_device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,download=False, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,download=False, transform=transform)
    # img_size = [3, 32, 32]
    # label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [634.5614676475525, 743.0273356437683, 661.6675550937653, 643.1297233104706, 666.0178668498993, 657.1269762516022, 658.3760232925415, 681.7469754219055, 646.1022219657898, 647.4780356884003]
    # 663.9234181165696
    # [379.32507133483887, 369.56974387168884, 368.3130798339844, 368.95078229904175, 370.5482394695282, 372.466436624527, 368.6993930339813, 620.135359287262, 424.29627752304077, 377.38761043548584]
    # 401.9691993713379

    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,download=False, transform=transform)
    img_size = [1, 28, 28]
    label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [326.9892065525055, 327.2847099304199, 327.5886905193329, 327.00810146331787, 327.1637589931488, 326.9982736110687, 339.8074300289154, 333.9631824493408, 329.47580075263977, 326.1028995513916]
    # 329.2382053852081
    # [212.8362214565277, 212.9779336452484, 213.2959339618683, 213.38261032104492, 212.1617419719696, 213.24831199645996, 212.35106372833252, 213.22455549240112, 213.06076645851135, 212.71619844436646]
    # 212.92553374767303

    # trainset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=True,download=False, transform=transform)
    # testset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=False,download=False, transform=transform)
    # img_size = [1, 28, 28]
    # label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [366.18963527679443, 365.5438024997711, 366.4287836551666, 366.54384326934814, 367.18393874168396, 366.46316719055176, 367.64717054367065, 367.7793548107147, 367.5716528892517, 367.09298372268677]
    # 366.844433259964
    # [214.46444392204285, 213.79949641227722, 214.0853033065796, 213.70693707466125, 214.5185308456421, 215.3571970462799, 214.4174952507019, 214.76714634895325, 215.47247767448425, 214.42804265022278]
    # 214.5017070531845

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2, drop_last=True)

    node = [reduce((lambda x, y: x * y), img_size)] + NODE + [len(label)]
    network = Network(node, device).to(device)
    bp_net = BPNet(node, NUM_LAYERS, device).to(device)
    buffer_pool = BufferPool(buffer_device, device, NUM_LAYERS, node)

    bp_optimizer = torch.optim.Adam(bp_net.parameters(), lr=BP_LR)
    bp_criterion = torch.nn.MSELoss()
    net_optimizer = torch.optim.Adam(network.parameters(), lr=NET_LR)
    net_criterion = torch.nn.CrossEntropyLoss()

    net_train_loss_list = []
    bp_loss_list = []
    net_test_loss_list = []
    time_spend_list = []

    for epoch in range(EPOCHS):
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            
            bp_optimizer.zero_grad()

            inputs, labels = data
            labels = labels.reshape(-1, 1)
            labels = torch.zeros(len(labels), len(label)).scatter(1, labels, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            predict, a_list = network(inputs)
            dldz = CrossEntropy_derivative(labels, predict)
            
            out, grad = bp_net(network.get_parameters(), dldz, a_list)
            updates = [torch.mean(i, dim=1).t() for i in out]
            buffer_pool.push(dldz, a_list)

            bp_loss_1 = bp_criterion(grad, torch.zeros_like(grad))
            bp_loss_1.backward(retain_graph=True)
            bp_loss_2 = bp_criterion(dldz, torch.zeros_like(dldz))
            bp_loss_2.backward()

            network_buffer = network.clone()
            network_buffer.update_weights(updates, lr = NET_LR)
            new_predict, _ = network_buffer(inputs)
            bp_loss_3 = net_criterion(new_predict, labels)
            bp_loss_3.backward()

            bp_optimizer.step()

            if i % 5 == 0 and i != 0:
                net_optimizer.zero_grad()
                dldz, a_list = buffer_pool.get()
                out, grad = bp_net(network.get_parameters(), dldz, a_list)
                updates = [torch.mean(i, dim=1).t() for i in out]
                network.update_weights(updates, lr = NET_LR)
                net_optimizer.step()

            if i % 20 == 0 and i != 0:
                net_test_loss = test_loss(network, testloader, device, net_criterion)
                time_spend = time.time() - start
                speed = time_spend / i
                bp_loss = bp_loss_2 + bp_loss_1 + bp_loss_3
                bp_loss_list.append(bp_loss.item())
                net_train_loss_list.append(net_criterion(predict, labels).item())
                net_test_loss_list.append(net_test_loss)
                print('Epoch: %d, Iteration: %d, Network Train Loss: %.4f, BP Loss: %.4f, Network Test Loss: %.4f, Speed: %.4fs /batch' % (epoch, i, net_train_loss_list[-1], bp_loss_list[-1], net_test_loss, speed))
                
        time_spend_list.append(time.time() - start)
    
    print(time_spend_list)
    print(sum(time_spend_list) / EPOCHS)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    f, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(bp_loss_list, label='BP loss')
    ax2.plot(net_train_loss_list, label='network train loss')
    ax2.plot(net_test_loss_list, label='network test loss')
    plt.show()

        