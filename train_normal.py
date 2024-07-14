import matplotlib.pyplot as plt
from component import *
import torch
import torchvision
from torchvision.transforms import transforms
from functools import reduce
import time

NODE = [128, 256, 512, 256]
# NODE = [128, 256]
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

if __name__ == '__main__':
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU.") 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,download=False, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,download=False, transform=transform)
    # img_size = [3, 32, 32]
    # label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [142.65895867347717, 142.20681977272034, 142.87596368789673, 142.56674766540527, 143.09841680526733, 142.9792559146881, 142.42921376228333, 142.50923895835876, 142.51723980903625, 142.70671486854553]
    # 142.65485699176787
    # [197.4711995124817, 197.09233593940735, 196.9389624595642, 196.65071821212769, 196.42269229888916, 196.49675941467285, 196.86926221847534, 197.27102184295654, 196.92987489700317, 196.68608713150024]
    # 196.88289139270782

    # trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,download=False, transform=transform)
    # testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,download=False, transform=transform)
    # img_size = [1, 28, 28]
    # label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [142.89960741996765, 143.1848258972168, 142.58619904518127, 142.9075939655304, 145.38318419456482, 146.16836166381836, 144.62705373764038, 145.37395977973938, 142.82651042938232, 143.17833280563354]
    # 143.9135628938675
    # [142.3551800251007, 142.39623498916626, 142.99327969551086, 142.59698581695557, 142.6131944656372, 143.2145345211029, 142.74769163131714, 142.88124990463257, 142.62017631530762, 142.639564037323]
    # 142.7058091402054

    trainset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=True,download=False, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=False,download=False, transform=transform)
    img_size = [1, 28, 28]
    label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [143.01452350616455, 142.26429343223572, 142.38484907150269, 143.20822668075562, 143.2968201637268, 143.07478523254395, 143.26613688468933, 143.07062149047852, 142.92303323745728, 142.66599655151367]
    # 142.9169286251068
    # [144.07834148406982, 142.98441743850708, 142.96697974205017, 143.31221556663513, 142.39209532737732, 142.49823260307312, 142.40430116653442, 141.94830179214478, 142.9246768951416, 142.30711269378662]
    # 142.781667470932
    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2, drop_last=True)

    node = [reduce((lambda x, y: x * y), img_size)] + NODE + [len(label)]
    network = Network(node, device).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_list = []
    test_loss_list = []
    time_spend_list = []

    network.train()
    for epoch in range(EPOCHS):
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels.reshape(-1, 1)
            labels = torch.zeros(len(labels), len(label)).scatter(1, labels, 1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predict, _ = network(inputs)
            err = criterion(predict, labels)

            err.backward()
            optimizer.step()


            if i % 20 == 0 and i != 0:
                net_test_loss = test_loss(network, testloader, device, criterion)
                time_spend = time.time() - start
                speed = time_spend / i
                train_loss_list.append(err.item())
                test_loss_list.append(net_test_loss)
                print('Epoch: %d, Iteration: %d, Network Train Loss: %.4f, Network Test Loss: %.4f, Speed: %.4fs /batch' % (epoch, i, train_loss_list[-1], net_test_loss, speed))
                
        time_spend_list.append(time_spend)
    
    print(time_spend_list)
    print(sum(time_spend_list) / EPOCHS)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='test loss')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()

        