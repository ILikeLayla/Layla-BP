from typing import Union
from torch import nn
import torch

class Network(nn.Module):
    def __init__(self, node, device, activation_func=torch.relu):
        super(Network,self).__init__()
        self.activation_func = activation_func
        self.device = device
        self.linear = nn.ModuleList([
            nn.Linear(in_features=node[i], out_features=node[i+1], bias=False)
            for i in range(len(node)-1)
        ])

    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        a_list = []
        for i in range(len(self.linear)):
            a = self.linear[i](x)
            x = self.activation_func(a)
            a_list.append(a)
        return torch.softmax(x, dim=-1), a_list
    
    def get_parameters(self):
        return [self.linear[i].weight for i in range(len(self.linear))]
    
    def get_parameters_shape(self):
        return [self.linear[i].weight.shape for i in range(len(self.linear))]
    
    def update_weights(self, updates: list[torch.tensor], lr: float= 0.1):
        for i in range(len(self.linear)):
            self.linear[i].weight = nn.Parameter(torch.sub(self.linear[i].weight, updates[i] * lr))
    
    def clone(self):
        buffer =  Network([1], self.device, self.activation_func)
        buffer.linear = self.linear
        return buffer

class FitNet(nn.Module):
    def __init__(self, node: list[int]):
        super(FitNet, self).__init__()
        buffer = [node[0]] + node
        self.main = nn.ModuleList([nn.Linear(in_features=buffer[i], out_features=buffer[i+1], bias=False) for i in range(len(node)-1)])
        print(self.main)
        

class BPNet(nn.Module):
    def __init__(self, node: list[int], num_layers: int, device, activation_func=torch.relu):
        super(BPNet, self).__init__()
        node = node[::-1]
        self.fit_net = FitNet(node)
        self.device = device
        buffer = node
        self.num_layers = num_layers
        self.activation_func = activation_func
        self.main = nn.ModuleList([
            nn.RNN(input_size=buffer[i], hidden_size=buffer[i], num_layers=num_layers)
            for i in range(len(node)-1)
        ])
        print(self.main)

    def forward(self, parameters: list, dl_dz: torch.tensor, a_list: list[torch.tensor]):
        # dy_dl [batch, output_num]
        parameters = parameters[::-1]
        a_list = a_list[::-1]
        
        h = torch.stack([dl_dz] * self.num_layers, dim=0)
        # h [num_layers, batch, output_num]

        output = []
        for p, fit, rnn, a in zip(parameters, self.fit_net.main, self.main, a_list):
            # p [output_num, input_num]
            # a [batch, output_num]

            p = torch.stack([p.t()] * dl_dz.shape[0], dim=1)
            # p [output_num, batch, input_num]

            fitted = fit(h)
            # fitted [num_layers, batch, output_num]
            
            dz_da = get_activation_derivative(self.activation_func)(a).to(self.device)
            # dz_da [batch, output_num, output_num]


            dz_da = torch.stack([dz_da] * self.num_layers, dim=0)
            # dz_da [num_layers, batch, output_num]

            dl_da = torch.bmm(fitted, dz_da)
            # dl_da [num_layers, batch, output_num]

            # dl_da [num_layers, batch, (hidden_size)10]
            # p [(seq_len), batch, (input_size)]
            out, h = rnn(p, dl_da)
            # h [num_layers, batch, (hidden_size)10]

            output.append(out)
        
        return output[::-1], h
    
    def get_fit_shape(self):
        return [self.fit_net.main[i].weight.shape for i in range(len(self.fit_net.main))]

class BufferPool:
    def __init__(self, buffer_device, device, num_layers, node):
        self.buffer_device = buffer_device
        self.device = device
        self.node = node
        self.dldz = torch.zeros((1, node[-1])).to(self.buffer_device)
        self.a_list = [torch.zeros((1, node[i+1])).to(self.buffer_device) for i in range(len(node)-1)]
    
    def push(self, dldz, a_list):
        dldz = dldz.to(self.buffer_device)
        a_list = [i.to(self.buffer_device) for i in a_list]
        self.dldz = torch.cat((self.dldz, dldz), dim=0)
        self.a_list = [torch.cat((a, a_list[i]), dim=0) for i, a in enumerate(self.a_list)]
    
    def get(self):
        output = self.dldz[1:].to(self.device), [i[1:].to(self.device) for i in self.a_list]
        self.dldz = torch.zeros((1, self.node[-1])).to(self.buffer_device)
        self.a_list = [torch.zeros((1, self.node[i+1])).to(self.buffer_device) for i in range(len(self.node)-1)]
        return output

def MSE(ground_truth, prediction):
    return torch.mean((ground_truth - prediction) ** 2)

def MSE_derivative(ground_truth, prediction):
    return 2 * (prediction - ground_truth)

def CrossEntropy(ground_truth, prediction):
    return -torch.mean(torch.sum(ground_truth * torch.log(prediction), dim=-1))

def CrossEntropy_derivative(ground_truth, prediction):
    return ground_truth - prediction

def ddx_relu(x):
    shape = x.shape[1]
    arr = torch.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = torch.where(x > 0, 1, 0)[0][i]
    return arr

def ddx_sigmoid(x):
    shape = x.shape[1]
    arr = torch.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = torch.sigmoid(x)[0][i] * (1 - torch.sigmoid(x)[0][i])
    return arr

def ddx_tanh(x):
    shape = x.shape[1]
    arr = torch.zeros((shape, shape))
    for i in range(shape):
        arr[i][i] = 1 - torch.tanh(x)[0][i] ** 2
    return arr

__derivative_list = {
        torch.relu: ddx_relu,
        torch.sigmoid: ddx_sigmoid,
        torch.tanh: ddx_tanh
    }

def get_activation_derivative(activation_func):
    return __derivative_list[activation_func]

def test_loss(model, testloader, device, criterion):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss += criterion(outputs, labels).item()
    return loss / len(testloader)