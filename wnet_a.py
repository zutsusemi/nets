import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as s
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

NUM_TRAIN = 49000
NUM_VAL = 1000

NC = 10


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

cifar10_train = s.CIFAR10('./samples', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = s.CIFAR10('./samples', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = s.CIFAR10('./samples', train=False, download=True,
                          transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)


for X,y in loader_train:
    print('The shape of X: ', X.shape)
    print('The shape of y', y.shape)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device,'.')
class resmodule(nn.Module):
    '''
    Use the basic B(3,3,...) structure. If using dropout, the block structure changes to :
    Conv(3,3)-dropout-Conv(3,3)-dropout-...-Conv(3,3).
    '''
    def __init__(
        self,
        size_input,
        size_output,
        in_channel,
        use_dropout=False,
        width = 1, # number of channels = 16 * width
        l = 2, # num of conv layers in one block
    ):
        super(resmodule,self).__init__()
        self.in_channel = in_channel
        self.si = size_input
        self.so = size_output
        self.use_dropout = use_dropout
        self.width = 16*width
        self.l = l

        stride = int((self.si - 1)/(self.so - 1))
        temp_channel = self.in_channel
        self.conv3_special = nn.Conv2d(temp_channel, self.width, 3, stride, 1)
        self.conv1 = nn.Conv2d(temp_channel, self.width, 1, stride)
        self.conv = nn.Conv2d(self.width, self.width, 3, 1, 1)
        self.relu = nn.LeakyReLU(1e-1)
        self.b = nn.BatchNorm2d(self.width)
        self.b0 = nn.BatchNorm2d(self.in_channel)
        self.dropout = nn.Dropout(p=0.4)
    
    
    def forward(self, x):
        # temp_size = self.input_size
        for layer in range(self.l):
            if layer == 0:
                out =  self.b0(x)
                out =  self.relu(out)
                out = self.conv3_special(out) #n,32,32,c_i==>n,32,32,self.width
                if self.in_channel == self.width and self.si == self.so:
                    highway = x
                else:
                    highway = self.conv1(x)

            else:
                out =  self.b(out)
                out =  self.relu(out)
                out =  self.conv(out) #n,32,32,c_i==>n,32,32,self.width
            if self.use_dropout == True:
                out = self.dropout(out)
        out = out + highway
        return out



class superresnet(nn.Module):
    def __init__(
        self,
        in_channel,
        use_dropout = False,
        width = 1,
        l=2,
        num_layers = 10,
    ):
        super(superresnet, self).__init__()
        self.in_channel = in_channel
        self.resolution = (32,32,16,8,1)
        self.use_dropout = use_dropout
        self.width = width
        self.num_layers = num_layers

        self.res_2 = resmodule(32, 32, 16*self.width, True, 1* self.width, l)
        self.res_2_t = resmodule(32, 32, 16*self.width, True, 1* self.width, l)
        self.res_3 = resmodule(32, 16, 16*self.width, True, 2 * self.width, l)
        self.res_4 = resmodule(16, 8, 32*self.width, True, 4 * self.width, l)
        self.res_3_t = resmodule(16, 16, 32*self.width, True, 2 * self.width, l)
        self.res_4_t = resmodule(8, 8, 64*self.width, True, 4 * self.width, l)
        self.pool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(1e-1)
        self.bn = nn.BatchNorm2d(64*self.width)
        self.affine = nn.Linear(64*self.width, NC)
        self.conv = nn.Conv2d(self.in_channel, 16*self.width, 3, 1, 1)
        self.n = int((self.num_layers - 4)/6)
        

    
    def forward(self, x):
        out = self.conv(x)
        out = self.res_2(out)
        for _ in range(1, self.n):
            out = self.res_2_t(out)
        out = self.res_3(out)
        for _ in range(1, self.n):
            out = self.res_3_t(out)
        out = self.res_4(out)
        for _ in range(1, self.n):
            out = self.res_4_t(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        o = self.affine(out)
        return o









model = superresnet(3,True,10,2,40).to(device)
# model.load_state_dict(torch.load('./models'))
print(model)

loss_fn = nn.CrossEntropyLoss()



def train(
    dataloader,
    model,
    loss_fn,
    optim = optim.Adam,
    lr = 0.0002
    ):
    size = NUM_TRAIN
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer = optim(model.parameters(), lr, weight_decay=0.001)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(
    dataloader,
    model,
    loss_fn
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, cnt = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cnt+=1
        test_loss /= num_batches
        correct /= cnt * 64
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss


rate=0.001/shrink
shrink=0.955
epochs = 50
performance = []
profile = []
# for rate in [0.001,0.0002,0.0003,0.0006]:
for t in range(epochs):

    if t % 30 == 0:
        rate *= shrink
        torch.save(model.state_dict(), './models__'+str(t))

    print(f"Epoch {t+1}\n-------------------------------")
    print('lr: ', rate)
    
    train(loader_train, model, loss_fn, optim.Adam, rate)
    cor, t_loss = test(loader_val, model, loss_fn)
    profile.append(cor)
    
    performance.append(t_loss)
    # performance.append(cor)

print("Done!")