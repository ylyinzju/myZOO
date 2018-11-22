import os
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import math


class innerLayer:
    def __init__(self,reallayer):
        print('1')
        self.reallayer = reallayer
        self.initialized = False
        self.databuffer = None 
        self.reallayer.register_forward_hook(self.obtain_middle_result)

    def __call__(self,data):
        print('2')
        if self.initialized == False:
            self._initialize_weight()
            self.initialized = True
        return self.reallayer(data)
    
    def _initialize_weight(self):
        if isinstance(self.reallayer, nn.Conv2d):                                    
            n = self.reallayer.kernel_size[0] * self.reallayer.kernel_size[1] * self.reallayer.in_channels
            value = math.sqrt(3. / n)
            self.reallayer.weight.data.uniform_(-value, value)
            if self.reallayer.bias is not None:
                self.reallayer.bias.data.zero_()
        elif isinstance(self.reallayer, nn.Linear):
            n = self.reallayer.in_features
            value = math.sqrt(3. / n)
            self.reallayer.weight.data.uniform_(-value, value)
            if self.reallayer.bias is not None:
                self.reallayer.bias.data.zero_()
    
    def get_run_result(self):
        return self.databuffer

    def obtain_middle_result(self, module,input,output):
#        print('3')
        self.databuffer = output.data

    def parameters(self):
        yield self.reallayer.parameters()


def getInnerLayer(typename,**args):
    if typename == "conv":
        if 'in_channels' not in args or 'out_channels' not in args or 'kernel_size' not in args:
            print("layer initialize parameter error")
            return
        return innerLayer(nn.Conv2d(**args))
    if typename == "fc":
        if 'in_channels' not in args or 'out_channels' not in args:
            print("layer initialize parameter error")
            return 
        return innerLayer(nn.Linear(**args))


class GetParams(nn.Module):
    def __init__(self, encode, decode):
        super(GetParams, self).__init__()

        self.conv1 = encode
        self.conv2 = decode 

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)    
        return x1,x2

    
def getLoss(name,**args):
    if name == "MSE":
        return nn.MSELoss(size_average = True)
    else:
        print("loss name not supported")
'''
def getOptimizer(name,**args):
    if name == "SGD":
        return optim.SGD(**args)
    else:
        print("optimizer name not supported")
'''
def getOptimizer(name, param, lr, momentum, weight_decay):
    if name == 'SGD':
        return optim.SGD(param, lr, momentum, weight_decay)
    else:
        print("optimizer name not supported")

###################not ok
def calcLoss(criterion, optimizier, input, target, data_loader):
  #  for i, (data, labels) in enumerate(data_loader):
  #      if use_cuda:
  #          data = data.cuda()
  #      data = Variable(data, valatile=True)
    loss = criterion(target, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def concatFeature(featurelist):
    resultfeature=None
    for feature in featurelist:
        if resultfeature==None:
            resultfeature=feature
        else:
            resultfeature=torch.cat((resultfeature,feature),dim=1)
    return resultfeature

'''
L = getInnerLayer('conv', in_channels=64, out_channels=24, kernel_size=1)
img = torch.autograd.Variable((torch.arange(32*32*64).view(1,64,32,32)))
#handle = L.reallayer.register_forward_hook(L.obtain_middle_result)
L(img)
print(L.get_run_result())
print(L.parameters())
'''
