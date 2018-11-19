import os
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch optim import lr_scheduler

def obtain_middle_result(module,input,output):
    return output.data

class innerLayer:
    def __init__(self,reallayer):
        self.reallayer=reallayer
        self.initialized=False
        self.databuffer=None
        self.reallayer.register_forward_hook(obtain_middle_result)
        #self._initialize_weights()

    def __call__(self,data):
        if self.initialized == False:
            self._iniialize_weight()
            self.initalized=True
        return self.reallayer(data)
    
    def _initialize_weight(self):
        if isinstance(self.reallayer,nn.Conv2d):                                    
            n = self.reallayer.kernel_size[0] * self.reallayer.kernel_size[1] *
            self.reallayer.in_channels
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
        return self.databuffer=output.data
    def parameters():
        return self.reallayer.paramters()


def getInnerLayer(typename,**args):
    if typename == "conv":
        if 'in_channels' not in args or 'out_channels' not in args or
        'kernel_size' not in args:
            print("layer initialize parameter error")
            return
        return innerLayer(nn.Conv2d(**args))
    if typename == "fc":
        if 'in_channels' not in args or 'out_channels' not in args:
            print("layer initialize parameter error")
            return 
        return innerLayer(nn.Linear(**args))
    
def getLoss(name,**args):
    if name=="MSE":
        return nn.MSELoss(size_average=True)
    else:
        print("loss name not supported")

def getOptimizer(name,**args):
    if name=="SGD":
        return optim.SGD(**args)

def calcLoss(criterion, optimizier, input, target, data_loader):
  #  for i, (data, labels) in enumerate(data_loader):
  #      if use_cuda:
  #          data = data.cuda()
  #      data = Variable(data, valatile=True)
   ##需要输入数据 
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
