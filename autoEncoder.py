import os
import torch
import modules
from modules import *


class AutoEncoder:
    def __init__(self,typename,teacher,student):
        self.input_size=0
        for i in teacher:
            self.input_size = self.input_size + i
        self.hidden_layer_size = student
        if typename == 'conv':
            self.encode_net = getInnerLayer('Conv2d', in_channels = self.input_size, out_channels = self.hidden_layer_size, kernel_size=1)
            self.decode_net = getInnerLayer('Conv2d', in_channels = self.hidden_layer_size, out_channels = self.input_size, kernel_size=1)

        if typename == 'fc':
            self.encode_net = getInnerLayer('Linear', in_channels = self.input_size, out_channels = self.hidden_layer_size)
            self.decode_net = getInnerLayer('Linear', in_channels = self.hidden_layer_size, out_channels = self.input_size)
        
        self.getParam = GetParams(self.encode_net.reallayer, self.decode_net.reallayer)
        self.criterion = getLoss('MSE')
        self.optimizer = getOptimizer('SGD', self.getParam.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0)
    
    '''
    def parameters(self):
        for pram in [self.encode_net.reallayer,self.decode_net.reallayer]:
            yield pram.parameters()  
    '''

    def run_step(self,x):
        distill_feature, reconstruct_feature = self.getParam(x)
        return distill_feature, reconstruct_feature
    
    '''
    def cal_loss(self, x , target):
        calcLoss(self.criterion, self.optimizier, x ,target, data_loader)        
    
    def train_step(self,data):
        return self.encode_net(data)

    def run_step(self,data):
        return self.decode_net(data)
    '''
    def cal_loss(self, x, target):
        loss = self.criterion(target, x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
'''
A = AutoEncoder('conv', [64,64],72) #initialized autoencoder
img = torch.autograd.Variable((torch.arange(32*32*128).view(1,128,32,32)))#concatfeature
print(img)
for i in range(1):
    print(i)
    print(A)
    dis, res = A.getParam(img) 
 #   print(dis)
    print(res)
    loss = A.cal_loss(img, res)
    print(loss)
'''
