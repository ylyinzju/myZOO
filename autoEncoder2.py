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
            self.encode_net = getInnerLayer('conv', in_channels = self.input_size, out_channels = self.hidden_layer_size, kernel_size=1)
            self.decode_net = getInnerLayer('conv', in_channels = self.hidden_layer_size, out_channels = self.input_size, kernel_size=1)

        if typename == 'fc':
            self.encode_net = getInnerLayer('fc', in_channels = self.input_size, out_channels = self.hidden_layer_size)
            self.decode_net = getInnerLayer('fc', in_channels = self.hidden_layer_size, out_channels = self.input_size)
        self.criterion = getLoss('MSE')
        self.optimizer = getOptimizer('SGD', parameters(), lr=0.01, momentum=0.9, weight_decay=0)

    def parameters():
        for pram in [self.encode_net,self.decode_net]:
            yield pram.parameters()

    def run_step(self,x):
        distill_feature = self.encode_net(x)
        reconstruct_feature = self.decode_net(distill_feature)
        return distill_feature, reconstruct_featrue

    def cal_loss(self, x , target):
        calcLoss(self.criterion, self.optimizier, x ,target, data_loader)        
    #def train_step(self,data):
    #    return self.encode_net(data)
    
   # def run_step(self,data):
    #    return self.decode_net(data)
    '''
    def cal_loss(self,x):
        distill_feature = train_step(x)
        reconstruct_feature = run_step(distill_feature)
        loss = self.criterion(x, reconstruct_feature)
        self.optimizier.zero_grad()
        loss.backward()
        self.optimizier.step()
    '''



