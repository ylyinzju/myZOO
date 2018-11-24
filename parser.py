import os
import math
import numpy as mp
import torch
from torch import nn
from modules import *

class infostruct:
    def __init__(self,typename, in_channels=None, out_channels=None,kernel_size=None ):
        self.typename = typename
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
    def getargs(self):
        return self.in_channels, self.out_channels, self_kernel_size
    
    def getinfo(self):
        return self.typename, self.in_channels, self.out_channels, self.kernel_size
       # print(self.typename)


def parser(model):
    test = torch.load(model)
    namelist = []
    nameinfo = []
    #namelisthh = {}
    count = 0
    for id_x,m in enumerate(test.modules()):
        if m.__class__.__name__== "Sequential":
#            print("*********************")
#            print(id_x, '->', m)    
            for id_y, m1 in enumerate(m.modules()):
                if m1.__class__.__name__ != "Sequential":
                    print(count,'->',m1)
                    namelist.append(m1.__class__.__name__)
                    count += 1
                    '''   
                    if m1.__class__.__name__ == "Conv2d":
                        namelisthh['type']= 'Conv2d'
                        namelisthh['in_channels']= m1.in_channels
                        namelisthh['out_channels'] = m1.out_channels
                        namelisthh['kernel_size'] = m1.kernel_size
                    elif m1.__class__.__name__ == "Linear":
                        namelisthh['type'] = 'Linear'
                        namelisthh['in_channels'] = m1.in_features
                        namelisthh['out_channels'] = m1.out_features
                    else:
                        namelisthh['type'] = m1.__class__.__name__
                    nameinfo.append(namelisthh)
                    namelisthh = {}
                   '''
                    if m1.__class__.__name__ == "Conv2d":
                        tmp = infostruct(typename = 'Conv2d', in_channels =
                        m1.in_channels, out_channels = m1.out_channels,
                        kernel_size = m1.kernel_size)
                    elif m1.__class__.__name__ == "Linear":
                        tmp = infostruct(typename = 'Linear',in_channels =
                        m1.in_features, out_channels = m1.out_features)
                    ###### for maxpooltype maybe some args need
                    else:
                        tmp = infostruct(typename = m1.__class__.__name__)

                    nameinfo.append(tmp)
    return namelist, nameinfo

####add a funciton to compare whether two teachera can be merge

#### add a function to parser student net

### test ###
'''
model = '../../hd/teacher1.pb'
namelist, nameinfo = parser(model)    
print(namelist)
for m in nameinfo:
    print(m.getinfo())
'''
