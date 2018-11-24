import os
import torch
import math
import numpy as np
from modules import getInnerLayer
from alexnet import alexnet
from autoEncoder import *

class layer:
    def __init__(self,teacherInfo,studentInfo,tranable=False):
        if tranable:
            for t in teacherInfo:
                self.teacher.append(getInnerLayer(t.typename,t.getargs()))
            self.student = getInnerLayer(studentInfo.typename,studentInfo.getargs())
            self.autoEncoder = AutoEncoder(studentInfo.type,[i.out_channels for i in teacherInfo],studentInfo.out_channels)
        else:#### initialize a none-trainable layer
            #### should be able to use the type to construct a nn.Relu
            self.teacher = innerLayer(teacherInfo.type)
            self.student = None
            self.autoEncoder = None
        

    def processFeatureDistill(self):
        teacherConcatFeature = concatFeature([teacher.get_run_result() for teacher in self.teacher])
        distill_feature, reconstruct_featrue = self.autoEncoder.run_step(teacherConcatFeature)
        loss = self.autoEncoder.cal_loss(teacherConcatFeature, reconstruct_feature)
        

    def processFeatureAmalgamation(self):
        #do sth
        return 

    def processLayerwiseLearning(self):
        #do sth
        return 
    
    #self.teacher = []
    #self.innerLayers = []
    

def main():
    return


if __name__=='__main__':
    main()  
