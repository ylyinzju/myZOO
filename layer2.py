import os
import torch
import math
import numpy as np
from modules import getInnerLayer
from alexnet import alexnet

class layer:
    def __init__(self,tranable=false,teacherInfo,studentInfo):
        if tranable:
            for t in teacherInfo:
                self.teacher.append(getInnerLayer(t.type,t.args()))
            self.student = getInnerLayer(studentInfo.type,studentInfo.args())
            self.autoEncoder = AutoEncoder(studentInfo.type,[i.out_channels for i in teacherInfo],studentInfo.out_channels)
    
    def processFeatureDistill(self):
        teacherConcatFeature = concatFeature([teacher.get_run_result() for teacher in self.teacher])
 #       distill_feature = self.autoEncoder.train_step(teacherConcatFeature)
#        reconstruct_feature = self.autoEncoder.run_step(distill_feature)
#        loss = self.autoEncoder.criterion(reconstruct_feature, teacherConcatFeature)
        distill_feature, reconstruct_featrue = self.autoEncoder.run_step(teacherConcatFeature)
        loss = self.autoEncoder.cal_loss(teacherConcatFeature, reconstruct_feature)


    def processFeatureAmalgamation(self):
        #do sth

    def processLayerwiseLearning(self):
        #do sth

    
    self.teacher = []
    self.innerLayers = []
    

def main():



if __name__=='__main__':
    main()  
