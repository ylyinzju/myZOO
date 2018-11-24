import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from alexnet import alexnet
from data import getDataSet
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
from autoEncoder import *

outputs = []
count = 0

def obtain_features(module, input, output):
    outputs.append(output.data)

def add_alexnet_hook(model):
    for conv in model.features:
        conv.register_forward_hook(obtain_features)

    for fc in model.classifier:
        fc.register_forward_hook(obtain_features)

if torch.cuda.is_available():
    print('yes !!!')

teacher1 = alexnet(pretrained=False, num_classes=60)
teacher1.load_state_dict(torch.load('../../snapshot/alexnet-part1/alexnet-part1-030.pkl'))

teacher2 = alexnet(pretrained=False, num_classes=60)
teacher2.load_state_dict(torch.load('../../snapshot/alexnet-part2-100epoch/alexnet-part2-081.pkl'))

add_alexnet_hook(teacher1)
add_alexnet_hook(teacher2)
teacher1 = teacher1.cuda()
teacher2 = teacher2.cuda()

train_loader = getDataSet()

A = AutoEncoder('conv', [64, 64] ,72)
losses = 0
optimizer = A.optimizer
criterion = A.criterion
A.getParam.cuda()
num_image = 0

print(A.encode_net.reallayer.weight)
print(A.encode_net.reallayer.bias)
print(A.decode_net.reallayer.weight)
print(A.decode_net.reallayer.bias)


for epoch in range(10):
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
       
        teacher1(data)
        concatfeature = outputs[0]
        outputs = []
        teacher2(data)
        concatfeature = torch.cat((concatfeature, outputs[0]), dim=1)
        outputs = []

        feed_feat = Variable(concatfeature)

        distill_feature, reconstruct_feature = A.getParam(feed_feat)
        loss = criterion( reconstruct_feature, feed_feat)
        losses = losses + loss.data.cpu().numpy()[0]
        num_image += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%64 == 0:
            print(losses/num_image)
 #   print(feed_feat)
 #   print(reconstruct_feature)
    print('*******************************')

print(A.encode_net.reallayer.weight)
print(A.encode_net.reallayer.bias)
print(A.decode_net.reallayer.weight)
print(A.decode_net.reallayer.bias)
#print(distill_feature)
