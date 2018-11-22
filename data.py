from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from alexnet import alexnet
import torch


def getDataSet():
    
    train_set = datasets.ImageFolder('../../data/images_whole/train', transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(227), transforms.ToTensor()]))
    
    train_loader = DataLoader(train_set, batch_size = 8, shuffle = False, num_workers = 4)
    print("Train dataset size:{}".format(len(train_set)))

    return train_loader

getDataSet()



