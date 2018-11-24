from layer import *
from parser import *
from modules import *

class net:
    def __init__(self, namelist, teacherlist, studentlist):
        ##teacher net and student net 
        self.namelist = namelist
        self.teacherlist = teacherlist
        self.studentlist = studentlist
        self.layers = None
        

    def initialize_layers(self):
        index = 0
        for id_x, name in enumerate(self.namelist):
            print(id_x, name)
            if name != "Conv2d" and name != "Linear":
                layers[index] = layer(teacherInfo = self.teacherlist[index], studentInfo = self.studentlist[index],tranable=False)

            if name == "Conv2d":
                layers[index] = layer(teacherInfo = self.teacherlist[index], studentInfo = self.studentlist[index],tranable=True)

            if name == "Linear":
                layers[index] = layer(teacherInfo = self.teacherlist[index], studentInfo = self.studentlist[index],tranable=True)

            index +=1
        self.layers = layers
        


model = '../../hd/teacher1.pb'
model2 = '../../hd/teacher2.pb'
namelist, nameinfo = parser(model)
namelist2, nameinfo2 = parser(model2)
studentlist = [
('Conv2d', 3, 72, (11, 11)),
('ReLU', None, None, None),
('MaxPool2d', None, None, None),
('Conv2d', 72, 210, (5, 5)),
('ReLU', None, None, None),
('MaxPool2d', None, None, None),
('Conv2d', 210, 420, (3, 3)),
('ReLU', None, None, None),
('Conv2d', 420, 320, (3, 3)),
('ReLU', None, None, None),
('Conv2d', 320, 320, (3, 3)),
('ReLU', None, None, None),
('MaxPool2d', None, None, None),
('Dropout', None, None, None),
('Linear', 11520, 4200, None),
('ReLU', None, None, None),
('Dropout', None, None, None),
('Linear', 4200, 4200, None),
('ReLU', None, None, None),
('Linear', 4200, 120, None)
]

teacherlist = []
for i in range(20):
    print(i)
    tmp = []
    teacherlist[i] = tmp.append(nameinfo[i])
    teacherlist[i] = teacherlist[i].append(nameinfo2[i])

print(teacherlist)
Net = net(namelist, teacherlist, studentlist)
Net.initialize_layers()
