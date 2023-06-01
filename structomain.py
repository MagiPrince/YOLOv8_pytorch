from sturcto import *
import torch
import torch.nn as nn

inputs = torch.full((1, 3, 64, 64), 1.)

print(inputs)

x = Conv(3, 64*0.25, 3, 2)(inputs)
x = Conv(64*0.25, 128*0.25, 3, 2)(x)
x = C2f(128*0.25, 128*0.25, 3*0.33, True)(x)
x = Conv(128*0.25, 256*0.25, 3, 2)(x)
x_4 = C2f(256*0.25, 256*0.25, 6*0.33, True)(x)
x = Conv(256*0.25, 512*0.25, 3, 2)(x_4)
x_6 = C2f(512*0.25, 512*0.25, 6*0.33, True)(x)
x = Conv(512*0.25, 512*0.25*2.0, 3, 2)(x_6)
x = C2f(512*0.25*2.0, 512*0.25*2.0, 3*0.33, True)(x)
x_9 = SPPF(512*0.25*2.0, 512*0.25*2.0)(x)
x = nn.Upsample(scale_factor=2)(x_9)
x = torch.cat((x_6, x), dim=1)
x_12 = C2f(512*0.25*(1+2.0), 512*0.25, False, 3*0.33)(x)
x = nn.Upsample(scale_factor=2)(x_12)
x = torch.cat((x_4, x), dim=1)
x_15 = C2f(768*0.25, 256*0.25, False, 3*0.33)(x)
x = Conv(256*0.25, 256*0.25, 3, 2)(x_15)
x = torch.cat((x_12, x), dim=1)
x_18 = C2f(768*0.25, 512*0.25, False, 3*0.33)(x)
x = Conv(512*0.25, 512*0.25, 3, 2)(x_18)
x = torch.cat((x_9, x), dim=1)
x_21 = C2f(512*0.25*(1+2.0), 512*0.25*2.0, False, 3*0.33)(x)

y_1, a = Detect(256*0.25, 1)(x_15)
y_2, b = Detect(512*0.25, 1)(x_18)
y_3, c = Detect(512*0.25*2.0, 1)(x_21)

# print(y_1)
# print(y_2)
# print(y_3.size())
# print(a.size())
# print(b.size())
# print(c.size())
y, x_prim = DetectThree(ch=[x_15.size()[1], x_18.size()[1], x_21.size()[1]], nc=1)([x_15, x_18, x_21])
print(y.size())
print(y)