import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Parameters:
        c_in (int): number of channel of the input tensor
        c_out (int): number of channel of the output tensor
        kernel (int): 
        stride (int): 
        padding (int): 
    """
    def __init__(self, c_in, c_out, kernel, stride, padding):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-5, momentum=0.1),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, input):
        return self.conv(input)


class Bottleneck(nn.Module):
    """
    Parameters:
        c_ (int): number of channel of the input tensor
        shortcut (boolean): indicates if the shorcut should be used or not
    """
    def __init__(self, c_, shortcut):
        super(Bottleneck, self).__init__()

        self.shortcut = shortcut
        # self.bottleneck = nn.Sequential(
        #     Conv(c_, int(c_*0.5), 3, 1, 1),
        #     Conv(int(c_*0.5), c_, 3, 1, 1),
        # )
        self.cv1 = Conv(c_, int(c_*0.5), 3, 1, 1)
        self.cv2 = Conv(int(c_*0.5), c_, 3, 1, 1)

    def forward(self, input):
        # if self.shortcut:
        #     return self.bottleneck(input) + input
        return input + self.cv2(self.cv1(input)) if self.shortcut else self.cv2(self.cv1(input))


class BottleneckTest(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class C2fCustom(nn.Module):
    def __init__(self, c_in, c_out, shortcut, n):
        super(C2fCustom, self).__init__()

        self.c_out = c_out
        self.n = n

        self.c1 = Conv(c_in, c_out, 1, 1, 0)
        self.bottleneck = BottleneckTest(int(c_out*0.5), int(c_out*0.5), shortcut, g=1, k=((3, 3), (3, 3)))
        self.c_last = Conv(int(0.5*(n+2)*c_out), c_out, 1, 1, 0)

    def forward(self, input):
        tmp_res = self.c1(input)
        # Split tensor in two in the canal dimension
        splitted_tensors = torch.split(tmp_res, int(0.5*self.c_out), 1)
        next_i = splitted_tensors[-1]

        concated_tensors = torch.cat(splitted_tensors, 1)

        # iterates n times over the bottleneck block and cat the results to the previous results
        for i in range((self.n)):
            next_i = self.bottleneck(next_i)
            concated_tensors = torch.cat((concated_tensors, next_i), 1)

        return self.c_last(concated_tensors)



class C2f(nn.Module):
    def __init__(self, c_in, c_out, shortcut, n):
        super(C2f, self).__init__()

        self.c_out = c_out
        self.n = n

        self.c1 = Conv(c_in, c_out, 1, 1, 0)
        self.bottleneck = Bottleneck(int(0.5*c_out), shortcut)
        self.c_last = Conv(int(0.5*(n+2)*c_out), c_out, 1, 1, 0)

    def forward(self, input):
        tmp_res = self.c1(input)
        # Split tensor in two in the canal dimension
        splitted_tensors = torch.split(tmp_res, int(0.5*self.c_out), 1)
        next_i = splitted_tensors[-1]

        concated_tensors = torch.cat(splitted_tensors, 1)

        # iterates n times over the bottleneck block and cat the results to the previous results
        for i in range((self.n)):
            next_i = self.bottleneck(next_i)
            concated_tensors = torch.cat((concated_tensors, next_i), 1)

        return self.c_last(concated_tensors)


class SPPF():
    def __init__(self, c_in, c_out):
        super(SPPF, self).__init__()

        c_ = int(c_in//2)

        self.c1 = Conv(c_in, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_last = Conv(c_*4, c_out, 1, 1, 0)

    def forward(self, input):
        x = self.c1(input)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_last(torch.cat((x, pool1, pool2, pool3), 1))


class Detect():
    def __init__(self, c_, nc=1):
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
