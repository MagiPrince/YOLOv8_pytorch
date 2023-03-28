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
            nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03),
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
        self.bottleneck = nn.Sequential(
            Conv(c_, c_*0.5, 3, 1, 1),
            Conv(c_*0.5, c_, 3, 1, 1),
        )

    def forward(self, input):
        if self.shortcut:
            return self.bottleneck(input) + input
        return self.bottleneck(input)


class C2f(nn.Module):
    def __init__(self, c_in, c_out, shortcut, n):
        super(C2f, self).__init__()

        self.c_out = c_out
        self.n = n

        self.c1 = Conv(c_in, c_out, 1, 1, 0)
        self.bottleneck = Bottleneck(0.5*c_out, shortcut)
        self.c_last = Conv(0.5*(n+2)*c_out, c_out, 1, 1, 0)

    def forward(self, input):
        tmp_res = self.c1(input)
        # Split tensor in two in the canal dimension
        splitted_tensors = torch.split(tmp_res, 0.5*self.c_out, 2)
        next_i = splitted_tensors[-1]

        concated_tensors = torch.cat(splitted_tensors, 2)

        # iterates n times over the bottleneck block and cat the results to the previous results
        for i in range(len(self.n)):
            next_i = self.bottleneck(next_i)
            concated_tensors = torch.cat((concated_tensors, next_i), 2)

        return self.c_last(concated_tensors)


class SPPF():
    pass


class Detect():
    pass
