import torch.nn as nn

class C(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, padding, stride, bias=True):
        super(C, self).__init__()
        self.layers = nn.Conv2d(in_channel, out_channel, kernel_size=(ksize, ksize), padding=padding,
                                stride=stride, bias=bias)

    def forward(self, input):
        return self.layers(input)
