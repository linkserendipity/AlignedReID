import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()


    def forward(self, x):
        # x feature map
        # pooling height = 1,  pooling width = W
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))


if __name__ == "__main__":
    import torch
    x = torch.tensor(32,2048,8,4)
    hp = HorizontalMaxPool2d()
    y = hp()
    print(y.shape)
