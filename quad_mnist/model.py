import torch.nn as nn
import torch.nn.functional as F


class FCNMulti(nn.Module):
    def __init__(self):
        super(FCNMulti, self).__init__()
        self.c1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.c3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.c5 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.c7 = nn.Conv2d(64, 128, 1)
        self.d8 = nn.Dropout(0.5)
        self.c9 = nn.Conv2d(128, 10, 1)

        self.num_layers = 11
        self.outputs = [0] * self.num_layers

    def __call__(self, x):
        self.outputs[0] = x
        self.outputs[1] = h = self.c1(x)
        self.outputs[2] = h = F.max_pool2d(h, 3, stride=2).relu()
        self.outputs[3] = h = self.c3(h).relu()
        self.outputs[4] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[5] = h = self.c5(h).relu()
        self.outputs[6] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[7] = h = self.c7(h)
        self.outputs[8] = h = self.d8(h).relu()
        self.outputs[9] = h = self.c9(h)
        width = h.shape[-2]
        height = h.shape[-1]
        self.outputs[10] = h = F.avg_pool2d(h, (width, height)).view(-1, 10)
        return h

    def get_weights(self, layer):
        if layer == 1:
            return self.c1.weight.cpu().detach().numpy()
        elif layer == 3:
            return self.c3.weight.cpu().detach().numpy()
        elif layer == 5:
            return self.c5.weight.cpu().detach().numpy()
        elif layer == 7:
            return self.c7.weight.cpu().detach().numpy()
        elif layer == 9:
            return self.c9.weight.cpu().detach().numpy()
        else:
            raise ValueError("Layer does not have weights: {}".format(layer))
