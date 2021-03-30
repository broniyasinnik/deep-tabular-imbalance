import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, lr):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                        nn.Linear(32, 32), nn.ReLU(),
                                        nn.Linear(32, 1))
        self.lr = lr
        self.z = None

    def produce_samples(self, n_samples, minority):
        self.n_samples = n_samples
        with torch.no_grad():
            z = torch.tensor(minority + 0.2 * np.random.randn(*minority.shape), dtype=torch.float32)
        z.requires_grad = True
        self.z = z

    def gradient_step(self, gradients):
        self.classifier[0].weight.data = self.classifier[0].weight.data - self.lr * gradients[0]
        self.classifier[0].bias.data = self.classifier[0].bias.data - self.lr * gradients[1]
        self.classifier[2].weight.data = self.classifier[2].weight.data - self.lr * gradients[2]
        self.classifier[2].bias.data = self.classifier[2].bias.data - self.lr * gradients[3]
        self.classifier[4].weight.data = self.classifier[4].weight.data - self.lr * gradients[4]
        self.classifier[4].bias.data = self.classifier[4].bias.data - self.lr * gradients[5]

    def forward(self, x, z_gradients=None):
        if z_gradients:
            layer1_weight = self.classifier[0].weight - self.lr * z_gradients[0]
            layer1_bias = self.classifier[0].bias - self.lr * z_gradients[1]
            layer2_weight = self.classifier[2].weight - self.lr * z_gradients[2]
            layer2_bias = self.classifier[2].bias - self.lr * z_gradients[3]
            layer3_weight = self.classifier[4].weight - self.lr * z_gradients[4]
            layer3_bias = self.classifier[4].bias - self.lr * z_gradients[5]
            out1 = F.linear(x, layer1_weight, layer1_bias)
            out2 = F.linear(out1, layer2_weight, layer2_bias)
            out3 = F.linear(out2, layer3_weight, layer3_bias)
            return out3
        else:
            return self.classifier(x)
