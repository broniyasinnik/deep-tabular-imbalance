import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_samples):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                        nn.Linear(32, 32), nn.ReLU(),
                                        nn.Linear(32, 1))
        self.n_samples = n_samples
        # sample random points
        z = torch.normal(0, 3, size=(n_samples, 2))
        z.requires_grad = True
        self.z = z

    def gradient_step(self, gradients):
        params = self.classifier.parameters()
        for i, gradient in enumerate(gradients):
            params[i] = params[i] - gradient

    def forward(self, x, z_gradients=None):
        if z_gradients:
            alpha = 0.1
            layer1_weight = self.classifier[0].weight - alpha*z_gradients[0]
            layer1_bias = self.classifier[0].bias - alpha*z_gradients[1]
            layer2_weight = self.classifier[2].weight - alpha*z_gradients[2]
            layer2_bias = self.classifier[2].bias - alpha*z_gradients[3]
            layer3_weight = self.classifier[4].weight - alpha*z_gradients[4]
            layer3_bias = self.classifier[4].bias - alpha*z_gradients[5]
            out1 = F.linear(x, layer1_weight, layer1_bias)
            out2 = F.linear(out1, layer2_weight, layer2_bias)
            out3 = F.linear(out2, layer3_weight, layer3_bias)
            return out3
        else:
            return self.classifier(x)
