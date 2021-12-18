from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, classifier: nn.Module = None):
        super(Net, self).__init__()
        self.classifier = classifier

    def gradient_step_(self, lr: torch.Tensor, gradients: Tuple[torch.Tensor], alpha: float = 0.0):
        i = 0
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data = alpha * layer.weight.data + (1 - alpha) * (layer.weight.data - lr * gradients[i])
                layer.bias.data = alpha * layer.bias.data + (1 - alpha) * (layer.bias.data - lr * gradients[i + 1])
                i += 2

        # self.classifier[0].weight.data = self.classifier[0].weight.data - lr * gradients[0]
        # self.classifier[0].bias.data = self.classifier[0].bias.data - lr * gradients[1]
        # self.classifier[2].weight.data = self.classifier[2].weight.data - lr * gradients[2]
        # self.classifier[2].bias.data = self.classifier[2].bias.data - lr * gradients[3]
        # self.classifier[4].weight.data = self.classifier[4].weight.data - lr * gradients[4]
        # self.classifier[4].bias.data = self.classifier[4].bias.data - lr * gradients[5]

    def forward(self, x, lr: torch.Tensor = None, gradients: torch.Tensor = None):
        if gradients:
            linear_layers = []
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    linear_layers.extend([layer.weight, layer.bias])

            updated_layers = []
            for layer_params, z_grads in zip(linear_layers, gradients):
                update_layer = layer_params - lr * z_grads
                updated_layers.append(update_layer)

            layer_weights = updated_layers[0::2]
            bias_weights = updated_layers[1::2]
            out = x
            last = len(bias_weights)
            for weight, bias in zip(layer_weights, bias_weights):
                out = F.linear(out, weight, bias)
                if last != 1:
                    out = F.relu(out)
                    last -= 1
            return out
            # layer1_weight = self.classifier[0].weight - lr * z_gradients[0]
            # layer1_bias = self.classifier[0].bias - lr * z_gradients[1]
            # layer2_weight = self.classifier[2].weight - lr * z_gradients[2]
            # layer2_bias = self.classifier[2].bias - lr * z_gradients[3]
            # layer3_weight = self.classifier[4].weight - lr * z_gradients[4]
            # layer3_bias = self.classifier[4].bias - lr * z_gradients[5]
            # self._gradient_step(lr, gradients=z_gradients)
            # out1 = F.linear(x, layer1_weight, layer1_bias)
            # out1 = F.relu(out1)
            # out2 = F.linear(out1, layer2_weight, layer2_bias)
            # out2 = F.relu(out2)
            # out3 = F.linear(out2, layer3_weight, layer3_bias)
            # return out3
            # return out2
        else:
            return self.classifier(x)
