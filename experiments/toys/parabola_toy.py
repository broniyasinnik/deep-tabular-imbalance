from IPython import display
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)


class Parabola(nn.Module):
    def __init__(self):
        super(Parabola, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.))
        self.b = nn.Parameter(torch.tensor(-3.))
        self.c = nn.Parameter(torch.tensor(2.))

    def initilaize_params(self):
        self.a.data = torch.tensor(0.)
        self.b.data = torch.tensor(-3.)
        self.c.data = torch.tensor(2.)

    def forward(self, x):
        y = self.a * (x ** 2) + self.b * x + self.c
        return y


hyper_parameters = {
    'alpha1': 1e-4,
    'momentum1': 0.9,
    'alpha2': 4,
    'momentum2': 0.9,
    'milestones': [50, 80]
}

a_real = torch.tensor([2.])
b_real = torch.tensor([-1.])
c_real = torch.tensor([1.])
parabola_real = lambda x: a_real * x ** 2 + b_real * x + c_real
# The regressor
parabola = Parabola()
# Real Data
x_real = torch.tensor([-3., -1., 0., 1., 2.5, 3., 3.5, 5.])
y_real = a_real * x_real ** 2 + b_real * x_real + c_real

# Synthetic Data
z = torch.tensor([0, 30]) + torch.normal(0, 5, size=(10, 2))
z_x = z[:, 0]
z_x.requires_grad = True
z_y = z[:, 1]


@torch.no_grad
def visualization(parabola, parabola_real, x_real, y_real, z):
    x = np.linspace(-5, 5)
    y = parabola(x)
    plt.plot(x, y, label="regressor")
    plt.plot(x, parabola_real(x), c='g', label='realizable')
    plt.scatter(x_real, y_real)
    plt.scatter(z[:, 0], z[:, 1], label="syntetic")
    plt.legend()
    plt.show()


def get_parabola_optimizer():
    optimizer_parabola = torch.optim.Adam(parabola.parameters(), lr=hyper_parameters['alpha1'], betas=(0.5, 0.9999))
    return optimizer_parabola


def get_z_optimizer():
    optimizer_z = torch.optim.Adam([z_x], lr=hyper_parameters['alpha2'], betas=(0.9, 0.999))
    return optimizer_z


def get_z_scheduler(optimizer):
    scheduler = MultiStepLR(optimizer, milestones=hyper_parameters['milestones'], gamma=0.01)
    return scheduler


def train(parabola, parabola_real, x_real, y_real, z, epochs=56):
    torch.manual_seed(42)
    np.random.seed(42)
    optimizer_parabola = get_parabola_optimizer()
    optimizer_z = get_z_optimizer()
    scheduler = get_z_scheduler(optimizer_z)
    for epoch in range(epochs):
        p_z = parabola(z_x)

        optimizer_parabola.zero_grad()
        optimizer_z.zero_grad()

        loss_z = F.mse_loss(p_z, z_y)
        gradients = grad(loss_z, parabola.parameters(), create_graph=True)
        a = parabola.a - hyper_parameters['alpha1'] * gradients[0]
        b = parabola.b - hyper_parameters['alpha1'] * gradients[1]
        c = parabola.c - hyper_parameters['alpha1'] * gradients[2]

        p_x = a * x_real ** 2 + b * x_real + c
        loss_x = F.mse_loss(p_x, y_real.squeeze())
        loss_x.backward()
        print(f"epoch {epoch}")
        print(f"Gradient of z {z_x.grad}")
        print(f"First Gradient of a {parabola.a.grad}")
        print(f"First Gradient of b {parabola.b.grad}")
        print(f"First Gradient of c {parabola.c.grad}")
        p_x = parabola(x_real)
        loss_x = F.mse_loss(p_x, y_real.squeeze())
        loss_x.backward()
        print(f"Second Gradient of a {parabola.a.grad}")
        print(f"Second Gradient of b {parabola.b.grad}")
        print(f"Second Gradient of c {parabola.c.grad}")

        optimizer_z.step()
        scheduler.step()
        parabola.a.data = a
        parabola.b.data = b
        parabola.c.data = c
        # optimizer_parabola.step()

        # plt.clf()
        display.clear_output(wait=True)
        visualization(parabola, parabola_real, x_real, y_real, z)
        # with torch.no_grad():
        #     display.clear_output(wait=True)
        #     x = torch.tensor(np.linspace(-5, 5))
        #     y = parabola(x)
        #     plt.plot(x.detach().numpy(), y.detach().numpy())  # parabola
        #     plt.plot(x, a_real * x ** 2 + b_real * x + c_real, c='g')
        #     plt.scatter(x_real, y_real.detach())  # points
        #     plt.scatter(z[:, 0].detach(), z[:, 1].detach())
        #     # plt.savefig(Path.cwd()/f"file{epoch}.png")
        #     plt.show()
    print(f"Result a {parabola.a.data}")
    print(f"Result b {parabola.b.data}")
    print(f"Result c {parabola.c.data}")


train(parabola, parabola_real, x_real, y_real, z, epochs=1)