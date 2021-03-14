import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from IPython import display
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)


class Parabola(nn.Module):
    def __init__(self):
        super(Parabola, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.))
        self.b = nn.Parameter(torch.tensor(-3.))
        self.c = nn.Parameter(torch.tensor(2.))

    def forward(self, x):
        y = self.a * (x ** 2) + self.b * x + self.c
        return y


a_real = torch.tensor([2.])
b_real = torch.tensor([-1.])
c_real = torch.tensor([1.])
# The regressor
parabola = Parabola()
# Real Data
x_real = torch.tensor([-3., -1., 0., 1., 2.5, 3., 3.5, 5.])
y_real = a_real * x_real ** 2 + b_real * x_real + c_real
y_real = y_real  # + 2 * torch.randn([1, 5])
# Synthetic Data
z = torch.tensor([0, 30]) + torch.normal(0, 5, size=(10, 2))
z_x = z[:, 0]
z_x.requires_grad = True
z_y = z[:, 1]

# Plot the parabola
with torch.no_grad():
    x = np.linspace(-5, 5)
    y = parabola(x)
    plt.plot(x, y)
    plt.scatter(x_real, y_real.detach())
    plt.scatter(z[:, 0].detach(), z[:, 1].detach())
    plt.show()

lr = 1e-3
optimizer_parabola = torch.optim.Adam(parabola.parameters(), lr=1e-1, betas=(0.5, 0.9999))
optimizer_z = torch.optim.Adam([z_x], lr=4, betas=(0.99, 0.999))

epochs = 50
beta = 1e-4
for epoch in range(epochs):
    p_z = parabola(z_x)

    optimizer_parabola.zero_grad()
    optimizer_z.zero_grad()

    loss_z = F.mse_loss(p_z, z_y)
    gradients = grad(loss_z, parabola.parameters(), create_graph=True)
    a = parabola.a - beta * gradients[0]
    b = parabola.b - beta * gradients[1]
    c = parabola.c - beta * gradients[2]

    p_x = a*x_real**2 + b*x_real + c
    loss_x = F.mse_loss(p_x, y_real.squeeze())
    loss_x.backward()
    print("Second Step:")
    print(f"Gradient of z {z_x.grad}")
    print(f"Gradient of a {parabola.a.grad}")
    print(f"Gradient of b {parabola.b.grad}")
    print(f"Gradient of c {parabola.c.grad}")
    optimizer_z.step()
    parabola.a.data = a
    parabola.b.data = b
    parabola.c.data = c
    print(f"Gradient of a {parabola.a.grad}")
    print(f"Gradient of b {parabola.b.grad}")
    print(f"Gradient of c {parabola.c.grad}")
    # optimizer_parabola.step()

    with torch.no_grad():
        plt.clf()
        display.clear_output(wait=True)
        x = torch.tensor(np.linspace(-5, 5))
        y = parabola(x)
        plt.plot(x.detach().numpy(), y.detach().numpy())  # parabola
        plt.plot(x, a_real * x ** 2 + b_real * x + c_real, c='g')
        plt.scatter(x_real, y_real.detach())  # points
        plt.scatter(z[:, 0].detach(), z[:, 1].detach())
        # plt.savefig(Path.cwd()/f"file{epoch}.png")
        plt.show()
