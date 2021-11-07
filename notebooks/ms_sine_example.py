import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################
#Define the model
####################3

# Fully connected neural network with one hidden layer
class SimpleNNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 out_size,
                 learning_rate,
                 device):
        super(SimpleNNet, self).__init__()
        self.nnet = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, out_size), nn.Tanh())
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.device = device
        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.nnet(x)
        return (out)

    def model_train(self,
                    num_epochs,
                    x_train,
                    y_train):

        # configuration
        learning_rate = self.learning_rate

        #####################################################3
        # Loss and optimizer
        criterion = nn.MSELoss()
        self.criterion = criterion
        simple_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        i=0
        for epoch in range(num_epochs):
            # Forward pass of the first phase we compute the gradients as a function of the selection
            y_predict = self.forward(x_train)
            loss = criterion(y_predict, y_train)

            # Backward and optimize
            simple_optimizer.zero_grad()
            loss.backward()
            simple_optimizer.step()

            if (i + 1) % 100 == 0:
                print('SimpleModel Epoch [{}/{}], '
                      ' Loss: {:.4f}, '
                      .format(epoch + 1,
                              num_epochs,
                              loss.item(), ))
            i+=1



    def model_test(self,x_test,y_test):

        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0



            y_predict = self.forward(x_test)
            loss = self.criterion(y_predict, y_test)

            print('Loss of SimpleNet: {} %'.format(loss))

            return loss
        tmp = 1


####################
# Fully connected neural network with one hidden layer
class SynthNNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 out_size,
                 learning_rate,
                 synth_size,
                 glr,
                 device):
        super(SynthNNet, self).__init__()
        self.nnet = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, out_size), nn.Tanh())
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.device = device
        self.learning_rate = learning_rate
        self.z_x = torch.nn.Parameter(torch.normal(0, 2.5, size=(synth_size, 1)))
        self.z_x.requires_grad=True
        self.z_x.retain_grad_grad = True
        self.z_y = torch.tensor(sin_gini(self.z_x.detach().numpy()))
        self.z_x = torch.nn.Parameter(torch.normal(0, 2.5, size=(synth_size, 1)))
        self.glr = glr

    def forward(self,
                x,
                ggradients: torch.Tensor = None):

        if self.training:
            if ggradients is not None:

                self.gc_layer1_weight = self.nnet[0].weight - self.glr * ggradients[0]
                self.gc_layer1_bias = self.nnet[0].bias - self.glr * ggradients[1]
                self.gc_layer2_weight = self.nnet[2].weight - self.glr * ggradients[2]
                self.gc_layer2_bias = self.nnet[2].bias - self.glr * ggradients[3]

                x = F.tanh(F.linear(x, self.gc_layer1_weight, self.gc_layer1_bias))
                out = F.tanh(F.linear(x, self.gc_layer2_weight, self.gc_layer2_bias))

            else:
                out = self.nnet(x)
        else:
            out = self.nnet(x)

        return out

    def model_train(self,
                    num_epochs,
                    x_train,
                    y_train):

        # configuration
        self.train()
        learning_rate = self.learning_rate

        #####################################################3
        # Loss and optimizer
        criterion = nn.MSELoss()
        self.criterion = criterion
        simple_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer_z = torch.optim.SGD(self.parameters('z_x'), lr=learning_rate)
        # Train the model
        i=0
        for epoch in range(num_epochs):
            # Forward pass of the first phase we compute the gradients as a function of the selection
            optimizer_z.zero_grad()

            #stage 1 synth pass through the presistent net
            z_y_predict = self.forward(torch.cat((self.z_x,x_train )))
            loss_z = criterion(z_y_predict, torch.cat((self.z_y,y_train)))

            #z_y_predict = self.forward(self.z_x)
            #loss_z = criterion(z_y_predict, self.z_y)

            #compute the gradients
            gradients = grad(loss_z, self.nnet.parameters(), create_graph=True)

            #stage 2 pass gradients and train data through the emulator
            y_predict = self.forward(x_train,gradients)
            loss = criterion(y_predict,y_train)

            # Backward and optimize
            loss.backward()
            optimizer_z.step()
            self.update_nnet()

            if (i + 1) % 100 == 0:
                print('SynthModel Epoch [{}/{}], '
                      ' Loss   Stage 2: {:.4f}, '
                      ' Loss_z Stage 1: {:.4f}, '
                      .format(epoch + 1,
                              num_epochs,
                              loss.item(),
                              loss_z.item()))
            i+=1



    def model_test(self,x_test,y_test):

        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0



            y_predict = self.forward(x_test)
            loss = self.criterion(y_predict, y_test)

            print('Loss of SynthNet: {} %'.format(loss))

            return loss
        tmp = 1

    def update_nnet(self):

        self.nnet[0].weight = nn.Parameter(self.gc_layer1_weight)
        self.nnet[0].bias = nn.Parameter(self.gc_layer1_bias)
        self.nnet[2].weight = nn.Parameter(self.gc_layer2_weight)
        self.nnet[2].bias = nn.Parameter(self.gc_layer2_bias)


####################
#1. gen sine model
def sin_gini(x):

    w = 2
    t = 0
    s = 1

    return s*np.sin(x*w+t)

x_all = np.arange(-5,5,0.1)
y_all = sin_gini(x_all)

plt.figure(0)
plt.plot(x_all,y_all)
plt.title('The sin samples')
plt.show()

#2. sample from model
train_size = 20
x_train = np.random.uniform(-2.5,2.5, train_size)
x_train = torch.tensor([x_train])
x_train = torch.transpose(x_train,0,1).float()
y_train = sin_gini(x_train)

valid_size = 100
x_valid = np.random.uniform(-5,5, valid_size)
x_valid = torch.tensor([x_valid])
x_valid = torch.transpose(x_valid,0,1).float()
y_valid = sin_gini(x_valid)

#3. train model
input_size = 1
hidden_size = 15
out_size = 1
learning_rate = 0.1
nepochs = 2000
SimpleModel = SimpleNNet(input_size,
                         hidden_size,
                         out_size,
                         learning_rate,
                         device)

SimpleModel.model_train(nepochs,x_train,y_train)

ValidLoss = SimpleModel.model_test(x_train,y_train)

y_train_predict = SimpleModel.forward(x_train)
plt.figure(1)
plt.plot(x_all,y_all)
plt.plot(x_train,y_train,'bo')
plt.plot(x_train,y_train_predict.detach().numpy(),'+r')
plt.title('Comparing the train data')
plt.show()

y_all_predict = SimpleModel.forward(torch.transpose(torch.tensor([x_all]),0,1).float())
plt.figure(2)
plt.plot(x_all,y_all,'b')
plt.plot(x_all,y_all_predict.detach().numpy(),'+r')
plt.title('Comparing the train data')
plt.show()

#####################################################################
#The synth model
synth_size = 3
glr = .01
nepochs = 400000
SynthModel = SynthNNet(input_size,
                         hidden_size,
                         out_size,
                         learning_rate,
                         synth_size,
                         glr,
                         device)

SynthModel.model_train(nepochs,x_train,y_train)

ValidLoss = SynthModel.model_test(x_train,y_train)

y_train_predict = SynthModel.forward(x_train)
plt.figure(10)
plt.plot(x_all,y_all)
plt.plot(x_train,y_train,'bo')
plt.plot(x_train,y_train_predict.detach().numpy(),'+r')
plt.title('Comparing the train data')
plt.show()

y_all_predict = SynthModel.forward(torch.transpose(torch.tensor([x_all]),0,1).float())
plt.figure(11)
plt.plot(x_all,y_all,'b')
plt.plot(x_all,y_all_predict.detach().numpy(),'+r')
plt.plot(SynthModel.z_x.detach().numpy(),SynthModel.z_y.detach().numpy(),'ko')
plt.plot(x_train,y_train,'bo')
plt.title('Comparing the train data')
plt.show()

tmp=1
#4. train with synth data again

#The synth model

SynthModel.glr = 10
nepochs = 20000
SynthModel.model_train(nepochs,x_train,y_train)
####################