import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


class AutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        # super().__init__()
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_dataset():
    # Mnist digits dataset
    data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        # this is training data
        transform=torchvision.transforms.ToTensor(),
        # Converts a PIL.Image or numpy.ndarray to

        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,
        # download it if you don't have it
    )
    return data


def plot_one(data):
    # plot one example
    print(data.data.size())     # (60000, 28, 28)
    print(data.targets.size())   # (60000)
    plt.imshow(data.data[2].numpy(), cmap='gray')
    plt.title('%i' % data.targets[2])
    plt.show()


train_data = load_dataset()
# plot_one(train_data)
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

autoencoder = AutoEncoder()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# original data (first row) for viewing
view_data = Variable(train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)


for epoch in range(EPOCH):
    count = 0
    epoch_loss = 0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))   # batch y, shape (batch, 28*28)
        b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        epoch_loss += loss
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        count += 1

        print(f'--Epoch: {epoch} | Step: {step}')

    print(f'Loss: {epoch_loss / count}')

    # plotting decoded image (second row)
    _, decoded_data = autoencoder(view_data)

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[1][i].clear()
        a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
    plt.show()
    #plt.pause(0.05)