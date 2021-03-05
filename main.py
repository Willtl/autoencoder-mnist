import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import kmeans

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = False
N_TEST_IMG = 10


class AutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        # super().__init__()
        super(AutoEncoder, self).__init__()

        n = 28 * 28     # 784
        # When using Sequential model you must use Variables
        self.encoder = nn.Sequential(
            nn.Linear(n, n // 2),
            nn.ReLU(),
            nn.Linear(n // 2, n // 4),
            nn.ReLU(),
            nn.Linear(n // 4, n // 8),
            nn.ReLU(),
            nn.Linear(n // 8, n // 16),   # dim. red. to 49
        )

        self.decoder = nn.Sequential(
            nn.Linear(n // 16, n // 8),
            nn.ReLU(),
            nn.Linear(n // 8, n // 4),
            nn.ReLU(),
            nn.Linear(n // 4, n // 2),
            nn.ReLU(),
            nn.Linear(n // 2, n),
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
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    return data


def plot_one(data, target):
    # plot one example
    # print(data.data.size())     # (60000, 28, 28)
    # print(data.targets.size())   # (60000)
    # plt.imshow(data.numpy(), cmap='gray')
    plt.imshow(data, cmap='gray')
    plt.title('%i' % target)
    plt.show()


# Define GPU and CPU devices
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")

# Load dataset
train_data = load_dataset()

# Define model
autoencoder = AutoEncoder()
# Move it to the GPU
autoencoder.to(device)
# Define optimizer after moving to the GPU
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
criterion = nn.MSELoss().to(device)

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Move data to GPU
batch_x = []
batch_y = []
for x, y in train_loader:
    batch_x.append(Variable(x.view(-1, 28 * 28)).to(device))
    batch_y.append(Variable(x.view(-1, 28 * 28)).to(device))

gpu_data = zip(batch_x, batch_y)

# Training
for epoch in range(EPOCH):
    # To calculate mean loss over this epoch
    epoch_loss = []
    start = time.time()

    # Loop through batches
    for b_x, b_y in zip(batch_x, batch_y):
        encoded, decoded = autoencoder(b_x)

        loss = criterion(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        epoch_loss.append(loss.to(cpu).item())      # used to calculate the epoch mean loss
    print(f'Epoch {epoch}, mean loss: {np.mean(np.array(epoch_loss))}, time: {time.time() - start}')

# First N_TEST_IMG images for visualization
view_data = Variable(train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)

# Testing - Plotting decoded image
with torch.no_grad():
    # Set the model to evaluation mode
    autoencoder = autoencoder.eval()

    # Encode and decode view_data to visualize the outcome
    _, decoded_data = autoencoder(view_data.to(device))
    decoded_data = decoded_data.to(cpu)

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

    # Set the model to train mode
    autoencoder = autoencoder.train()

# # Compute K-Means for the raw images
means = kmeans.ClusteringMNIST(train_data)
means.run_raw()

# Compute K-Means for the encoded representations
with torch.no_grad():
    # Encode entire dataset
    train_loader = Data.DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=True, pin_memory=True)
    for x, y in train_loader:
        b_x = Variable(x.view(-1, 28 * 28)).to(device)
        encoded, _ = autoencoder(b_x)
        encoded = encoded.to(cpu)
    # Run K-means on encoded representations
    means.run_encoded(x, y, encoded)