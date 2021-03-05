import numpy as np
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import random


class ClusteringMNIST:
    def __init__(self, train_data):
        self.N_CLUSTERING = 60000
        self.train_data = train_data

    def plot_one(self, data, target):
        plt.imshow(data, cmap='gray')
        plt.title('%i' % target)
        plt.show()

    def plot_one_hundred(self, figures):
        # Flatten
        lin = []
        for i in range(10):
            for j in range(20):
                lin.append(figures[i][j])
        # Plot
        fig = plt.figure(figsize=(10, 9))
        columns = 20
        rows = 10
        for i in range(1, columns * rows + 1):
            img = np.reshape(lin[i - 1], (28, 28))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()

    def run_raw(self):
        # To compare K-means is applied in the images (28*28) dim, and with the encoded representation 6 dim
        cluster_data = self.train_data.data[:self.N_CLUSTERING]
        targets = self.train_data.targets[:self.N_CLUSTERING].numpy()
        clusterize = Variable(cluster_data.view(-1, 28*28).type(torch.FloatTensor)/255.).numpy()
        print(clusterize.shape)

        # Compute K-means
        kmeans = MiniBatchKMeans(n_clusters=10).fit(clusterize)
        predicted_labels = kmeans.predict(clusterize)

        # Compute confusion matrix
        predictions = np.zeros((10, 10))
        cluster_figures = [[] for i in range(10)]
        for i in range(len(targets)):
            real = int(targets[i])
            predicted = int(predicted_labels[i])
            predictions[real][predicted] += 1
            # Store 10 images of each cluster to plot
            if len(cluster_figures[predicted]) < 20:
                cluster_figures[predicted].append(cluster_data[i].numpy())
        # Plot confusion matrix
        for i in range(10):
            print(f"{i}: {predictions[i]}")
        # Plot one hundred images
        self.plot_one_hundred(cluster_figures)

    def run_encoded(self, data, targets, encoded):
        # To compare K-means is applied in the images (28*28) dim, and with the encoded representation 6 dim
        cluster_data = data[:self.N_CLUSTERING]
        clusterize = encoded[:self.N_CLUSTERING].numpy()

        # Compute K-means
        kmeans = MiniBatchKMeans(n_clusters=10).fit(clusterize)
        predicted_labels = kmeans.predict(clusterize)

        # Compute confusion matrix
        predictions = np.zeros((10, 10))
        cluster_figures = [[] for i in range(10)]
        for i in range(len(targets)):
            real = int(targets[i])
            predicted = int(predicted_labels[i])
            predictions[real][predicted] += 1
            # Store 10 images of each cluster to plot
            if len(cluster_figures[predicted]) < 20:
                cluster_figures[predicted].append(cluster_data[i].numpy())
        # Plot confusion matrix
        for i in range(10):
            print(f"{i}: {predictions[i]}")
        # Plot one hundred images
        self.plot_one_hundred(cluster_figures)

