# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        h = 128 # Between 1 and 256

        # Based on description from "https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(in_size, h),
            nn.Sigmoid(),
            nn.Linear(h, out_size)
        )
    
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lrate, momentum=0.9)

    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        
        x = self.flatten(x)
        logits = self.model(x)
        return logits

        # return torch.ones(x.shape[0], 1)



    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    lrate = .01
    num_classification_categories = 4
    # TODO Standardize across both train and dev tensors
    train_set = F.normalize(train_set, 2, 1)
    dev_set = F.normalize(dev_set, 2, 1)
    model = NeuralNet(lrate, nn.CrossEntropyLoss(), 3 * 31 * 31, num_classification_categories)
    
    # Create DataLoader
    params = { 'batch_size' : batch_size, 'shuffle' : False, 'num_workers' : 4 }
    training_set = get_dataset_from_arrays(train_set, train_labels)
    training_generator = DataLoader(training_set, **params)

    dev_labels = torch.zeros(750)
    development_set = get_dataset_from_arrays(dev_set, dev_labels)
    development_generator = DataLoader(development_set, **params)
    
    # Loop through epochs to train model
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(training_generator):
            inputs, labels = data['features'], data['labels']
            losses.append(model.step(inputs, labels))

    # Attempt to classify development set
    estimated_labels = np.zeros(750)
    for i, data in enumerate(development_generator):
        inputs = data['features']
        logits = model(inputs)
        pred_probab = nn.Softmax(dim=1)(logits)
        yhat = pred_probab.argmax(1)
        estimated_labels[i] = yhat
    
    return losses, estimated_labels, model
