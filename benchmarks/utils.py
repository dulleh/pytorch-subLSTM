import numpy as np
import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt

from subLSTM.basic.nn import SubLSTMCell, SubLSTMCudaCell, SubLSTM

def detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(detach_hidden_state(h) for h in hidden_state)
    return None

def drawepochs(epochs, epochsb, model_name):
    fig, axs = plt.subplots(len(epochs), 2, constrained_layout=True)
    fig.suptitle('Times Across Epochs for {}'.format(model_name))
    for i, epochtimes in enumerate(epochs):
        axs[i][0].bar(np.linspace(1, len(epochtimes)+1, len(epochtimes)), epochtimes)
        axs[i][0].set(xlabel='total forward() time for epoch {} was {}s'.format(i, sum(epochtimes)), ylabel='Time (s)')
    for j, epochtimes in enumerate(epochsb):
        axs[j][1].bar(np.linspace(1, len(epochtimes)+1, len(epochtimes)), epochtimes)
        axs[j][1].set(xlabel='total backward() time for epoch {} was {}s'.format(i, sum(epochtimes)), ylabel='Time (s)')
    plt.show()

def train(model, data_loader, criterion, optimizer, grad_clip,
        track_hidden, log_interval, device, verbose):
    """
    Train the model for one epoch over the whole dataset.
    """
    model.train(True)
    loss_trace, running_loss = [], 0.0
    n_batches = len(data_loader)

    # Keep track or the hidden state over the whole epoch. This allows faster training?
    hidden = None

    model.rnn.times = []
    model.rnn.seqtimes = []
    model.rnn.backwardtimes = []

    for i, data in enumerate(data_loader):
        # Load one batch into the device being used.
        inputs, labels = data

        # The expectation is that this should be the case anyway and if make sure it is at intialisation.
        #inputs, labels = inputs.to(device), labels.to(device)

        # Set all gradients to zero.
        optimizer.zero_grad()

        # If reusing hidden states, detach them from the computation graph
        # of the previous batch. Using the previous value may speed up training
        # but detaching is needed to avoid backprogating to the start of training.
        # hidden = detach_hidden_state(hidden) if track_hidden else None

        # Forward and backward steps
        outputs, hidden = model(inputs)

        loss = criterion(outputs, labels)

        backwardstart = time.time()
        loss.backward()
        model.rnn.backwardtimes.append(time.time() - backwardstart)

        """
        if (i == 2) and verbose:
            for module in model.rnn.children():
                if isinstance(module, SubLSTMCell):
                    print("weights.grad: ")
                    print(torch.cat((module.recurrent_layer.weight.grad,  module.input_layer.weight.grad), 1))
                    # bias is significantly diff for i = 0
                    print("bias.grad")
                    #print(module.input_layer.bias.grad + module.recurrent_layer.bias.grad)
                    print(module.input_layer.bias.grad)
                elif isinstance(module, SubLSTMCudaCell):
                    print("weights.grad: ")
                    print(module.weights.grad)
                    print("bias.grad")
                    print(module.bias.grad)
        """

        # Clipping (helps with exploding gradients) and then gradient descent
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

        # Print the loss every log-interval mini-batches and save it to the trace
        if i % log_interval == log_interval - 1:
            if verbose:
                print('\t[batches %5d / %5d] loss: %.5f' %
                    (i + 1, n_batches, running_loss / log_interval))

            loss_trace.append(running_loss / log_interval)
            running_loss = 0.0

    """
    for module in model.rnn.children():
        if isinstance(module, SubLSTMCell):
            drawbargraph(forwardtimes, forwardseqtimes, 'SubLSTM (python)')
        else:
            drawbargraph(forwardtimes, forwardseqtimes, 'SubLSTM (cuda forward, c++ backward)')
        break
    """

    print("total forward time: ", sum(model.rnn.times))
    print("total backward time: ", sum(model.rnn.backwardtimes))
    model.rnn.epochtimes.append(model.rnn.times)
    model.rnn.epochbackwardtimes.append(model.rnn.backwardtimes)

    return loss_trace


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            total_loss += criterion(model(inputs)[0], labels)

    return total_loss / (i + 1)


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = model(inputs)[0]
            predictions = torch.argmax(outputs, dim=1)

            total += predictions.size(0)
            correct += (labels == predictions).sum().item()

    return correct / total
