import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
import pickle 
import sys; sys.path.append("..")
import argparse
import os

from typing import Dict, List, Tuple
from tqdm import tqdm
import torch.nn as nn
import torch 

from data import DataLoader
import subprocess

# need to get a batch
# get bow for comparative
# concatenate
# fit
# repeat
# let's define in pytorch so we can use GPUs:

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def as_one_hot(words, vocab_size=125):
    #output = np.zeros((words.shape[0], vocab_size))
    output = torch.zeros((words.shape[0], vocab_size)).type(words.type())
    output[words] += 1
    return output

def prepare_batch(batch, device):
    # turn word index into one-hot
    refs, words, targets = batch
    word_oh = as_one_hot(words)
    # cast back to torch
    words_tensor = word_oh
    words_tensor = words_tensor.type(refs.type())
    print(words_tensor.type())
    # concatenate oh and refs
    x_tensor = torch.cat([words_tensor, refs], dim=1) 
    print(x_tensor.type())
    print(targets.type())
    if device is not None:
        return x_tensor.cuda(device), targets.cuda(device)
    return x_tensor, targets

def run(args):
    if args.use_gpu:
        try:
            output = subprocess.check_output("free-gpu", shell=True)
            output=int(output.decode('utf-8'))
            gpu = output
            print("claiming gpu {}".format(gpu))
            torch.cuda.set_device(int(gpu))
            a = torch.tensor([1]).cuda()
            device = int(gpu)
        except (IndexError, subprocess.CalledProcessError) as e:
            device = None
    else:
        device=None
    print(f"device is : {device}")
    train_loader = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw", "train", device=device)
    dev_generator = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "dev", device=device)
    model = LinearRegression(128, 3)
    if device is not None:
        model = model.cuda(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    for i in range(args.n_epochs):
        for batch in train_loader:
            X, Y = prepare_batch(batch, device)
            optimizer.zero_grad()
            Y_pred = model.forward(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            print(f"epoch: {i}, loss: {loss.data[0]}") 
        torch.save(model.state_dict(), "../../models/baseline/{args.lr}_linear_{i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, required=False, default=10)
    parser.add_argument("--lr", type=float, required=False, default=1e-6)
    parser.add_argument("--use-gpu", type=bool)
    args = parser.parse_args()
    run(args)


