import io
from typing import Dict, List, Tuple

import numpy as np
import torch as th
from torch import nn
from torchtext.vocab import Vocab, Vectors

import pickle
import tqdm

COLOR_DIM = 3
FC1_OUTPUT_SIZE = 30


class ColorNet(nn.Module):
    """docstring for ColorNet"""
    def __init__(self,
            color_dim,
            vocab: dict,
            pretrained_embeddings: np.array = None,
            beta: float=0.01,
            trainable_embeddings: bool = False,
            n_layers: int = 1,
            device: int =None) -> None:
        super().__init__()
    
        self.color_dim = color_dim
        self.vocab = vocab
        self.device = device
        self.n_layers = n_layers
        print(f"n_layers: {n_layers}")
        # embedding layer
        try:
            assert(pretrained_embeddings.shape[0] == len(self.vocab))
        except AssertionError:
            #raise AssertionError(f"trained embeddings and vocab do not match: {pretrained_embeddings.shape[0]}, {len(self.vocab}")
            raise AssertionError("trained embeddings and vocab do not match: {}, {}".format(pretrained_embeddings.shape[0], len(self.vocab)))
            
        #self.embedding_dim = len(vocab["<PAD>"])
        self.embedding_dim = pretrained_embeddings.shape[1]
        
        self.embedding = nn.Embedding(len(self.vocab), self.embedding_dim)
        # load in pre-trained weights
        self.embedding.weight.data.copy_(th.from_numpy(pretrained_embeddings)) 
        if not trainable_embeddings:
           self.embedding.requires_grad = False 
        self.fc1 = nn.Linear(self.embedding_dim * 2 + COLOR_DIM, FC1_OUTPUT_SIZE)
        self.fc2 = nn.Linear(FC1_OUTPUT_SIZE + 3, COLOR_DIM)
        self.linear_layers = nn.ModuleList([nn.Linear(FC1_OUTPUT_SIZE, FC1_OUTPUT_SIZE) for i in range(n_layers-1)])
        self.nonlinearity = nn.ReLU()

        self.beta = beta
    
    def forward(self, instance: Tuple) -> Dict:
        output = {}
        reference = instance[0]
        comparative_as_ints = instance[1]
        if self.device is not None:
            comparative_as_ints = comparative_as_ints.cuda(self.device)
            reference = reference.cuda(self.device)
        comparative_as_embedding = self.embedding(comparative_as_ints)
        comparative_as_embedding = comparative_as_embedding.reshape((-1, comparative_as_embedding.size()[1] * comparative_as_embedding.size()[2]))
        
        
        inputs = th.cat([comparative_as_embedding, reference], dim=1)
        
        x = self.fc1(inputs)
        x = self.nonlinearity(x)
        for layer in self.linear_layers:
            x = layer(x)
            x = self.nonlinearity(x)
        pred = self.fc2(th.cat([x, reference], dim=1))

        return pred, reference, self.beta

if __name__ == '__main__':
    with open("../../data/embeddings/subset-wb.p", 'rb') as f:
        embeddings = pickle.load(f)
    vocab = {x: th.FloatTensor(y) for x, y in embeddings.items()}
    vocab["<pad>"] = vocab["<PAD>"]
    net = ColorNet(
        color_dim=3,
        vocab=vocab
    )
    data1 = {
        "reference": th.Tensor([80, 124, 192]),
        "comparative": "lighter",
        "target": th.Tensor([114, 191, 220])
    }
    print(net([data1]))
