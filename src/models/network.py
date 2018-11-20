import io
from typing import Dict, List

import numpy as np
import torch as th
from torch import nn
from torchtext.vocab import Vocab, Vectors

import tqdm

COLOR_DIM = 3
FC1_OUTPUT_SIZE = 30

def euclidean_distance(a: th.Tensor, b: th.Tensor):
    assert len(a) == len(b)
    return th.sqrt(th.sum((a - b) ** 2))

class ColorNet(nn.Module):
    """docstring for ColorNet"""
    def __init__(self,
            color_dim,
            vocab: Vocab,
            beta: float=0.01) -> None:
        super().__init__()
        self.color_dim = color_dim

        self.vocab = vocab
        self.embedding_dim = len(vocab["<PAD>"])

        self.fc1 = nn.Linear(self.embedding_dim * 2 + COLOR_DIM, FC1_OUTPUT_SIZE)
        self.fc2 = nn.Linear(FC1_OUTPUT_SIZE + 3, COLOR_DIM)
        self.nonlinearity = nn.ReLU()

        self.metric1 = nn.CosineSimilarity(dim=0)
        self.metric2 = euclidean_distance
        self.beta = beta

    def forward_for_one_item(self, instance: Dict) -> Dict:
        output = {}
        # Shape: (COLOR_DIM, )
        reference: th.Tensor = instance["reference"]
        comparative: str = instance["comparative"]
        adjective: List[str] = comparative.split()
        # Output is of shape (2 * EMBEDDING_DIM, )
        word_vectors: th.Tensor = self.lookup_vector(adjective)

        # Shape: (2 * EMBEDDING_DIM + COLOR_DIM, )
        inputs = th.cat([word_vectors, reference])
        # Shape: (FC1_OUTPUT_SIZE, )
        x = self.fc1(inputs)
        # Shape: (FC1_OUTPUT_SIZE, )
        x = self.nonlinearity(x)
        # Shape: (COLOR_DIM, )
        pred = self.fc2(th.cat([x, reference]))  # Pass the reference into the second layer, too.

        output["pred"] = pred
        target = instance["target"]
        output["cosine_sim"] = self.metric1(pred, target - reference)
        output["distance"] = self.metric2(reference + pred, target)
        loss1, loss2 = -output["cosine_sim"], output["distance"]

        output["loss"] = loss1 + self.beta * loss2

        return output

    def forward(self, instances: List[Dict]) -> Dict:
        outputs: List[Dict] = [self.forward_for_one_item(instance) for instance in instances]
        return outputs

    # Return shape: (EMBEDDING_DIM, )
    def lookup_vector(self, target_words: List[str]) -> th.Tensor:
        assert len(target_words) in {1, 2}
        if len(target_words) == 1:
            target_words.insert(0, "<pad>")
        assert len(target_words) == 2
        # Tensors are of shape (EMBEDDING_DIM, )
        word_vectors = [self.vocab[word] for word in target_words]
        combined = th.cat(word_vectors)
        return combined


if __name__ == '__main__':
    import pickle
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
