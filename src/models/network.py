import io
from typing import Dict, List

import torch as th
from torch import nn

import tqdm

COLOR_DIM = 3
EMBEDDING_DIM = 300
FC1_OUTPUT_SIZE = 30
ZERO_VECTOR = th.Tensor([0.0 for _ in range(EMBEDDING_DIM)])

def euclidean_distance(a: th.Tensor, b: th.Tensor):
    assert len(a) == len(b)
    return th.sqrt(th.sum((a - b) ** 2))

class ColorNet(nn.Module):
    """docstring for ColorNet"""
    def __init__(self, color_dim, embedding_dim, embedding_file):
        super().__init__()
        self.color_dim = color_dim
        self.embedding_dim = embedding_dim
        self.embedding_file = embedding_file

        self.fc1 = nn.Linear(EMBEDDING_DIM * 2 + COLOR_DIM, FC1_OUTPUT_SIZE)
        self.fc2 = nn.Linear(FC1_OUTPUT_SIZE + 3, COLOR_DIM)
        self.nonlinearity = nn.ReLU()

        self.loss1 = nn.CosineSimilarity(dim=0)
        self.loss2 = euclidean_distance

    def forward_for_one_item(self, instance: Dict) -> Dict:
        output = {}
        # Shape: (COLOR_DIM, )
        reference: th.Tensor = instance["reference"]
        comparative: str = instance["comparative"]
        adjective: List[str] = comparative.split()
        assert len(adjective) in {1, 2}
        if len(adjective) == 1:
            adjective.insert(0, "<PAD>")

        adjective = ["<PAD>", "<PAD>"]  # HACK!!! Ignores words completely.

        # Tensors are each of shape (EMBEDDING_DIM, )
        word_vectors: List[th.Tensor] = [self.lookup_vector(x) for x in adjective]

        # Shape: (2 * EMBEDDING_DIM + COLOR_DIM, )
        inputs = th.cat(word_vectors + [reference])
        # Shape: (FC1_OUTPUT_SIZE, )
        x = self.fc1(inputs)
        # Shape: (FC1_OUTPUT_SIZE, )
        x = self.nonlinearity(x)
        # Shape: (COLOR_DIM, )
        pred = self.fc2(th.cat([x, reference]))  # Pass the reference into the second layer, too.

        output["pred"] = pred
        target = instance["target"]
        loss1 = -self.loss1(pred, target)
        output["cosine_sim"] = -loss1
        loss2 = self.loss2(reference + pred, target)
        output["distance"] = loss2

        output["loss"] = loss1 + loss2

        return output

    def forward(self, instances: List[Dict]) -> Dict:
        outputs: List[Dict] = [self.forward_for_one_item(instance) for instance in instances]
        return outputs

    # Return shape: (EMBEDDING_DIM, )
    def lookup_vector(self, target_word: str) -> th.Tensor:
        fname = self.embedding_file
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        if target_word == "<PAD>":
            return ZERO_VECTOR
        for line in tqdm.tqdm(fin, total=999995):
            tokens = line.rstrip().split(' ')
            # data[tokens[0]] = map(float, tokens[1:])
            if tokens[0] == target_word:
                return th.Tensor(list(map(float, tokens[1:])))
        else:
            raise KeyError



if __name__ == '__main__':
    net = ColorNet(
        color_dim=3,
        embedding_dim=300,
        embedding_file="../../data/external/wiki-news-300d-1M-subword.vec"
    )
    data1 = {
        "reference": th.Tensor([80, 124, 192]),
        "comparative": "lighter",
        "target": th.Tensor([114, 191, 220])
    }
    print(net([data1]))