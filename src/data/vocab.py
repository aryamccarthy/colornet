from collections import Counter

import torch as th
import torchtext as txt
from torchtext.vocab import Vocab

class ExtensibleVocab(Vocab):
    """docstring for ExtensibleVocab"""
    def __init__(self, *args, **kwargs):
        def unk_init(tensor: th.Tensor) -> th.Tensor:
            return th.nn.init.uniform_(tensor)
            kwargs.update(unk_init=unk_init)
        super().__init__(*args, **kwargs)
        self._unk_init = unk_init

    def __getitem__(self, word: str) -> th.Tensor:
        if word in self.stoi:
            return self.vectors[self.stoi[word]]
        elif word == "<pad>":
            return th.zeros_like(th.Tensor(300))
        else:
            new_vocab = Vocab(Counter([word]))
            vector = self._unk_init(th.Tensor(300))
            new_vectors = th.cat((self.vectors, vector.unsqueeze(0)), 0)
            self.vectors = new_vectors
            self.extend(new_vocab)
            return vector

if __name__ == '__main__':
    words = "The quick brown fox jumps over the lazy dog".lower().split()
    freqs = Counter(words)
    vocab = ExtensibleVocab(freqs, vectors='fasttext.simple.300d')

    assert (vocab["the"] == vocab["the"]).all()
    assert (vocab["west"] == vocab["west"]).all()
    assert (th.sum(vocab["<pad>"]) == 0.0)
    # print(vocab["the"])
