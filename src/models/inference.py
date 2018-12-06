import sys; sys.path.append("..")
import pickle 
import subprocess
from collections import Counter
from typing import Tuple

import torch as th
from torch import nn
from data import DataLoader
from network import ColorNet
from data.vocab import ExtensibleVocab
import numpy as np


def angle(y_pred: Tuple[th.Tensor], y: None):
    pred,reference,beta = y_pred
    cosine_sim = nn.CosineSimilarity(dim=0)(pred, y - reference)
    return th.mean(cosine_sim, dim=0)

def get_device(use_gpu=False):
    if use_gpu:
        try:
            output = subprocess.check_output("free-gpu", shell=True)
            output=int(output.decode('utf-8'))
            gpu = output
            print("claiming gpu {}".format(gpu))
            th.cuda.set_device(int(gpu))
            a = th.tensor([1]).cuda()
            device = int(gpu)
        except (IndexError, subprocess.CalledProcessError):
            device = None
    else:
        device = None
    return device


if __name__ == '__main__':
    # load vocab from pickled dict
    str_to_ids = "../../data/embeddings/str_to_ids.pkl"
    ids_to_str  = "../../data/embeddings/ids_to_str.pkl"

    with open(str_to_ids, "rb") as f1, open(ids_to_str, "rb") as f2:
        str_to_ids = pickle.load(f1)
        ids_to_str = pickle.load(f2)

    # get the corresponding vectors, sorted by vocab index
    words = str_to_ids.keys() 
    freqs = Counter(words)
    vocab = ExtensibleVocab(freqs, vectors='fasttext.simple.300d')
    sorted_vocab_items = sorted(str_to_ids.items(), key=lambda x: x[1])
    num_embeddings, embedding_dim = len(words), list(vocab["<PAD>"].size())[0]

    # init empty array for embeddings
    print(num_embeddings, embedding_dim)
    embedding_arr = np.zeros((num_embeddings, embedding_dim))
    
    # fill the array in order
    for word, idx in sorted_vocab_items:
        corresponding_embedding = vocab[word]
        embedding_arr[idx,:] = corresponding_embedding


    # cpu for now...
    device = get_device(use_gpu=False)
    model = ColorNet(color_dim=3, vocab=str_to_ids, pretrained_embeddings=embedding_arr, beta=0.3/0.7, device=device)
    test_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test")

    count = 0.0
    cos_accum = 0.0
    for inst in test_dl:
        gold = inst[2]
        result = model(inst)
        cos_accum += angle(result, gold)
        count += 1.0
        if count % 10 == 0:
            print(count)
    print (count)
    print(cos_accum / count)



    
