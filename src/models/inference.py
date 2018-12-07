import argparse
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

def get_device(use_gpu):
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

    if device is not None:
        device = th.device(device)
    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action='store_true')
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    use_gpu = args.use_gpu
    model_path = args.model_path


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

    device = get_device(use_gpu)

    # load the model
    batch_size = 2048
    model = ColorNet(color_dim=3, vocab=str_to_ids, pretrained_embeddings=embedding_arr, beta=0.3/0.7, device=device)
    model.load_state_dict(th.load(model_path))
    model.eval()
    model = model.cuda(device)

    # get the data loader
    test_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test", batch_size=batch_size, device=device)

    # import pdb;pdb.set_trace()
    batch_count = 0
    inst_count = 0
    cos_accum = 0.0
    for inst in test_dl:

        # the dataloader should already do this...?
        # but it complains if we don't do this
        inst = [_.cuda(device) for _ in inst]

        gold = inst[2]
        result = model(inst)
        cos_accum += angle(result, gold)

        batch_count += 1
        inst_count += batch_size
        if batch_count % 10 == 0:
            print('batch_count:%d, inst_count:%d, avg_cosine:%.3f' % (batch_count, inst_count, cos_accum / batch_count))

    print(cos_accum / batch_count)



    
