import numpy as np
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
from linear import LinearRegression, prepare_batch
from inference import angle
def load_model(path):
    model = LinearRegression(128, 3)
    model.load_state_dict(path)
    return model

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
    test_generator = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test", device=device)
    model = load_model(args.model_path)
    batch_count = 0
    inst_count = 0
    cos_accum = 0.0
    for batch in test_generator:
        X, Y = prepare_batch(batch, device)
        Y_pred = model.forward(batch)
        cos_accum += angle(Y_pred, Y)
        batch_count+=1
        inst_count += batch_size
        if batch_count % 10 == 0:
            print('batch_count:%d, inst_count:%d, avg_cosine:%.3f' % (batch_count, inst_count, cos_accum / batch_count))

    print(cos_accum / batch_count)
