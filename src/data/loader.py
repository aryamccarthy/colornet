from collections import defaultdict
import os 
import re
import numpy as np
import csv
from pprint import pprint
import pickle
import sys
from typing import Dict, List
from data_maker import data_maker, clean_data_map
import torch
from pathlib import Path
import time 

WRITE_DESTINATION = Path("../../data/processed")

class DataLoader(object):
    def __init__(self, raw_dir, file_dir, split, batch_size = 64):
        self.raw_dir = raw_dir
        self.quant_file = os.path.join(file_dir, "quantifiers.txt")
        self.comp_file = os.path.join(file_dir, "comparatives.txt")
        self.color_file = os.path.join(file_dir, "words_to_labels.txt")

        self.split = split
        self.batch_size = batch_size

        self.data_map, self.label2word, self.comp_dict, vocab = data_maker(self.comp_file, self.quant_file, self.color_file)
        data_map_subset = {k:v for k,v in self.data_map.items() if len(v) > 0}
        # data_map is {ref: [([comp], target}
        self.final_vocab = set(vocab)
        t0 = time.time()
        self.data = self.linearize_data(data_map_subset)
        t1 = time.time()
        print(f"linearization took {t1-t0}")

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            t0 = time.time()
            batch_data = []
            try:
                batch_tups = self.data[i:i + self.batch_size]
            except IndexError:
                batch_tups = self.data[i:]
            
            for j in range(len(batch_tups)):
                ref, comp, target = batch_tups[j]
                target_average = self.get_avgs(target, self.split)
                comp_str = self.comp_dict[comp]
                comp_list = comp_str.split(" ")
                if len(comp_list)<2:
                    comp_list = ["<PAD>", comp_list[0]]
                # pad tokens 
                batch_data.append({"reference": ref, "comparative": " ".join(comp_list), "target": target_average})
            t1 = time.time()
            yield batch_data 

    def linearize_data(self, data_map):
        # linear form: (ref, comp, target)
        linear = []
        for ref, comp_list in data_map.items():
            ref_values = self.get_values(ref, self.split)
            for (comp, target) in comp_list:
                # check if ref and target are in split
                target_path = Path(os.path.join(self.raw_dir, target + "." + self.split))
                ref_path = Path(os.path.join(self.raw_dir, ref + "." + self.split))
                # turn the comparative into a string
                comp = comp[0]
                if target_path.exists() and ref_path.exists(): 
                    for ref_val in ref_values:
                        linear.append((ref_val, comp, target))
        return linear
    
    def get_values(self, ref, split):
        try:
            with open(os.path.join(self.raw_dir, ref + "." + split), "rb") as f1:
                rgb_list = np.array(pickle.load(f1))
        except FileNotFoundError:
            return None
        return rgb_list

    def get_avgs(self, color, split):
        try:
            with open(os.path.join(self.raw_dir, color+ ".avg" + "."+split), "rb") as f1:
                full_rgb = pickle.load(f1)
        except FileNotFoundError:
            return None    
        return full_rgb 

def read_csv(path, delimiter = ","):
    with open(path) as f1:
        csvreader = csv.reader(f1, delimiter=delimiter)
        return [x for x in csvreader]


#def write(loader: DataLoader, split: str):
#    assert split in {"train", "test", "dev"}
#
#    full_batch, vocab = dl.load_split(split)
#    instances = [instance for key in full_batch for instance in full_batch[key]]
#    complete_instances = [i for i in instances if len(i["target"]) and len(i["reference"])]
#
#    for i, inst in enumerate(complete_instances):
#        inst["target"] = torch.from_numpy(inst["target"]).float()
#        inst["reference"] = torch.from_numpy(inst["reference"]).float()
#        torch.save(inst, WRITE_DESTINATION / f"{split}{i}.pt")
#
#    with open("../../data/embeddings/{}_vocab.txt".format(split), "w") as f1:
#        f1.write("\n".join(vocab))

if __name__ == "__main__":
    train_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "train")
    test_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test")
    dev_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "dev")
    total_len = 0
    t0 = time.time()
    for batch in train_dl:
        total_len += len(batch)
    t1 = time.time()
    print(f"train data has {total_len} exemplars")
    print(f"took {t1-t0} seconds")


