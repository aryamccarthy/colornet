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

WRITE_DESTINATION = Path("../../data/processed")

class DataLoader:
    def __init__(self, raw_dir, file_dir):
        self.raw_dir = raw_dir
        self.quant_file = os.path.join(file_dir, "quantifiers.txt")
        self.comp_file = os.path.join(file_dir, "comparatives.txt")
        self.color_file = os.path.join(file_dir, "words_to_labels.txt")
        
        

    def load_split(self, split="train"):
        data_map, label2word, comp_dict, vocab = data_maker(self.comp_file, self.quant_file, self.color_file)
        data_map_subset = {k:v for k,v in data_map.items() if len(v) > 0}
        # data_map is {ref: [([comp], target}
        final_vocab = set(vocab)
        final_data = defaultdict(list) 
        for ref, comp_list in data_map_subset.items():
            ref_values = self.get_values(ref, split)
            if ref_values is None:
                # if the ref color isn't in the split, keep going
                continue  
            for (comp, target) in comp_list:
                if comp is None:
                    continue
                target_average = self.get_avgs(target, split)
                if target_average is None:
                    # if target color isn't in the split, keep going
                    continue
                # turn the comparative into a string
                comp = comp[0]
                comp_str = comp_dict[comp]
                comp_list = comp_str.split(" ")
                if len(comp_list)<2:
                    comp_list = ["<PAD>", comp_list[0]]
                final_vocab |= set(comp_list)
                    # pad tokens 
                for rgb_triple in ref_values:
                    final_data[ref].append({"reference": rgb_triple, "comparative": " ".join(comp_list), "target": target_average})
        return final_data, final_vocab
    
    def get_values(self, ref, split):
        try:
            with open(os.path.join(self.raw_dir, ref + "." + split), "rb") as f1:
                rgb_list = np.array(pickle.load(f1))
        except FileNotFoundError:
            return None
        return rgb_list

    def get_avgs(self, color, split):
        try:
            with open(os.path.join(self.raw_dir, color+ "."+split), "rb") as f1:
                full_rgb = np.array(pickle.load(f1))
        except FileNotFoundError:
            return None    
        return np.mean(full_rgb, 0) 

def read_csv(path, delimiter = ","):
    with open(path) as f1:
        csvreader = csv.reader(f1, delimiter=delimiter)
        return [x for x in csvreader]


def write(loader: DataLoader, split: str):
    assert split in {"train", "test", "dev"}

    full_batch, vocab = dl.load_split(split)
    instances = [instance for key in full_batch for instance in full_batch[key]]
    complete_instances = [i for i in instances if len(i["target"]) and len(i["reference"])]

    for i, inst in enumerate(complete_instances):
        inst["target"] = torch.from_numpy(inst["target"]).float()
        inst["reference"] = torch.from_numpy(inst["reference"]).float()
        torch.save(inst, WRITE_DESTINATION / f"{split}{i}.pt")

    with open("../../data/embeddings/{}_vocab.txt".format(split), "w") as f1:
        f1.write("\n".join(vocab))

if __name__ == "__main__":
    dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/")

    # Check that we can do things right.
    train_batch, train_vocab = dl.load_split("train")
    test_batch, test_vocab = dl.load_split("test")
    dev_batch, dev_vocab = dl.load_split("dev")
    print(len(train_batch))
    agg_len = 0
    for ref, val in train_batch.items():
        agg_len+= len(val)
    print("train has {} total".format(agg_len))
    for batch, batch_name in zip((train_batch, test_batch, dev_batch), ("train", "test", "dev")):
        with open("../../data/processed/{}.pkl".format(batch_name), "wb") as f1:
            pickle.dump(batch, f1)


    #for key in train_batch:
    #    for instance in train_batch[key]:
    #        pprint(instance)

    # Delete files.
    #for file in WRITE_DESTINATION.glob("*.pt"):
    #    file.unlink()  # Equivalent to `rm`

    ## Write new ones.
    #write(dl, "train")
    #write(dl, "dev")
    #write(dl, "test")
