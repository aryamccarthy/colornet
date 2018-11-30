from collections import defaultdict
import os 
import re
import numpy as np
import csv
from pprint import pprint
import pickle
import sys
from typing import Dict, List
from .data_maker import data_maker, clean_data_map
import torch
import tqdm
from pathlib import Path
import time

class Timer:    
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


WRITE_DESTINATION = Path("../../data/processed")

class DataLoader(object):
    def __init__(self, raw_dir, file_dir, split, batch_size = 256, write_vocab = False):
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
        with Timer() as t:
            self.data = self.linearize_data(data_map_subset, split)
        # one-time thing: 
        if write_vocab:
            # get all the data
            all_data = []
            for split in ["train", "test", "dev"]:
                split_data = self.linearize_data(data_map_subset, split)
                all_data.extend(split_data)

            str_to_ids = {}
            ids_to_str = {}
            vocab = set()
            for data_tup in all_data:
                comp_str = data_tup[1]
                comp_words = comp_str.split(" ")
                vocab |= set(comp_words)
            vocab |= {"<PAD>", "<pad>", "more", "less"}
            # add the comparatives
            with open(self.comp_file) as f1, open(self.quant_file) as f2:
                comp_lines = f1.readlines()
                quant_lines = f2.readlines()
            for line in comp_lines + quant_lines:
                w1, w2 = [x.strip() for x in line.split(":")]
                vocab |= set([w1, w2])
      
            sorted_vocab = sorted(list(vocab))
            for i, word in enumerate(sorted_vocab):
                str_to_ids[word] = i
                ids_to_str[i] = word

            with open("../../data/embeddings/str_to_ids.pkl", "wb") as f1, open("../../data/embeddings/ids_to_str.pkl", "wb") as f2:
                pickle.dump(str_to_ids, f1)
                pickle.dump(ids_to_str, f2)

            self.str_to_ids = str_to_ids
            self.ids_to_str = ids_to_str
        else:
            with open("../../data/embeddings/str_to_ids.pkl", "rb") as f1, open("../../data/embeddings/ids_to_str.pkl", "rb") as f2:
                self.str_to_ids = pickle.load(f1)
                self.ids_to_str = pickle.load(f2)
        self.length = int(len(self.data)/batch_size) + 1
            
        print(f"linearization took {t.interval}")
    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_data = []
            ref_data = []
            comp_data = []
            target_data = []
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
                ref_data.append(ref)
                target_data.append(target_average)
                # get comp ints 
                comp_str_as_ints = self.convert_to_ids(comp_list)
                comp_data.append(comp_str_as_ints)

                #batch_data.append({"reference": torch.FloatTensor(ref), "comparative": " ".join(comp_list), "target": torch.FloatTensor(target_average)})
            # convert to tensor arrays
            as_tensor = (torch.FloatTensor(ref_data), torch.LongTensor(comp_data), torch.FloatTensor(target_data))
            yield as_tensor

    def __len__(self):
        return len(self.data) // self.batch_size

    def convert_to_ids(self, comp_list):
        return [self.str_to_ids[x] for x in comp_list] 

    def linearize_data(self, data_map, split):
        # linear form: (ref, comp, target)
        linear = []
        for ref, comp_list in data_map.items():
            ref_values = self.get_values(ref, split)
            for (comp, target) in comp_list:
                # check if ref and target are in split
                target_path = Path(os.path.join(self.raw_dir, target + "." + split))
                ref_path = Path(os.path.join(self.raw_dir, ref + "." + split))
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



if __name__ == "__main__":
    print("Preparing training data...")
    train_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "train", write_vocab=True)
    print(train_dl.length)
    test_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test")
    print(test_dl.length)
    dev_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "dev")
    print(dev_dl.length)
    total_len = 0
    
    with Timer() as t:
        total_len = sum(len(batch) for batch in tqdm.tqdm(train_dl, total=len(train_dl)))

    print(f"train data has {total_len} exemplars")
    print(f"took {t.interval} seconds")


