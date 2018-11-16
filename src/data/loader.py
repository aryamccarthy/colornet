from collections import defaultdict
import os 
import re
import numpy as np
import csv
from pathlib import Path
from pprint import pprint
import pickle
import sys
from typing import Dict, List

import torch

WRITE_DESTINATION = Path("../../data/processed")

# get current estimate of subset
PRED_FILE = Path(__file__).parent / "predcolor.txt"
with open(PRED_FILE) as f1:
    SUBSET = [x.strip() for x in f1.readlines()]
#SUBSET = ["blue", "green", "yellow", "red", "gray", "orange", "purple"]
class DataLoader:
    def __init__(self, raw_dir, file_dir):
        self.raw_dir = raw_dir
        self.quant_file = os.path.join(file_dir, "quantifiers.txt")
        self.comp_file = os.path.join(file_dir, "comparatives.txt")
        self.color_file = os.path.join(file_dir, "words_to_labels.txt")
        
        self.get_refs_from_file()

    def get_refs_from_file(self):
        all_refs = []
        all_colors = []
        color_lines = read_csv(self.color_file)
        self.color_lines = color_lines
        self.quant_lines = read_csv(self.quant_file, delimiter=":")
        self.comp_lines = read_csv(self.comp_file, delimiter=":")

        self.quant_dict = {l[0]: l[1] for l in self.quant_lines}
        self.comp_dict = {l[0]: l[1] for l in self.comp_lines}

        all_comp_quants = [x[0] for x in self.quant_lines + self.comp_lines]
        # go over color lines, get all possible ref colors (last colors in space split string on left)
        for line in color_lines:
            space_str = line[0].split(" ")
            all_colors.append(space_str)
            ref_color = space_str[-1]
            all_refs.append(ref_color)
        no_dups = set(all_refs)
        just_good = []
        counts = defaultdict(int)
        # only allow colors as reference that appear in more than 1 color
        for ref in all_refs:
            for color in all_colors:
                if ref == color[-1]:
                    counts[ref] += 1
        for ref in no_dups:
            if counts[ref] > 1:
                just_good.append(ref)
        filtered_by_adj = []
        # filter by whether color ever appears with any of the quants or comparatives
        for ref in no_dups:
            for adj in all_comp_quants:
                to_search = re.compile(".*{}.*{}.*".format(adj, ref))
                for color_line in color_lines:
                    if to_search.match(color_line[1]) is not None:
                        # keep
                        filtered_by_adj.append(ref)
                        break
        #TODO: this is broken and has the wrong number of refs
        all_refs = set(filtered_by_adj)|set(just_good)
        self.refs = all_refs


    def load_full_batch(self, split="train"):
        assert split in {"train", "dev", "test"}
        # subset these for now
        refs_to_colors = defaultdict(list)
        for ref in self.refs:
            if ref in SUBSET:
                for space_str, filename in self.color_lines:
                    if ref == space_str.split(" ")[-1] and ref != filename:
                        refs_to_colors[ref].append((space_str, filename))
        color_to_avgs = self.get_avgs([x[1] for x in self.color_lines], split)
        # dict to return:
        # {ref_name: ref_dict}
        # ref_dict: {reference: avg, comparative: str, target: avg}
        final_dict = defaultdict(list)
        for ref in self.refs:
            if len(refs_to_colors[ref])>0:
                ref_avg = color_to_avgs[ref]
                for space_str, color in refs_to_colors[ref]:
                    comparative_name = self.get_comp_name(space_str)
                    if comparative_name is not None:
                        color_avg = color_to_avgs[color]
                        ref_dict = {"reference": ref_avg, "comparative": comparative_name, "target": color_avg}
                        final_dict[ref].append(ref_dict)
        return final_dict
    
    def get_comp_name(self, space_str):
        # check quant and comp for non-color part of spacestr
        not_color = space_str.split(" ")[0]
        # make quant and comp_dict
        if not_color in self.comp_dict.keys():
            adj = self.comp_dict[not_color]
        elif not_color in self.quant_dict.keys():
            adj = "more {}".format(self.quant_dict[not_color])
        else:
            return None
        return adj

    def get_pkls(self, ref_to_colors, split="train"):
        refs_to_avgs = defaultdict(list)
        for ref, color_list in ref_to_colors.items():
            for  color in color_list:
                try:
                    with open(os.path.join(self.raw_dir, color + "."+split), "rb") as f1:
                        full_rgb = np.array(pickle.load(f1))
                        refs_to_avgs[ref].append(np.mean(full_rgb, axis=0))
                except FileNotFoundError:
                    continue
        return refs_to_avgs

    def get_avgs(self, colors, split):
        color_to_avgs = defaultdict(list)
        for color in colors:
            try:
                with open(os.path.join(self.raw_dir, color+ "."+split), "rb") as f1:
                    full_rgb = np.array(pickle.load(f1))
                    color_to_avgs[color] = np.mean(full_rgb, axis=0)
            except FileNotFoundError:
                continue
        return color_to_avgs 

def read_csv(path, delimiter = ","):
    with open(path) as f1:
        csvreader = csv.reader(f1, delimiter=delimiter)
        return [x for x in csvreader]


def write(loader: DataLoader, split: str):
    assert split in {"train", "test", "dev"}

    full_batch = dl.load_full_batch(split)
    instances = [instance for key in full_batch for instance in full_batch[key]]
    complete_instances = [i for i in instances if len(i["target"]) and len(i["reference"])]

    for i, inst in enumerate(complete_instances):
        inst["target"] = torch.from_numpy(inst["target"]).float()
        inst["reference"] = torch.from_numpy(inst["reference"]).float()
        torch.save(inst, WRITE_DESTINATION / f"{split}{i}.pt")

if __name__ == "__main__":
    dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/")

    # Check that we can do things right.
    train_batch = dl.load_full_batch("train")
    for key in train_batch:
        for instance in train_batch[key]:
            pprint(instance)
    print(len(train_batch["green"]))

    # Delete files.
    for file in WRITE_DESTINATION.glob("*.pt"):
        file.unlink()  # Equivalent to `rm`

    # Write new ones.
    write(dl, "train")
    write(dl, "dev")
    write(dl, "test")
