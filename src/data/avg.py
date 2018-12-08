import os
import sys
import pickle
import numpy as np
from tqdm import tqdm 

for root, dirs, files in os.walk("../../data/raw/xkcd_colordata"):
    file_len = len(files)
    for i in tqdm(range(file_len)):
        filename = files[i]
        try:
            if ".avg" not in filename:
                color_name, split = filename.split(".")
            else:
                continue
        except ValueError:
            print(filename)
        with open(os.path.join(root, filename), "rb") as f1:
            try:
                data = np.array(pickle.load(f1))
            except:
                print(filename)
                continue
            data_mean = np.mean(data, axis=0)
        avg_name = color_name + ".avg" + "." + split
        with open(os.path.join(root, avg_name), "wb") as f1:
            pickle.dump(data_mean, f1)


