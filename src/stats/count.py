import numpy as np
import os
import sys; sys.path.append("..")
from data import DataLoader 
from data import data_maker , clean_data_map
#print("Preparing training data...")
train_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "train", write_vocab=False)
#print(train_dl.length)
test_dl = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "test")

counts = {x : 0 for x in ["test_seen", "test_unseen_ref", "test_unseen_comp", "test_fully_unseen", "test_unseen_pairings"]}
test_seen = []

#The reference color label, the comparative adjective and their pairing have been seen in the training data.
test_unseen_pairings = []
#The reference color label and the comparative adjective have been seen in the training data, but not their pairing
test_unseen_ref = []
# The reference color label, and thus all the corresponding RGB color datapoints, have not be seen in training, while the comparative has been seen in the training data.
test_unseen_comp = []
# The comparative adjective has not been seen in training, but the reference color label has been seen.
test_fully_unseen = []
# Neither the comparative adjective nor the reference color have been seen in the training.
def get_names(split):
    names = []
    for root, dirs, files in os.walk("../../data/raw/xkcd_colordata"):
        for file in files:
            if split in file:
                names.append(file.split(".")[0])


    return names 

test_names = get_names("test")
dev_names = get_names("dev")
train_names = get_names("train") 
data_map, label2word, comp_dict, vocab = data_maker("../../data/raw/quantifiers.txt", "../../data/raw/comparatives.txt", "../../data/raw/words_to_labels.txt")
test_data_map = clean_data_map(data_map, test_names)
train_data_map = clean_data_map(data_map, train_names)


def search_data(data, query, idx=0):
    count = 0
    for tup in data:
        word = tup[2]
        gen_word = query[1] + query[0]
        if word == gen_word:
            count+=1
    return count 


train_comps = set([x[0][0] for value in train_data_map.values() for x in value ])
test_comps = set([x[0][0] for value in test_data_map.values() for x in value ])
print(test_data_map["tan"])
print(train_comps)
#[(['dark'], 'darktan'), (['yellow'], 'yellowtan'), (['light'], 'lighttan'), (['greenish'], 'greenishtan')]
# {ref, [([comp], pairing)]}
not_in_train = set(test_data_map.keys()) - set(train_data_map.keys())
print(not_in_train)
for test_key in not_in_train:
    comps = set([x[0][0] for x in test_data_map[test_key]])
    for comp in comps:
        if comp in train_comps:
            test_unseen_ref.append(test_key)
            counts["test_unseen_ref"] += search_data(test_dl.data, (test_key, comp))
        else:
            test_fully_unseen.append(test_key)
            counts["test_fully_unseen"] += search_data(test_dl.data, (test_key, comp))

in_train = set(test_data_map.keys()) & set(train_data_map.keys())
print(in_train)
for test_key in in_train:
    comps = set([x[0][0] for x in test_data_map[test_key]])
    for comp in comps:
        if comp in train_comps: 
            # see if pairing in there
            if comp in [x[0][0] for x in train_data_map[test_key]]:
                test_seen.append(test_key)
                counts["test_seen"] += search_data(test_dl.data, (test_key, comp))
            else: 
                test_unseen_pairings.append(test_key)
                counts["test_unseen_pairings"] += search_data(test_dl.data, (test_key, comp))
        else:
            test_unseen_comp.append(test_key)
            counts["test_unseen_comp"] += search_data(test_dl.data, (test_key, comp))

print("seen", len(set(test_seen)))
print("unseen pairings", len(set(test_unseen_pairings)))
print("unseen comp", len(set(test_unseen_comp)))
print("unseen ref", len(set(test_unseen_ref)))
print("fully unseen", len(set(test_fully_unseen)))

print(counts)



