"""
Code to generate comparative tuples from XKCD data
Last edited: 11.19.18
Author: Olivia Winn
"""

import re
from collections import defaultdict

""" DATA CONSTANTS """
SPLIT_STRING = '\s|-'
PAD_WORD = '<PAD>'
COMP_FILE = 'comparatives.txt'
QUANT_FILE = 'quantifiers.txt'
LABEL_FILE = 'words_to_labels.txt'


def data_maker(comp_file=COMP_FILE, quant_file=QUANT_FILE, file_list=LABEL_FILE):
    """
    Generates the comparative data

    Creates dictionary that gives all comparative tuples per reference color, dict to map filename
    versions of color terms to separate words, and dict to map adjectives to comparative form.
    The data_map can be used to determine the comparative tuples:

        data_map[<reference term>] = list((<adjective>, <target term>))
        comparative_form = comp_dict[<adjective>]

        e.g.:
            data_map['purple'] = [('light', 'lightpurple'), ('electric', 'electricpurple'), ...]
        yields
            ('purple', ['<PAD>', 'lighter'], 'lightpurple')
            ('purple', ['more', 'electric'], 'electricpurple')
            ...

    If the color data is stored in a dict, then the color datapoints are generated as

        reference_colors = data_dict[<reference term>]
        target_color = np.average(data_dict[<target term>], 0)

    :param comp_file: (str) file containing adjectives which have a comparative form
    :param quant_file: (str) file of adjectives that require 'more' or 'less' for comparison
    :param file_list: (str) file with list of color filenames separated into words
    :return: (dict, dict, dict, set)
        data_map: (dict) Keys -> labels. Values -> modified version (comp word, full comp label)
        label2word: (dict) Maps file label name (single string) to separate English words
        comp_dict: (dict) Dict to convert adjective to comparative form
        vocab: (set) set of all words in vocabulary of color terms
    """

    # Create dictionary for comparative terms
    with open(comp_file, 'r') as infile:
        comp_dict = {x.split(':')[0]: x.split(':')[1].strip() for x in infile.readlines()}
    with open(quant_file, 'r') as infile:
        comp_dict.update({x.split(':')[0]: 'more ' + x.split(':')[1].strip()
                          for x in infile.readlines()})

    # Load label/word file; create vocab list
    english_names = open(file_list, 'r').readlines()
    label2word = {}
    vocab = set()
    if '' in english_names:
        english_names.remove('')  # Remove trailing blank line

    # Create dict of file labels to words
    splitter = re.compile(SPLIT_STRING)
    for line in english_names:
        eng, name = line.split(',')
        eng, name = eng.strip(), name.strip()
        words = re.split(splitter, eng)
        if type(words) == list:
            label2word[name] = words
            [vocab.add(x) for x in words]
        else:
            label2word[name] = [words]
            vocab.add(words)

    # Add PADDING symbol to vocab
    vocab.add(PAD_WORD)
    vocab = vocab.union(set(comp_dict.keys()))

    # Create comp dict
    data_map = {}
    for label, words in label2word.items():
        base_len = len(words)
        # Get all color terms that add one word to the beginning of the base word
        superset = [([info[1][0]], info[0]) for info in label2word.items()
                    if len(info[1]) == base_len + 1
                    and info[1][-base_len:] == words]
        label_comps = []
        for comp in superset:
            # If the first word is a comparison (as opposed to nominal), save as comparative tuple
            if comp[0][0] in comp_dict:
                label_comps.append(comp)
        data_map[label] = label_comps

    # NOTE: to add 'less' to the dictionary, and the corresponding reversed-direction
    # comparisons, use the following code
    """
    # Update comparative dict
    less_comps = {'less ' + adj: 'less ' + adj for adj, comp in comp_dict.items()
                 if 'more' in comp}
    comp_dict.update(less_comps)
    # Create 'less' comparisons; i.e. opposite direction
    less_dict = defaultdict(list)
    for ref, targets in data_map:
        for tar in targets:
            less_dict[tar[1]].append(['less ' + tar[0], ref])
    # Add to original dict
    for ref in data_map.keys():
        data_map[ref].extend(less_dict[ref])
    """

    return data_map, label2word, comp_dict, vocab


def clean_data_map(orig_data_map, current_color_list):
    """
    I found it helpful to pre-adjust the datamap depending on whether I was looking at dev,
    train, or test data, rather than checking if the comparative tuple was relevant each time.
    This function prunes from the data_map the comparative tuples that contain references or
    targets not in the current data (comparatives are removed by virtue of the corresponding
    target not being in the data)
    :param orig_data_map: (dict) data_map created from create_data_map()
    :param current_color_list: (list) list of current color terms (in the filename format,
    i.e. 'lightblue')
    :return: (dict) data_map only containing relevant comparative tuples
    """
    new_map = {}
    for ref, targets in orig_data_map.items():
        if ref in current_color_list:
            new_map[ref] = [x for x in targets if x[1] in current_color_list]
    return new_map

