import re
import os
import pickle 
import csv
from collections import defaultdict

def get_suffix(ref_file):
    ref_name = ref_file.split(".")[0]
    all_reffed = []
    for root, dirs, files in os.walk("../../data/raw/xkcd_colordata"):
        for file in files:
            if file.endswith("train") and ref_name in file and ref_file.strip() != file.strip():
                all_reffed.append(file)

    return all_reffed

def get_refs_from_file(color_path, quant_path, comp_path):
    all_refs, all_colors = [],[]
    
    color_lines = read_csv(color_path)
    quant_lines = read_csv(quant_path, delimiter=":")
    all_quants = [x[0] for x in quant_lines]
    comp_lines = read_csv(comp_path, delimiter=":")
    all_comps = [x[0] for x in comp_lines]
    all_comp_quants = all_quants + all_comps 
    # go over color lines, get all possible ref colors (last colors in space split string on left)
    for line in color_lines:
        space_str = line[0].split(" ")
        all_colors.append(space_str)
        ref_color = space_str[-1]
        all_refs.append(ref_color)
    no_dups = set(all_refs)
    counts = defaultdict(int)
    by_self = []
    # only allow colors that appear by themselves
    for potential_ref in all_refs:
        for line in color_lines:
            if line[0] == potential_ref:
                by_self.append(potential_ref)
    by_self = set(by_self)
    print("there are {} colors that appear alone".format(len(by_self)))
    # filter these by colors that also appear in something else
    with_other = []
    for potential_ref in all_refs:
        for line in color_lines:
            space_str = line[0].split(" ")
            if potential_ref in space_str and potential_ref != line[0]:
                with_other.append(potential_ref)
    with_other = set(with_other)
    print("there are {} colors that appear alone and with others".format(len(with_other)))
    filtered_by_adj = []
    # filter by whether color ever appears with any of the quants or comparatives
    for ref in with_other:
        for adj in all_comp_quants:
            to_search = re.compile("{}{}".format(adj, ref))
            if adj in all_quants:
                to_search_reverse = re.compile("{}{}".format(ref, adj))
            else:
                to_search_reverse = re.compile("NOCOLOR")
#            to_search_reverse = re.compile("{}{}".format(ref, adj))
            for color_line in color_lines:
                if to_search.match(color_line[1]) is not None or to_search_reverse.match(color_line[1]) is not None:
                #if to_search.match(color_line[1]) is not None:
                    # keep
                    filtered_by_adj.append(ref)
                    break
    filtered_by_adj = set(filtered_by_adj)
    print("there are {} after quant/comp filtering".format(len(filtered_by_adj)))
    with open("predcolor.txt","w") as f1:
        f1.write("\n".join(sorted(filtered_by_adj)))
    return filtered_by_adj
#def get_refs_from_file(color_path, quant_path, comp_path):
#    all_refs = []
#    all_colors = []
#    color_lines = read_csv(color_path)
#    quant_lines = read_csv(quant_path, delimiter=":")
#    comp_lines = read_csv(comp_path, delimiter=":")
#    all_comp_quants = [x[0] for x in quant_lines + comp_lines]
#    # go over color lines, get all possible ref colors (last colors in space split string on left)
#    for line in color_lines:
#        space_str = line[0].split(" ")
#        all_colors.append(space_str)
#        ref_color = space_str[-1]
#        all_refs.append(ref_color)
#    no_dups = set(all_refs)
#    just_good = []
#    counts = defaultdict(int)
#    # only allow colors as reference that appear in more than 1 color 
#    for ref in all_refs:
#        for color in all_colors:
#            if ref == color[-1]: 
#                counts[ref] += 1
#    for ref in no_dups:
#        if counts[ref] > 1:
#            just_good.append(ref)
#    filtered_by_adj = []
#    # filter by whether color ever appears with any of the quants or comparatives
#    for ref in no_dups:
#        for adj in all_comp_quants:
#            to_search = re.compile(".*{}.*{}.*".format(adj, ref))
#            to_search_reverse = re.compile(".*{}.*{}.*".format(ref, adj))
#            for color_line in color_lines:
#                if to_search.match(color_line[1]) is not None or to_search_reverse.match(color_line[1]) is not None:
#                    # keep
#                    filtered_by_adj.append(ref)
#                    break
#
##    return set(just_good)
#    return set(filtered_by_adj)|set(just_good)


def read_csv(path, delimiter = ","):
    with open(path) as f1:
        csvreader = csv.reader(f1, delimiter=delimiter)
        return [x for x in csvreader]


#for root, dirs, files in os.walk("/Users/Elias/Downloads/xkcd_colordata/"):
#    ref_dict = {}
#    for file in files:
#        if file.endswith("train"):
#        all_reffed = get_suffix(file)
#        if len(all_reffed) > 0:
#            ref_dict[file.strip()] = all_reffed
colors_path = "../../data/raw/words_to_labels.txt"
quant_path = "../../data/raw/quantifiers.txt"
comp_path = "../../data/raw/comparatives.txt"



file_keys = get_refs_from_file(colors_path, quant_path, comp_path)
print(len(file_keys))
with open("file_keys.txt", "w") as f1:
    f1.write("\n".join(sorted(file_keys)))
#ref_keys = set([x.split(".")[0] for x in set(ref_dict.keys())])

