#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script extract the landmark gene data from FPKM txt file 
the transform the extracted data to pickle file
"""

from os import listdir
from os.path import isfile, join
from dict_generator import find_gene_idx
import pickle
import pdb
import argparse
import sys
import time

def search_test(search_list, search_idx):
    """
    rtype: bool
    """
    tofind = search_list[search_idx]
    r_idx = find_gene_idx(search_list, tofind)
    return r_idx == search_idx

def method_index(search_list, search_idx):
    """
    slower than binary search.
    """
    tofind = search_list[search_idx]
    r_idx = search_list.index(tofind)
    return r_idx == search_idx

# --------------------------
def get_sample_vetor(gene_val_map, landmark_gene_list):
    len_of_vec = len(landmark_gene_list)
    val_vec = [0.0] * len_of_vec
    mis_detect = 0
    for idx in range(len_of_vec):
        cur_gene = landmark_gene_list[idx]
        if cur_gene in gene_val_map:
            val_vec[idx] = float(gene_val_map[cur_gene])
        else:
            mis_detect += 1
            print(cur_gene)
    if mis_detect != 0:
        print('missed = {}'.format(mis_detect))
    return val_vec

    
def generate_map(path_to_file):
    gene_val_map = {}
    with open(path_to_file) as f:
        for line in f:
            gene_name, val = line.split()
            dot_idx = gene_name.find('.')
            gene_name = gene_name[:dot_idx]
            if gene_name not in gene_val_map:
                gene_val_map[gene_name] = val
            else:
                raise ValueError('{} has multiple value in {}'.format(gene_name, path_to_file))
    return gene_val_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_file', '-P', help = 'path to DNA profile',
            type=str)
    parser.add_argument('--output_path', '-O', help = 'path to stored vector',
            type=str)
    parser.add_argument('--tags', '-T', help = 'tags/types of input data',
            type=str)
    FLAGS, unparsed = parser.parse_known_args()
    
    with open('../landmark_gene_name_list.pickle', 'rb') as f:
        name_list = pickle.load(f)
    
    # pdb.set_trace()
    
    # t1 = time.time()
    # for ii in range(len(name_list)):
    #     assert(search_test(name_list, ii))
    # t2 = time.time()
    # print('binary search cost = {:.6f}s'.format(t2 - t1))
    # 
    # t3 = time.time()
    # for ii in range(len(name_list)):
    #     assert(method_index(name_list, ii))
    # t4 = time.time()
    # print('method index cost = {:.6f}s'.format(t4 - t3))

    # gene_val_map = generate_map(FLAGS.path_to_file)
    all_files = [join(FLAGS.path_to_file, f) for f in listdir(FLAGS.path_to_file) if isfile(join(FLAGS.path_to_file, f))]
    # print(all_files[1])
    # gene_val_map = generate_map(all_files[1])
    
    # vec1 = get_sample_vetor(gene_val_map, name_list)
    # for i in range(5):
    #     print(name_list[i])
    #     print(vec1[i])
    vec1 = []
    for txt_file in all_files:
        cur_val_map = generate_map(txt_file)
        cur_sample = [get_sample_vetor(cur_val_map, name_list), FLAGS.tags]
        vec1.append(cur_sample)

    with open(FLAGS.output_path, 'wb') as f:
        pickle.dump(vec1, f)
    print('{} done.'.format(FLAGS.path_to_file))
