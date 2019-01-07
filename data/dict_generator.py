#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script generate the dictionay of landmark genes.
input:
    - landmark gene txt file
    - gene mapping file
output:
    - landmark gene dict, from name to symbol
"""

from collections import defaultdict
import argparse
import pickle
import numpy as np


def lm_generator(path_to_gene_file, pos_of_symbol):
    gene_dict = defaultdict(list)
    with open(path_to_gene_file, mode='r') as f:
        f.readline()
        for line in f:
            line_list = line.split()
            gene_symbol = line_list[pos_of_symbol]
            gene_dict[gene_symbol].append(1)
    gene_set = set(gene_dict.keys())
    return gene_dict, gene_set


def get_landmark_mapping(output_file_name, gene_set, withFake = False):
    landmark_gene_dict = defaultdict(list)
    numFake = 0
    with open('gene_name_mapping_list.txt', 'r') as f:
        for line in f:
            line_list = line.split('|')
            if line_list[5] in gene_set:
                gene_seq = line_list[1]
                gene_seq = gene_seq[:gene_seq.find('.')]
                landmark_gene_dict[line_list[5]].append(gene_seq)
            elif withFake and numFake < 1000:
                if np.random.rand() < 0.08:
                    gene_seq = line_list[1]
                    gene_seq = gene_seq[:gene_seq.find('.')]
                    landmark_gene_dict[line_list[5]].append(gene_seq)
                    numFake += 1
                    

    with open(output_file_name, 'wb') as f:
        pickle.dump(landmark_gene_dict, f)

    return landmark_gene_dict


def find_gene_idx(landmark_gene_list, tofind_gene):
    len_of_list = len(landmark_gene_list)
    if len_of_list == 0:
        raise ValueError('landmark gene list must not be empty')
    lower, upper = 0, len_of_list - 1
    while upper > lower:
        mid_idx = (lower + upper) // 2
        if landmark_gene_list[mid_idx] < tofind_gene:
            lower = mid_idx + 1
        else:
            upper = mid_idx
    if landmark_gene_list[upper] == tofind_gene:
        return upper
    else:
        # raise ValueError('{} is not a landmark gene!'.format(tofind_gene))
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-P', help='full path to the landmark gene file',
                        default='gene-space lm.txt', type=str)
    parser.add_argument('--position', help='position starts from 0',
                        default=0, type=int)
    parser.add_argument('--fake', help = 'include fake landmark or not',
                        default=0, type=int)
    FLAGS, unknown_arg = parser.parse_known_args()
    print(FLAGS)

    _, gene_set1 = lm_generator(FLAGS.path, FLAGS.position)
    with open('lm_pickle/landmark_gene_symbol_set.pickle', 'wb') as f:
        pickle.dump(gene_set1, f)
    # ---------comparison of different landmark file--------------
    # gene_dict2, gene_set2 = lm_generator(
    #     '../GSE92742_Broad_LINCS_gene_info_delta_landmark.txt', 1)
    # print('length of gene-space lm set = {}'.format(len(gene_set1)))
    # print('length of gene_info_delta_landmark = {}'.format(len(gene_set2)))

    # same_set = gene_set1 & gene_set2
    # print('num of same gene = {}'.format(len(same_set)))
    # -------------generator mapping from symbol to name/seq, ENSGxxxx -----------

    landmark_gene_dict = get_landmark_mapping(
        'lm_pickle/landmark_symbol2name_dict.pickle', gene_set1, FLAGS.fake == 1)
    print('length of landmark_gene_dict = {}'.format(
        len(landmark_gene_dict.keys())))
    tmp_list = list(landmark_gene_dict.keys())
    # sorted_list = list(gene_set1 - set(landmark_gene_dict.keys()))
    sorted_list = list(gene_set1 - set(tmp_list))
    sorted_list.sort()
    print('missing landmark = {}'.format(sorted_list))

    # ------- create a list of name of landmark genes --------------
    name_list_landmark = map(list, map(set, landmark_gene_dict.values()))
    landmark_name = []
    for name in name_list_landmark:
        # we need to make sure that each symbol corresponding to one name.
        assert len(name) == 1
        landmark_name += name
    landmark_name.sort()
    print('length of name list = {}'.format(len(landmark_name)))
    for idx in range(5):
        print(repr(landmark_name[idx]))

    with open('lm_pickle/landmark_gene_name_list.pickle', 'wb') as f:
        pickle.dump(landmark_name, f)
