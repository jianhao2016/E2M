#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script will plot the histogram of labels in decimal format.
"""

import pickle
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from data_preprocessing import generate_map, get_sample_vetor

def get_CpG_site_num(gtype):
    if gtype == 'mgmt': 
        num_sites = 7
    elif gtype == 'mlh1' or gtype == 'kras':
        num_sites = 6
    elif gtype == 'atm':
        num_sites = 4
    elif gtype == 'casp8' or gtype == 'tp53':
        num_sites = 3
    elif gtype == 'gata6':
        num_sites = 2
    else:
        raise ValueError('not valid gene type {}'.format(gtype))
        num_sites = 0
    return num_sites

def get_histogram_length(CpG_sites, error_type):
    if error_type == 'word_error':
        hist_len = 2 ** CpG_sites
    elif error_type == 'count_error' or error_type == 'runlen_error':
        hist_len = CpG_sites + 1
    elif error_type == 'threshold_error':
        hist_len = 2
    else:
        raise ValueError('error type {} dont exist'.format(error_type))
        hist_len = 0
    return hist_len

# plot histogram
def convert_array_int(arr):
    val = 0
    for idx in range(len(arr)):
        val += 2**idx * arr[idx]
    return int(val)
    
def longest_burst(arr):
    max_burst = 0
    rl_count = 0
    for val in arr:
        if val == 1:
            rl_count += 1
        elif val == 0:
            rl_count = 0
        max_burst = max(max_burst, rl_count)
    return max_burst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str,
            choices=['lgg', 'gbm', 'lusc', 'luad', 'stad', 'brain', 'lung'], help='cancer type',
            default='gbm')
    parser.add_argument('--gene', type=str,
            choices=['mgmt', 'mlh1', 'atm', 'gata6', 'casp8', 'kras', 'tp53'], help='gene type',
            default='mgmt')
    parser.add_argument('--error_type', type=str,
            choices=['word_error', 'count_error', 'runlen_error', 'threshold_error'],
            default='word_error')

    FLAG, _ = parser.parse_known_args()
    print('cancer type = {}'.format(FLAG.cancer))
    print('gene type = {}'.format(FLAG.gene))

    
    ctype = FLAG.cancer
    gtype = FLAG.gene
    if ctype not in ['brain', 'lung']:
        p2data_root = '../data/{}/data/'.format(ctype.upper())
        file_name = 'dataset/{}_all_{}'.format(ctype, gtype)
    elif ctype == 'brain':
        p2data_root = '../data/merge_GBM_LGG/'
        file_name = '{}_all_{}'.format(ctype, gtype)
    elif ctype == 'lung':
        p2data_root = '../data/merge_LUAD_LUSC/'
        file_name = '{}_all_{}'.format(ctype, gtype)


    # now reload the data and plot.
    error_type = FLAG.error_type
    CpG_sites = get_CpG_site_num(gtype)
    meth_threshold = CpG_sites // 2

    with open(p2data_root + file_name, 'r') as f:
        DS = json.load(f)
        for key1 in DS.keys():
            for key2 in DS[key1].keys():
                DS[key1][key2] = np.array(DS[key1][key2], dtype = np.float32)
                if key2 == 'label':
                    if error_type == 'word_error':
                        DS[key1][key2] = np.array(
                                [convert_array_int(x) for x in DS[key1][key2]], np.float32)
                    elif error_type == 'runlen_error':
                        DS[key1][key2] = np.array(
                                [longest_burst(x) for x in DS[key1][key2]], np.float32)
                    elif error_type == 'count_error':
                        DS[key1][key2] = np.array(
                                [sum(x) for x in DS[key1][key2]], np.float32)
                    elif error_type == 'threshold_error':
                        DS[key1][key2] = np.array(
                                [1.0 if sum(x) > meth_threshold else 0.0 for x in DS[key1][key2]], np.float32)
                    else:
                        print('error type {} dont exist, set all label to 0'.format(error_type))
                        DS[key1][key2] = np.zeros(len(DS[key1][key2]))

    
    print('train set size = {}'.format(len(DS['train']['label'])))
    print('test set size = {}'.format(len(DS['test']['label'])))
    
    # num_bins = 2 ** get_CpG_site_num(gtype)
    num_bins = get_histogram_length(CpG_sites, error_type)

    train_label_histogram = np.zeros(num_bins)
    for ll in DS['train']['label']:
        # idx = convert_array_int(ll)
        idx = int(ll)
        train_label_histogram[idx] += 1
    train_label_histogram = train_label_histogram / sum(train_label_histogram)
    train_non_zero = np.nonzero(train_label_histogram)[0]
    
    test_label_histogram = np.zeros(num_bins)
    for ll in DS['test']['label']:
        # idx = convert_array_int(ll)
        idx = int(ll)
        test_label_histogram[idx] += 1
    test_label_histogram = test_label_histogram / sum(test_label_histogram)
    test_non_zero = np.nonzero(test_label_histogram)[0]

    x_axis_union = np.union1d(train_non_zero, test_non_zero)
    index = np.arange(len(x_axis_union))

    train_label_histogram = train_label_histogram[(x_axis_union, )]
    test_label_histogram = test_label_histogram[(x_axis_union, )]

    fig, ax = plt.subplots()
    
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, train_label_histogram, bar_width, alpha = opacity, color = 'b', label = 'train_label')
    rects2 = ax.bar(index + bar_width, test_label_histogram, bar_width, alpha = opacity, color = 'y', label = 'test_label')

    ax.set_xlabel('Label in Decimal Format')
    ax.set_ylabel('Percentage')

    pic_name = 'Histogram of Methylation Status of Gene {}\nin Cancer {}, Error Type:{}'.format(gtype.upper(), ctype.upper(), error_type)
    if ctype == 'brain':
        pic_name += ' (i.e. LGG + GBM)'
    elif ctype == 'lung':
        pic_name += ' (i.e. LUAD + LUSC)'
    ax.set_title(pic_name)
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_axis_union)
    
    ax.legend()
    ax.yaxis.grid(True, linestyle=':')
    fig.tight_layout()
    
    pic_save_path = '../data/Histogram_of_labels/{}/{}_{}'.format(error_type, ctype, gtype)
    plt.savefig(pic_save_path)
    print('pic saved to {}'.format(pic_save_path))
    plt.close()
    
