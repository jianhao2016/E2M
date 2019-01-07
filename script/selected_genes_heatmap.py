#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script will plot the heatmap of a seq data.
in a form of a len 978 array.
"""

import numpy as np
import json
import pickle
import argparse
import os
import pdb
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# in the order of [mgmt, mlh1/kras, atm, casp8/tp53, gata6]
_BITS_IN_PROMOTER = [7, 6, 4, 3, 2]

# def test_matplot_backend():
#     import matplotlib
#     # import matplotlib.pyplot as plt
#     gui_env = ['GTKAgg', 'WXAgg', 'GTK3Cairo', 'TKAgg']
#     for gui in gui_env:
#         try:
#             print('testing backend:{}'.format(gui))
#             matplotlib.use(gui, warn=False, force=True)
#             from matplotlib import pyplot as plt
#             break
#         except:
#             continue
#     print('Using backend: {}'.format(matplotlib.get_backend()))
#     return plt


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


def load_data(p2f, mode='word_error', th=4):
    '''
    this script will load the json format data and return the test/train dict separably
    return dataset are in numpy format.
    '''
    with open(p2f, 'r') as f:
        DS = json.load(f)

    for key1 in DS.keys():
        for key2 in DS[key1].keys():
            DS[key1][key2] = np.array(DS[key1][key2])
            if key2 == 'label':
                if mode == 'word_error':
                    DS[key1][key2] = np.array(
                        [convert_array_int(x) for x in DS[key1][key2]], np.float32)
                elif mode == 'runlen_error':
                    DS[key1][key2] = np.array(
                        [longest_burst(x) for x in DS[key1][key2]], np.float32)
                elif mode == 'count_error':
                    DS[key1][key2] = np.array(
                        [sum(x) for x in DS[key1][key2]], np.float32)
                elif mode == 'threshold_error':
                    DS[key1][key2] = np.array(
                        [1.0 if sum(x) > th else 0.0 for x in DS[key1][key2]], np.float32)
                else:
                    print('mode {} doesnot exit, set all label to 0'.format(mode))
                    DS[key1][key2] = np.zeros(len(DS[key1][key2]))

    train_set = DS['train']
    test_set = DS['test']

    return train_set, test_set


def heatmap(data, ax=None, title='', x_label='', y_label='', x_ticks=[],
            x_ticklabels=[''], y_ticks=[], y_ticklabels=[''], cbar_kw={},
            cbarlabel='', **kwargs):
    '''
    create a heatmap from a numpy array 'data'
    and two lists of labels


    input:
        **kwargs: some keyword arguments to pass to later functions, 
                    like imshow etc.
    '''
    if ax == None:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize = 14)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize = 14)

    ax.set_xlabel(x_label, fontsize = 20)
    ax.set_ylabel(y_label, fontsize = 20)
    ax.set_title(title, color='r', weight='bold', fontsize = 20)
    return im


def get_label_histogram_length(gtype, error_type):
    if error_type == 'word_error':
        meth_sites = general_switch(gtype, *_BITS_IN_PROMOTER)
        hist_len = 2 ** meth_sites
    elif error_type == 'count_error' or error_type == 'runlen_error':
        hist_len = general_switch(gtype, *_BITS_IN_PROMOTER)
        # including value 0 sites
        hist_len += 1
    elif error_type == 'threshold_error':
        hist_len = 2
    return hist_len


def general_switch(gtype, mgmt_num, mlh1_num, atm_num, casp8_num, gata6_num):
    if gtype == 'mgmt':
        num_bits = mgmt_num
    elif gtype == 'mlh1' or gtype == 'kras':
        num_bits = mlh1_num
    elif gtype == 'atm':
        num_bits = atm_num
    elif gtype == 'casp8' or gtype == 'tp53':
        num_bits = casp8_num
    elif gtype == 'gata6':
        num_bits = gata6_num
    else:
        print('not valid gene {}'.format(gtype))
        num_bits = 0
    return num_bits


def get_histogram(in_list, hist_len):
    '''
    generate the histogram of input list
    '''
    label_histogram = np.zeros(hist_len, np.float32)
    for idx in in_list:
        label_histogram[int(idx)] += 1

    label_histogram = label_histogram / sum(label_histogram)
    non_zero_idx = np.nonzero(label_histogram)[0]

    return label_histogram, non_zero_idx


def get_top_k_from_array(in_array, k):
    '''
    input a numpy array, find the top k value
    return the index in a list
    '''
    # sort the list, return the index, and count form end k numbers.
    sort_k_array = in_array.argsort()[-k:]

    # inverse the index gives us the index of top k values
    inverse_index = sort_k_array[::-1]

    # by using in_array[inverse_index], we get the top k value
    return inverse_index


def process_data(DS, gtype, error_type, lm_gene_list, x_counts=None, y_counts=None,
                 mode='compact'):
    '''
    take one dataset as input, train_set/test_set
    and number of top counts in label histogram/gene_express

    return:
        x_ticks: list of integers, indicate the idx top gene with high expression
        x_ticklabels: list of string, name of top genes
        y_ticks: list of integer, shows the mid point of large portion label
        y_ticklabels: list of string, label themselves.
        seq_data: a packed dataset.
    '''
    # data and label in numpy format
    data = DS['data']
    label = DS['label']

    # begin the X axis
    if x_counts == None:
        # withou specification, print all gene
        x_ticks = list(range(0, len(lm_gene_list), 50))
        x_ticklabels = lm_gene_list[::50]
    else:
        # pick only top x_counts of genes in histogram
        max_express_gene = np.sum(data, axis=0)
        top_k_gene_idx = get_top_k_from_array(max_express_gene, x_counts)

        if mode == 'compact':
            # we wnat to plot the gene in a same order,
            # not by the value of expression.
            top_k_gene_idx = np.sort(top_k_gene_idx)
            x_ticks = range(len(top_k_gene_idx))
            data = data[:, top_k_gene_idx]
        else:
            x_ticks = top_k_gene_idx

        x_ticklabels = [lm_gene_list[idx] for idx in top_k_gene_idx]

    # ----------------------
    # now handle the Y axis
    hist_len = get_label_histogram_length(gtype, error_type)
    # label_histogram is just for taking the y count from it
    label_histogram, non_zero_count_labels = get_histogram(label, hist_len)

    if y_counts == None:
        # number of labels not specified. So plot all existing pattern.
        y_ticklabels = non_zero_count_labels
    else:
        # select the k largest portion from histogram.
        # Return the index of histogram, which is exactly the label's value
        y_ticklabels = get_top_k_from_array(label_histogram, y_counts)
        y_ticklabels = np.intersect1d(non_zero_count_labels, y_ticklabels)

    label_count_dict = {y: [] for y in y_ticklabels}

    # ----------------------
    # Pack DATA and Y ticks
    # create a dictionary, where each key is the label to be plotted,
    #    and the value is the matrix of corresponding expression data.
    # stack expression data with same label together.
    for x, y in zip(data, label):
        if y in label_count_dict:
            if np.size(label_count_dict[y]) == 0:
                # the label is currently empty. Make sure that the value is [[]]
                label_count_dict[y] = np.expand_dims(x, 0)
            else:
                label_count_dict[y] = np.vstack((label_count_dict[y], x))
    y_ticks = []

    # Re-organize the stack of expression data.
    # count the number of samples in each label, and put the pick in middle of range.
    seq_data = np.array([])
    for idx in range(len(y_ticklabels)):
        prev_sum = sum([len(label_count_dict[x]) +
                        1 for x in y_ticklabels[:idx]])

        cur_label = y_ticklabels[idx]
        new_tick = prev_sum + len(label_count_dict[cur_label]) // 2
        y_ticks.append(new_tick)

        if np.size(seq_data) == 0:
            seq_data = label_count_dict[cur_label]
        else:
            seq_data = np.vstack((seq_data, label_count_dict[cur_label]))
        # add an empty line to separate the dataset
        empty_line = np.ones(label_count_dict[cur_label].shape[1]) * np.max(data) * 0.65
        seq_data = np.vstack((seq_data, empty_line))

    seq_data = np.array(seq_data)

    # for selected y axis mode, slightly change the Y label
    if y_counts != None:
        percentage = label_histogram[(y_ticklabels,)]
        y_ticklabels = ['{}\n({:.0%})'.format(label, value) for label, value in zip(y_ticklabels, percentage)]

    return seq_data, x_ticks, x_ticklabels, y_ticks, y_ticklabels


if __name__ == '__main__':
    # plt = test_matplot_backend()
    matplotlib.use('TKAgg', warn=False, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_type', type=str, default='lung',
                        choices=['gbm', 'lgg', 'brain',
                                 'luad', 'lusc', 'lung', 'stad'],
                        help='default lung')
    parser.add_argument('--gene', type=str, default='mgmt',
                        choices=['mgmt', 'mlh1', 'atm',
                                 'gata6', 'kras', 'casp8', 'tp53'],
                        help='default mgmt')
    parser.add_argument('--error_type', type=str, default='word_error',
                        choices=['word_error', 'count_error',
                                 'runlen_error', 'threshold_error'],
                        help='default word_error')
    parser.add_argument('--quantization', type=int, default=0,
            choices=[0, 4, 8], help='0: raw data, 4/8: 4/8bit quantized. default 0')

    parser.add_argument('--x_counts', type=int, default=7,
            help='how many genes name to show on X axis, default None, shows every 50 gene')
    parser.add_argument('--y_counts', type=int, default=3,
            help='how many methylation pattern to show on Y axis. default None, show all')

    parser.add_argument('--mode_train', type=str,
            choices=['compact', 'full'], default='compact',
            help='full: show expression of all genes, compact: only show selected expression')
    parser.add_argument('--mode_test', type=str,
            choices=['compact', 'full'], default='compact',
            help='full: show expression of all genes, compact: only show selected expression')

    FLAG, _ = parser.parse_known_args()

    ctype = FLAG.cancer_type
    gtype = FLAG.gene
    error_type = FLAG.error_type
    x_counts = FLAG.x_counts
    y_counts = FLAG.y_counts
    q_bit = FLAG.quantization

    print('x_counts = {}, y_counts = {}'.format(repr(x_counts), repr(y_counts)))
    mode_train = FLAG.mode_train
    mode_test = FLAG.mode_test
    print('cancer: {}, gene: {}, error: {}\n'.format(ctype, gtype, error_type))
    data_root = '../data/'
    if ctype not in ['brain', 'lung']:
        p2data = data_root + '{}/data/selected_genes_dataset/{}_all_{}'.format(ctype.upper(), ctype, gtype)
    elif ctype == 'brain':
        p2data = data_root + 'merge_GBM_LGG/selected_genes_dataset/{}_all_{}'.format(ctype, gtype)
    elif ctype == 'lung':
        p2data = data_root + 'merge_LUAD_LUSC/selected_genes_dataset/{}_all_{}'.format(ctype, gtype)
    else:
        raise ValueError('cancer type: {} not exist'.format(ctype.upper()))

    if q_bit != 0:
        p2data += '_quantized_{}bits'.format(q_bit)
        quantized_status = '{}_bit_uniform_quantization'.format(q_bit)
    else:
        quantized_status = 'unquantized'

    p2lm_gene = '../selected_gene_symbol_list.pickle'
    with open(p2lm_gene, 'rb') as f:
        lm_gene_list = pickle.load(f)

    # load dataset into numpy array.
    th = general_switch(gtype, *_BITS_IN_PROMOTER) // 2
    train_set, test_set = load_data(p2data, mode=error_type, th=th)


    # create figure handles
    train_data_1, train_x_ticks_1, train_x_ticklabels_1, \
        train_y_ticks_1, train_y_ticklabels_1 = process_data(
            train_set, gtype, error_type, lm_gene_list, 
            x_counts=x_counts, y_counts=y_counts, mode = 'full')

    test_data_1, test_x_ticks_1, test_x_ticklabels_1, \
        test_y_ticks_1, test_y_ticklabels_1 = process_data(
            test_set, gtype, error_type, lm_gene_list, 
            x_counts=x_counts, y_counts=y_counts, mode='full')

    # pdb.set_trace()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (15, 12))

    im = heatmap(train_data_1, ax=axes[0], title='train heatmap with selected genes',
                 x_label='Selected Genes', y_label='Methylation Status',
                 x_ticks=train_x_ticks_1, x_ticklabels=train_x_ticklabels_1,
                 y_ticks=train_y_ticks_1, y_ticklabels=train_y_ticklabels_1, aspect='auto')

    im = heatmap(test_data_1, ax=axes[1], title='test heatmap with selected genes',
                 x_label='Selected Genes', y_label='Methylation Status',
                 x_ticks=test_x_ticks_1, x_ticklabels=test_x_ticklabels_1,
                 y_ticks=test_y_ticks_1, y_ticklabels=test_y_ticklabels_1, aspect='auto')

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45,
                 ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=45,
                 ha='right', rotation_mode='anchor')
    fig.subplots_adjust(hspace=0.4)
    # fig.subplots_adjust(wspace=0.4)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    fig.suptitle('cancer:{}   gene:{}   input data:{}\nY axises are shown under: {}'.format(
        ctype, gtype, quantized_status, error_type), fontsize = 28)

    save_path = '../data/heatmap/selected_genes/{2}/cancer_{0}/gene_{1}/'.format(
            ctype, gtype, error_type)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    file_name = '{0}_{1}_{2}_{3}.pdf'.format(ctype, gtype, error_type, quantized_status)

    plt.savefig(save_path + file_name)
    # plt.show()
