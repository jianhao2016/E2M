#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script run 2-d PCA on the landmark gene expression data.
and plot the fig with labels.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from sklearn.decomposition import PCA
from selected_genes_heatmap import load_data, get_label_histogram_length, \
        get_histogram, get_top_k_from_array, general_switch

# in the order of [mgmt, mlh1/kras, atm, casp8/tp53, gata6]
_BITS_IN_PROMOTER = [7, 6, 4, 3, 2]

def process_data(DS, gtype, error_type, num_labels=2):
    '''
    extract two numpy array from the data dict.
    select either: top num_labels of methylation patterns,
            or: use the union of the label_list and existing pattern in DS.
    output:
        - expression data: x shape (n, d)
        - methylation data: y shape (n,)
    '''
    data = DS['data']
    label = DS['label']

    hist_len = get_label_histogram_length(gtype, error_type)
    label_histogram, non_zero_count_labels = get_histogram(label, hist_len)

    try:
        if num_labels <= 0:
            raise ValueError('number of labels to plot {} should be positive.'.format(num_labels))
        else:
            pass
    except ValueError:
        num_labels = 2
        print('set number of labels to plot to 2')

    # return the top k index in the histogram, which is exactly the labels themselves.
    selected_labels = get_top_k_from_array(label_histogram, num_labels)
    selected_labels = np.intersect1d(non_zero_count_labels, selected_labels)

    data_array = []
    label_array = []

    for x, y in zip(data, label):
        if y in selected_labels:
            if np.size(data_array) == 0:
                data_array = np.expand_dims(x, 0)
                label_array = np.array([y])
            else:
                data_array = np.vstack((data_array, x))
                label_array = np.hstack((label_array, y))

    return data_array, label_array

if __name__ == '__main__':
    matplotlib.use('TKAgg', warn = False, force = True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_type', type=str, default='gbm',
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

    parser.add_argument('--num_labels', type=int, default=2,
            help='how many methylation pattern to show. default None, show all')

    FLAG, _ = parser.parse_known_args()

    ctype = FLAG.cancer_type
    gtype = FLAG.gene
    error_type = FLAG.error_type
    num_labels = FLAG.num_labels
    q_bit = FLAG.quantization

    print('cancer: {}, gene: {}, error: {}\n'.format(ctype, gtype, error_type))

    data_root = '../data/'
    if ctype not in ['brain', 'lung']:
        p2data = data_root + '{}/data/dataset/{}_all_{}'.format(ctype.upper(), ctype, gtype)
    elif ctype == 'brain':
        p2data = data_root + 'merge_GBM_LGG/{}_all_{}'.format(ctype, gtype)
    elif ctype == 'lung':
        p2data = data_root + 'merge_LUAD_LUSC/{}_all_{}'.format(ctype, gtype)
    else:
        raise ValueError('cancer type: {} not exist'.format(ctype.upper()))

    # p2data = '../data/{}_dataset/{}_all_{}'.format(ctype.upper(), ctype, gtype)

    if q_bit != 0:
        p2data += '_quantized_{}bits'.format(q_bit)
        quantized_status = '{}_bit_uniform_quantization'.format(q_bit)
    else:
        quantized_status = 'unquantized'

    # load dataset into numpy array.
    th = general_switch(gtype, *_BITS_IN_PROMOTER) // 2
    print('load data from {}'.format(p2data))
    train_set, test_set = load_data(p2data, mode=error_type, th=th)
    print('shape of train data: {}'.format(train_set['data'].shape))

    train_x, train_y = process_data(train_set, gtype, error_type, num_labels = num_labels)

    test_x, test_y = process_data(test_set, gtype, error_type, num_labels = num_labels)

    pca = PCA(n_components=2)
    transformer = pca.fit(train_set['data'])

    x_r_train = transformer.transform(train_x)
    train_target_names = np.unique(train_y)
    print(train_target_names)

    print('explained variance ration (first two components): {}'.format(pca.explained_variance_ratio_))

    # plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red']
    lw = 2

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))

    for color, i, target_name in zip(colors, train_target_names, train_target_names):
        target_name = 'type {:d}'.format(int(target_name))
        axes[0].scatter(x_r_train[train_y == i, 0], x_r_train[train_y == i, 1], alpha = 0.8, 
                color = color, lw = lw, label = target_name)
        axes[0].set_xlabel('1st Principle Component')
        axes[0].set_ylabel('2nd Principle Component')
        axes[0].set_title('Training Dataset', color='r')
    axes[0].legend(loc='best', shadow=False, scatterpoints = 1, title = 'Methylation\nPatterns')

    
    x_r_test = transformer.transform(test_x)
    test_target_names = np.unique(test_y)
    

    for color, i, target_name in zip(colors, test_target_names, test_target_names):
        target_name = 'type {:d}'.format(int(target_name))
        axes[1].scatter(x_r_test[test_y == i, 0], x_r_test[test_y == i, 1], alpha = 0.8, 
                color = color, lw = lw, label = target_name)
        axes[1].set_xlabel('1st Principle Component')
        axes[1].set_ylabel('2nd Principle Component')
        axes[1].set_title('Test Dataset', color='r')
    axes[1].legend(loc='best', shadow=False, scatterpoints = 1, title = 'Methylation\nPatterns')

    fig.subplots_adjust(top=0.8)
    fig.subplots_adjust(wspace=0.4)
    fig_title = 'PCA visualization of\nCancer:{}  Gene:{}  Error Type:{}  Input:{}'.format(ctype.upper(), gtype.upper(), error_type, quantized_status)
    fig.suptitle(fig_title)

    # save_path = '../data/PCA_visualization/{2}/cancer_{0}/gene_{1}/'.format(
    #         ctype, gtype, error_type)
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)

    # file_name = '{0}_{1}_{2}_{3}.pdf'.format(ctype, gtype, error_type, quantized_status)

    # fig.savefig(save_path + file_name)

    save_path = '../data/PCA_visualization/{2}/cancer_{0}/gene_{1}/'.format(
            ctype, gtype, error_type)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    file_name = '{0}_{1}_{2}_{3}.pdf'.format(ctype, gtype, error_type, quantized_status)

    plt.savefig(save_path + file_name)
