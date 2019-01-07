#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script will take the raw data from:
    "meth_<GENE>/" ==> thresholded and turn into label
    and 
    "seq/" ==> using the landmark gene mapping, and list, convert into [g1, g2, ... , g978]
inputs:
    landmark gene list
    landmark gene mapping
    raw data in folder meth_MGMT/ and seq/
output:
    dump a dict of {'train': {'data':[[]], 'label':[]},
                    'test': {'data':[[]], 'label':[]}}
"""

import pickle
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from data_preprocessing import generate_map, get_sample_vetor

def get_seq_data(p2f):
    cur_val_map = generate_map(p2f)
    cur_sample = get_sample_vetor(cur_val_map, name_list)
    # return np.array(cur_sample, dtype=np.float32)
    return cur_sample

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
        print('not valid gene type')
        num_sites = 0
    return num_sites

def get_meth_data(p2f, gtype, threshold):
    '''
    thresholded the methylation data to binary vector.
    gtype: gene type, different genes has different length of promoter.
    p2f: path to methylation file.
    '''
    num_of_sites = get_CpG_site_num(gtype)
    cur_target = np.zeros(num_of_sites, dtype = np.float32)

    line_counter = 0
    with open(p2f, 'r') as f:
        for line in f:
            line_counter += 1
            if line_counter > len(cur_target):
                print('file ', p2f, ' has more sites than expected')
                break
            line = line.strip()
            pos, val = line.split()
            if pos not in promotor_idx:
                # not a valid line. print error and exit
                line_counter -= 1
                print('{} is not in the promoter'.format(pos))
                exit(1)
            if val == 'NA':
                val = 0
            else:
                val = float(val)
                if val > threshold:
                    val = 1
                else:
                    val = 0
            idx = promotor_idx[pos]
            cur_target[idx] = val
    return cur_target.tolist()

def get_data_set(cancer_type, gtype, p2data_root, file_name):
    '''
    create the dataset dict for certain cancer, certain genes.
    and dump as a json file.
    if cancer_type is none, do nothing and return.
        cancer_type: one of the five cancers.
        gtype: gene type, mgmt/mlh1/atm.
    '''
    if cancer_type == 'lgg':
        p2mapping = 'LGG/mapping_caseID_seqID_methID_LGG'
    elif cancer_type == 'gbm':
        p2mapping = 'GBM/mapping_caseID_seqID_methID_GBM'
    elif cancer_type == 'luad':
        p2mapping = 'LUAD/mapping_caseID_seqID_methID_LUAD'
    elif cancer_type == 'lusc':
        p2mapping = 'LUSC/mapping_caseID_seqID_methID_LUSC'
    elif cancer_type == 'stad':
        p2mapping = 'STAD/mapping_caseID_seqID_methID_STAD'
    else:
        print('not valid cancer type.')
        return None
    
    with open(p2mapping, 'r') as f:
        mapping = json.load(f)
    
    case_count = 0
    dataset = {'train':{
                    'data':[],
                    'label':[]
                    },
                'test':{
                    'data':[],
                    'label':[]
                    }
            }

    np.random.seed(FLAG.rnd)

    suffice_path_meth = 'meth_{}/'.format(gtype.upper())

    # balance_dict keep and counting of each label, 
    # each key will be the decimal represent of the label
    # i.e. {'label_1':{'train_count': #1, 'test_count':#2}, 
    #       'label_2':{'train_count': #3, 'test_count':#2},... }
    balance_dict = {}

    for cases in mapping.keys():
        case_count += 1
        if 'meth_uuid' in mapping[cases].keys() and 'seq_uuid' in mapping[cases].keys():
            meth_id = mapping[cases]['meth_uuid']
            seq_id = mapping[cases]['seq_uuid']
        else:
            print('cases number {} has missing uuids'.format(cases))
            exit(1)

        # p2meth = 'meth/' + meth_id
        p2meth = suffice_path_meth + meth_id
        p2seq = 'seq/' + seq_id
        p2f = [p2meth, p2seq]

        full_path = [join(p2data_root, f) for f in p2f]
        pair_of_files = [join(fp, f) for fp in full_path for f in listdir(fp) 
                if isfile(join(fp, f))]
        # pair_of_files = [join(fp, listdir(fp)) for fp in full_path]
        assert (len(pair_of_files) == 2)

        seq_data = get_seq_data(pair_of_files[1])
        meth_data = get_meth_data(pair_of_files[0], gtype, FLAG.th)

        dec_label = convert_array_int(meth_data)
        if dec_label not in balance_dict.keys():
            # initialize the count.
            balance_dict[dec_label] = {'train_count':0, 'test_count':0}

        cur_train_count = balance_dict[dec_label]['train_count']
        cur_test_count = balance_dict[dec_label]['test_count']

        pi = np.random.uniform(0, 1)
        if pi < 0.2:
            # when we want to put a sample to test set, 
            # make sure that we have enough sample in train set
            if cur_train_count > 4 * cur_test_count:
                name_pick = 'test'
            else:
                # now that the train count is not much larger than test count
                # we decided to put into train set.
                name_pick = 'train'
        else:
            # when we want to put a sample into train set, 
            # we want to check if the test set is not too small.
            if 4 * cur_test_count >= cur_train_count:
                name_pick = 'train'
            else:
                # now that the test set is too small, we want to compensate this.
                name_pick = 'test'
        balance_dict[dec_label]['{}_count'.format(name_pick)] += 1

        dataset[name_pick]['data'].append(seq_data)
        dataset[name_pick]['label'].append(meth_data)

    
    # print(balance_dict)
    with open(p2data_root + file_name, 'w') as f:
        json.dump(dataset, f)
    # return dataset

# plot histogram
def convert_array_int(arr):
    val = 0
    for idx in range(len(arr)):
        val += 2**idx * arr[idx]
    return int(val)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--th', type=float,
            default=0.1, help='threshold for methlaytion readings')
    parser.add_argument('--cancer', type=str,
            choices=['lgg', 'gbm', 'lusc', 'luad', 'stad'], help='cancer type',
            default='gbm')
    parser.add_argument('--rnd', type=int,
            default=42, help='random seed in data generation, default 42')
    parser.add_argument('--gene', type=str,
            choices=['mgmt', 'mlh1', 'atm', 'gata6', 'casp8', 'kras', 'tp53'], help='gene type',
            default='mgmt')

    FLAG, _ = parser.parse_known_args()
    print('threshold = ', FLAG.th)
    print('cancer type = {}'.format(FLAG.cancer))
    print('gene type = {}'.format(FLAG.gene))

    with open('lm_pickle/landmark_gene_name_list.pickle', 'rb') as f:
        name_list = pickle.load(f)
    
    promotor_idx = {}
    gtype = FLAG.gene
    p2promotor_pos = 'promoter_pos/{}_promotor_pos'.format(gtype.upper())

    with open(p2promotor_pos, 'r') as f:
        idx = 0
        for line in f:
            line = line.strip()
            promotor_idx[line] = idx
            idx += 1
    print(promotor_idx)
    
    if FLAG.cancer == 'lgg':
        p2data_root = 'LGG/data/'
        file_name = 'dataset/lgg_all_' + gtype
    elif FLAG.cancer == 'gbm':
        p2data_root = 'GBM/data/'
        file_name = 'dataset/gbm_all_' + gtype
    elif FLAG.cancer == 'luad':
        p2data_root = 'LUAD/data/'
        file_name = 'dataset/luad_all_' + gtype
    elif FLAG.cancer == 'lusc':
        p2data_root = 'LUSC/data/'
        file_name = 'dataset/lusc_all_' + gtype
    elif FLAG.cancer == 'stad':
        p2data_root = 'STAD/data/'
        file_name = 'dataset/stad_all_' + gtype
    else:
        print('plotting lgg histogram only')
        p2data_root = 'LGG/data/'
        file_name = 'dataset/lgg_all'
    
    # this function will dump the dict to file.
    get_data_set(FLAG.cancer, gtype, p2data_root, file_name)
    print('#### -> done dumping {}\n'.format(file_name))


    # now reload the data and plot.
    # with open(p2data_root + file_name, 'r') as f:
    #     DS = json.load(f)
    #     for key1 in DS.keys():
    #         for key2 in DS[key1].keys():
    #             DS[key1][key2] = np.array(DS[key1][key2], dtype = np.float32)
    # 
    # print('train set size = {}'.format(len(DS['train']['label'])))
    # print('test set size = {}'.format(len(DS['test']['label'])))
    # 
    # num_bins = 2 ** get_CpG_site_num(gtype)
    # index = np.arange(num_bins)

    # histogram = np.zeros(num_bins)
    # for ll in DS['train']['label']:
    #     idx = convert_array_int(ll)
    #     histogram[idx] += 1
    # 
    # print(len(np.nonzero(histogram)[0]))
    # print(np.nonzero(histogram)[0])
    # print(histogram / sum(histogram))
    # 
    # histogram = np.zeros(num_bins)
    # for ll in DS['test']['label']:
    #     idx = convert_array_int(ll)
    #     histogram[idx] += 1
    # 
    # print(len(np.nonzero(histogram)[0]))
    # print(np.nonzero(histogram)[0])
    # print(histogram / sum(histogram))
