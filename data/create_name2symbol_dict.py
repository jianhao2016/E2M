#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This file take the symbol2name pickle as input, and reverse it.
"""

import pickle
import numpy as np

def invert_dict(dict_in):
    type_test_list = [type(dict_in[key]) for key in dict_in.keys()]
    expected_list = [type([])] * len(type_test_list)

    if type_test_list == expected_list:
        dict_out = {dict_in[key][0]: key for key in dict_in.keys()}
    else:
        mis_match = [item for item in type_test_list if item not in expected_list]
        print('mismatch data type exist in original data:{}'.format(mis_match))
        dict_out = None

    return dict_out

def create_symbol_list(name_list, name2symbol_dict):
    with open(name_list, 'rb') as f:
        n_list = pickle.load(f)

    with open(name2symbol_dict, 'rb') as f:
        n2s_dcit = pickle.load(f)

    symbol_list = [n2s_dcit[gname] for gname in n_list]
    return symbol_list

if __name__ == '__main__':
    p2f = 'lm_pickle/landmark_symbol2name_dict.pickle'
    with open(p2f, 'rb') as f:
        DS = pickle.load(f)

    name2symbol_dict = invert_dict(DS)
    # print(name2symbol_dict)
    # print(len(name2symbol_dict))

    save_path = 'lm_pickle/landmark_name2symbol_dict.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(name2symbol_dict, f)

    name_list = 'lm_pickle/landmark_gene_name_list.pickle'

    symbol_list = create_symbol_list(name_list, save_path)
    # print(symbol_list)
    save_symbol_list = 'lm_pickle/landmark_gene_symbol_list.pickle'
    with open(save_symbol_list, 'wb') as f:
        pickle.dump(symbol_list, f)
