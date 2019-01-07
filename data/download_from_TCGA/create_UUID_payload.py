#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to create the payload for files download using UUID
from TCGA
curl --remote-name --remote-header-name --request POST --header 'Content-Type: application/json' --data @request.txt 'https://api.gdc.cancer.gov/legacy/data'
"""
import json
from collections import defaultdict

cases_list = []
mapping = {}

with open('cases_GBM_all', 'r') as infile:
    for line in infile:
        if line == '\n':
            continue
        case_id, _ = line.split('\n')
        cases_list.append(case_id)
        mapping[case_id] = defaultdict()

# print(cases_list)

meth_UUIDs = {"ids":[]}
ll_1 = cases_list[:]
with open('GBM_meth_files_UUIDs', 'r') as meth:
    meth.readline()
    for line in meth:
        col_list = line.split()
        if col_list[6] in ll_1:
            ll_1.remove(col_list[6])
            meth_UUIDs["ids"].append(col_list[7])
            mapping[col_list[6]]["meth_uuid"] = col_list[7]

# print(meth_UUIDs)
# print(len(meth_UUIDs["ids"]))
# print(len(cases_list))
with open('payload_files/payload_meth', 'w') as p_meth:
    json.dump(meth_UUIDs, p_meth, indent=2)


seq_UUIDs = {"ids":[]}
ll_2 = cases_list[:]
with open('GBM_seq_files_UUIDs', 'r') as seq:
    seq.readline()
    for line in seq:
        col_list = line.split()
        if col_list[6] in ll_2:
            ll_2.remove(col_list[6])
            seq_UUIDs["ids"].append(col_list[7])
            mapping[col_list[6]]["seq_uuid"] = col_list[7]

print(len(seq_UUIDs["ids"]))
with open('payload_files/payload_seq', 'w') as p_seq:
    json.dump(seq_UUIDs, p_seq, indent=2)

with open('mapping_caseID_seqID_methID_GBM', 'w') as f_mapping:
    json.dump(mapping, f_mapping, indent=2)
