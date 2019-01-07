#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is used for getting files UUIDs from TCGA
Then we can apply gdc-client download method to them.
"""

import requests
import json
import argparse

def single_op(op, field, value):
    op_dict = {
            "op":op,
            "content":{
                "field":field,
                "value":value
                }
            }
    return op_dict

def and_op(op_1, op_2):
    and_dict = {
            "op":"and",
            "content":[
                op_1,
                op_2
                ]
            }
    return and_dict

def get_filter(p2f, DT, WT):
    id_list = []
    with open(p2f, mode='r') as f:
        f.readline()
        for line in f:
            case_id, _ = line.split()
            id_list.append(case_id)

    op_1 = single_op("in", "cases.case_id", id_list)
    op_2 = single_op("=", "files.data_type", DT)
    and_1 = and_op(op_1, op_2)

    op_3 = single_op("=", "files.analysis.workflow_type", WT)
    op_4 = single_op("=", "files.cases.samples.sample_type", ST)
    and_2 = and_op(op_3, op_4)

    base_filter = and_op(and_1, and_2)

    return base_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--p2f', type=str,
            default='GBM_cases.tsv',
            help='path to cases UUID file')
    parser.add_argument('--saveFile', type=str,
            default='GBM_meth_files_UUIDs',
            help='file to save the response.')
    parser.add_argument('--data_type', type=str,
            choices=['meth', 'RNAseq'], default='meth',
            help='data type in TCGA to be retrived.')
    parser.add_argument('--format', type=str,
            choices=['json', 'tsv', 'xml'], default='json',
            help='data format of return list')

    parser.add_argument('--size', type=int,
            default='1000', help='number of files UUID to retrived.')
    parser.add_argument('--endpoints', type=str,
            choices=['projects', 'cases', 'files'],
            default='files', help='endpoint in TCGA data api')
    
    FLAG, _ = parser.parse_known_args()
    if FLAG.data_type == 'meth':
        # DT - data type, in Data Information.
        # WT - workflow type, in Analysis
        DT = 'Methylation Beta Value'
        WT = 'Liftover'
    elif FLAG.data_type == 'RNAseq':
        DT = 'Gene Expression Quantification'
        WT = 'HTSeq - FPKM'
    
    # ST - sample type, in Associated Cases/Biospecimen
    ST = 'Primary Tumor'
    endpoint = 'https://api.gdc.cancer.gov/' + FLAG.endpoints

    filt = get_filter(FLAG.p2f, DT, WT)
    return_fields = ['cases.case_id', 'file_id', 'file_name', 
            'data_type', 'cases.samples.sample_type']
    return_fields = ','.join(return_fields)

    params = {'filters':json.dumps(filt), 'format':FLAG.format,
            'fields':return_fields, 'sort':'cases.case_id:asc',
            'size':FLAG.size}

    # request ULR-encodes automatcially.
    response = requests.post(endpoint, data = params)
    # response = requests.post(endpoint, params = params)
    # print(json.dumps(response.json(), indent=2))
    if FLAG.format == 'json':
        with open(FLAG.saveFile, 'w') as f:
            json.dump(response.json(), f, indent=2)
    else:
        with open(FLAG.saveFile, 'wb') as f:
            f.write(response.content)

