#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This file just create a json filter for the file retrival.
i.e.
json filter used in ./get_cases_tsv.sh <json filter> <output file>
"""

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pid', type=str,
            help='project id')
    parser.add_argument('--ctype', type=str,
            help='data type in each cases')

    FLAGS, _ = parser.parse_known_args()
    # project_id = 'LUAD'
    # cases_type = 'meth'
    project_id = FLAGS.pid
    cases_type = FLAGS.ctype

    op_1 = single_op('=', "project.project_id", "TCGA-" + project_id)

    op_2 = single_op('=', "files.data_type", "Methylation Beta Value")

    op_3 = single_op('=', "files.data_type", "Gene Expression Quantification")

    if cases_type == 'meth':
        base_filter = and_op(op_1, op_2)
        output_file = 'json_filter_' + project_id + '_meth'
    elif cases_type == 'seq':
        base_filter = and_op(op_1, op_3)
        output_file = 'json_filter_' + project_id + '_seq'

    tsv_filter = { "filters":base_filter,
                    "format":"tsv",
                    "fields":"case_id,files.cases.project.project_id",
                    "size":"2000",
                    "sort":"case_id:asc"
                }

    gg = json.dumps(tsv_filter, indent = 4)
    with open(output_file, 'w') as f:
        f.write(gg)

