#! /bin/sh
#
# get_cases_tsv.sh
# Copyright (C) 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#


# this script get the tsv file using GDC data portal api. 
# the tsv file is a key for the data extraction.

# --- usage ---
# ./get_cases_tsv.sh <project ID>

python create_json_filter.py --pid $1 --ctype meth
python create_json_filter.py --pid $1 --ctype seq

f1='json_filter_'$1'_meth'
f2='json_filter_'$1'_seq'

f3='tmp_cases_meth.tsv'
f4='tmp_cases_seq.tsv'
# now we have 2 filters, one for meth and one for seq
curl --request POST --header "Content-Type: application/json" --data @$f1 'https://api.gdc.cancer.gov/cases' > $f3
curl --request POST --header "Content-Type: application/json" --data @$f2 'https://api.gdc.cancer.gov/cases' > $f4

rm $f1 $f2

# now we need to find the common lines/cases in meth and seq file.
comm -12 $f3 $f4 > 'cases'_$1'_all'

# get the UUID of files.
python get_files_UUID.py --p2f $f3 --saveFile $1'_meth_files_UUIDs' --data_type 'meth' --format 'tsv' --size 2000
python get_files_UUID.py --p2f $f4 --saveFile $1'_seq_files_UUIDs' --data_type 'RNAseq' --format 'tsv' --size 2000

rm $f3 $f4
